import json
import time
import re
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
import boto3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# LangChain AWS integrations
from langchain_aws import SagemakerEndpoint
from langchain_aws.llms.sagemaker_endpoint import LLMContentHandler

# LangChain core components - Modern LCEL approach
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.output_parsers import StrOutputParser

# Import separate config
import config

# ============================================================================
# GLOBAL STATE & LIFESPAN
# ============================================================================

llm = None
retriever = None
chat_chain = None
initial_entry_chain = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, retriever, chat_chain, initial_entry_chain
    
    print("--- STARTUP: Initializing ---")
    
    try:
        # Initialize SageMaker LLM
        llm = SagemakerEndpoint(
            endpoint_name=config.SAGEMAKER_ENDPOINT_NAME,
            region_name=config.REGION_NAME,
            model_kwargs={
                "max_new_tokens": config.MAX_NEW_TOKENS,
                "temperature": config.TEMPERATURE,
                "top_p": config.TOP_P,
                "do_sample": True
            },
            content_handler=CustomContentHandler()
        )

        # Initialize Bedrock Knowledge Base Retriever
        retriever = BedrockKnowledgeBaseRetriever(
            knowledge_base_id=config.KNOWLEDGE_BASE_ID,
            region_name=config.REGION_NAME,
            num_results=config.NUM_RETRIEVAL_RESULTS,
            llm=llm,
            summarization_prompt=config.SUMMARIZATION_PROMPT,
            enable_summarization=config.ENABLE_SUMMARIZATION
        )
        
        # 3. BUILD CHAINS ONCE HERE
        
        # --- Chain A: Initial Entry ---
        initial_prompt = ChatPromptTemplate.from_messages([
            # MERGED: System prompt + User input combined
            ("user", f"""{config.SYSTEM_PROMPT}
            
            USER JOURNAL ENTRY BEGIN: {{journal_entry}} USER JOURNAL ENTRY END
            CBT FRAMEWORKS BEGIN: {{context}} CBT FRAMEWORKS END
            
            Analyze the USER JOURNAL ENTRY according to system prompt instructions.""")
        ])
        initial_entry_chain = initial_prompt | llm | StrOutputParser()

        # --- Chain B: Conversation ---
        chat_prompt = ChatPromptTemplate.from_messages([
            # MERGED: System prompt + User input combined
            ("user", f"""{config.SYSTEM_PROMPT}
            
            USER JOURNAL ENTRY BEGIN: {{journal_entry}} USER JOURNAL ENTRY END
            CHAT HISTORY BEGIN: {{chat_history}} CHAT HISTORY END
            CBT FRAMEWORKS BEGIN: {{context}} CBT FRAMEWORKS END
            
            USER RESPONSE BEGIN: {{prompt}} USER RESPONSE END""")
        ])
        chat_chain = chat_prompt | llm | StrOutputParser()
        
        print("--- STARTUP: Initialization Complete ---")
        
    except Exception as e:
        print(f"--- CRITICAL ERROR: {e} ---")
    
    yield

# Initialize FastAPI with lifespan
app = FastAPI(title="LangChain API", lifespan=lifespan)

# ============================================================================
# Pydantic Models (Input Validation)
# ============================================================================

class Message(BaseModel):
    message_id: str
    role: str
    message: str

class JournalEntry(BaseModel):
    journal_entry: str
    ai_conversation: List[Message]
    entry_id: str
    created_at: str

# ============================================================================
# LOGIC FROM NOTEBOOK
# ============================================================================

class CustomContentHandler(LLMContentHandler):
    """
    Content handler for transforming input/output to/from SageMaker endpoint.
    """
    
    content_type = "application/json"
    accepts = "application/json"
    
    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        """Transform input prompt to endpoint format."""
        input_str = json.dumps({
            "inputs": prompt,
            "parameters": model_kwargs
        })
        return input_str.encode("utf-8")
    
    def transform_output(self, output: bytes) -> str:
        """Transform endpoint output to string."""
        response_json = json.loads(output.read().decode("utf-8"))
        
        # Handle different response formats
        if isinstance(response_json, list) and len(response_json) > 0:
            if isinstance(response_json[0], dict):
                return response_json[0].get("generated_text", "")
            return str(response_json[0])
        elif isinstance(response_json, dict):
            if "generated_text" in response_json:
                return response_json["generated_text"]
            elif "outputs" in response_json:
                return response_json["outputs"]
            elif "generated_texts" in response_json:
                return response_json["generated_texts"][0]
        
        return str(response_json)

class BedrockKnowledgeBaseRetriever(BaseRetriever):
    """
    Custom retriever that queries AWS Bedrock Knowledge Base.
    """
    
    knowledge_base_id: str
    region_name: str = "us-east-1"
    num_results: int = 5
    llm: Any = None
    summarization_prompt: str = ""
    enable_summarization: bool = True
    _client: Any = None
    _summary_chain: Any = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize Bedrock client
        self._client = boto3.client(
            "bedrock-agent-runtime",
            region_name=self.region_name
        )
        
        # Initialize summarization chain using modern LCEL
        if self.enable_summarization and self.llm and self.summarization_prompt:
            prompt = ChatPromptTemplate.from_messages([
                ("user", f"{self.summarization_prompt} USER INPUT BEGIN {{query}} USER INPUT END")
            ])
            # Modern LCEL chain with pipe operator
            self._summary_chain = prompt | self.llm | StrOutputParser()
    
    def _summarize_query(self, query: str) -> str:
        """
        Summarize query using modern LCEL chain.
        """
        if not self.enable_summarization or not self._summary_chain:
            return query
        
        try:
            # Invoke the LCEL chain
            summary = self._summary_chain.invoke({"query": query})
            return summary
        except Exception as e:
            print(f"Warning: Summarization failed, using original query. Error: {e}")
            return query
    
    def _get_relevant_documents(
        self, 
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Retrieve relevant documents from Bedrock Knowledge Base.
        """
        # Handle empty prompt case to avoid Bedrock API errors
        if not query or not query.strip():
            return []

        try:
            # Summarize query before retrieval
            search_query = self._summarize_query(query)
            
            # Fallback if summarization fails
            if not search_query or not search_query.strip():
                print("Warning: Summarization returned empty string. Falling back to original query.")
                search_query = query

            # Query Bedrock Knowledge Base
            response = self._client.retrieve(
                knowledgeBaseId=self.knowledge_base_id,
                retrievalQuery={"text": search_query},
                retrievalConfiguration={
                    "vectorSearchConfiguration": {
                        "numberOfResults": self.num_results
                    }
                }
            )
            
            # Convert results to LangChain Documents
            documents = []
            for result in response.get("retrievalResults", []):
                content = result.get("content", {})
                text = content.get("text", "")
                
                metadata = {
                    "score": result.get("score", 0.0),
                    "source": result.get("metadata", {}).get(
                        "x-amz-bedrock-kb-source-uri", 
                        "Unknown"
                    )
                }
                
                if "location" in result:
                    location = result["location"]
                    if "webLocation" in location:
                        metadata["url"] = location["webLocation"].get("url", "")
                    if "s3Location" in location:
                        metadata["s3_uri"] = location["s3Location"].get("uri", "")
                
                documents.append(
                    Document(
                        page_content=text,
                        metadata=metadata
                    )
                )
            
            return documents
            
        except Exception as e:
            print(f"Error retrieving from Bedrock Knowledge Base: {e}")
            return []

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def unpack_json(input_json: list):
    """
    Unpacks conversation history JSON.
    """
    # Note: In the API context, input_json is a list of dicts (converted from Pydantic)
    entry_string = input_json[0]["journal_entry"]
    messages = input_json[0]["ai_conversation"]
    
    chat_history_string = ""
    prompt_string = ""
    
    # Only process messages if the list is not empty
    if messages:
        for message in messages[:-1]:
            chat_history_string += message["role"].upper() + ": "
            chat_history_string += message["message"] + "\n"
        
        prompt_string = messages[-1]["message"]
    
    return entry_string, chat_history_string, prompt_string

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "app": "SageMaker-Bedrock-Chatbot",
        "initialized": llm is not None and retriever is not None
    }

@app.post("/chat")
def chat_endpoint(payload: List[JournalEntry]):
    """
    Endpoint to process chat requests. 
    Replaces prompt_model() from notebook but keeps internal logic.
    """

    global llm, retriever

    if not payload:
        raise HTTPException(status_code=400, detail="Payload cannot be empty list")

    # Check if initialization succeeded
    if llm is None or retriever is None or initial_entry_chain is None or chat_chain is None:
         raise HTTPException(status_code=503, detail="System initializing...")

    # Convert Pydantic models to list of dicts to match notebook logic
    input_json = [entry.model_dump() for entry in payload]
    
    journal_entry, chat_history, prompt = unpack_json(input_json)
    
    # Determine the context and select the correct chat chain
    # Check if chat_history is empty, AND if prompt is empty/only contains a role label.
    is_initial_entry = not chat_history.strip() and not prompt.strip().endswith(": ")
    
    # Determine retrieval query and LLM input variables:
    if is_initial_entry:
        active_chain = initial_entry_chain
        retrieval_query = journal_entry
        chain_inputs = {
            "journal_entry": journal_entry,
            # No chat_history or prompt needed for this template
        }
    else:
        active_chain = chat_chain
        retrieval_query = prompt
        chain_inputs = {
            "journal_entry": journal_entry,
            "chat_history": chat_history,
            "prompt": prompt
        }

    # Invoke retriever
    docs = retriever.invoke(retrieval_query)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Add context to the chain_inputs dictionary
    chain_inputs["context"] = context

    for attempt in range(config.MAX_RETRIES):
        try:
            # Invoke the LCEL chain
            answer = active_chain.invoke(chain_inputs)
            
            # Validate answer
            if answer and len(answer.strip()) > 0:
                # Use regex to strip anything before the first ": " (inclusive)
                # This removes "Assistant: " or "AI: " labels if generated
                cleaned_answer = re.sub(r'^.*?: ', '', answer.strip(), count=1)

                # Construct sources list for response
                sources_list = []
                for doc in docs:
                    sources_list.append({
                        "score": doc.metadata.get("score", 0),
                        "source": doc.metadata.get("source", "Unknown"),
                        "text_preview": doc.page_content[:200]
                    })
                
                return {
                    "response": cleaned_answer,
                    "sources": sources_list
                }
            
            # Empty answer - retry
            if attempt < config.MAX_RETRIES - 1:
                time.sleep(config.RETRY_DELAY)
                
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt < config.MAX_RETRIES - 1:
                time.sleep(config.RETRY_DELAY)
    
    raise HTTPException(status_code=500, detail=f"Failed to get valid answer after {config.MAX_RETRIES} attempts")