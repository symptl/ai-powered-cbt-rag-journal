"""
Fixtures for testing the refrAIme inference API.

Mocks:
- SageMaker LLM endpoint
- Bedrock Knowledge Base retriever
- LCEL chains (initial_entry_chain, chat_chain)
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Set env vars before import
os.environ["SAGEMAKER_ENDPOINT_NAME"] = "test-endpoint"
os.environ["AWS_REGION"] = "us-east-1"
os.environ["KNOWLEDGE_BASE_ID"] = "test-kb-id"

# Mock boto3 and langchain_aws before app.py imports them
mock_boto3_client = MagicMock()
patch("boto3.client", return_value=mock_boto3_client).start()

mock_sagemaker_endpoint = MagicMock()
patch("langchain_aws.SagemakerEndpoint", return_value=mock_sagemaker_endpoint).start()

from fastapi.testclient import TestClient
from langchain_core.documents import Document

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import app


@pytest.fixture
def client():
    """
    FastAPI test client with mocked LLM and retriever globals.
    Bypasses lifespan initialization by setting globals directly.
    """
    # Set up mock chains that return canned responses
    mock_initial_chain = MagicMock()
    mock_initial_chain.invoke.return_value = "That sounds like a meaningful experience. What patterns do you notice?"

    mock_chat_chain = MagicMock()
    mock_chat_chain.invoke.return_value = "I hear you. What do you think triggered that reaction?"

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(
            page_content="CBT framework: cognitive restructuring involves identifying distorted thoughts.",
            metadata={"score": 0.85, "source": "cbt-handbook.pdf"},
        )
    ]

    # Inject mocks into app globals
    app.llm = mock_sagemaker_endpoint
    app.retriever = mock_retriever
    app.initial_entry_chain = mock_initial_chain
    app.chat_chain = mock_chat_chain

    yield TestClient(app.app, raise_server_exceptions=False)

    # Reset globals
    app.llm = None
    app.retriever = None
    app.initial_entry_chain = None
    app.chat_chain = None
