"""
Test suite for the refrAIme inference API (LangChain + SageMaker + Bedrock).

Covers:
- Health check endpoint
- unpack_json helper (conversation history parsing)
- CustomContentHandler (SageMaker input/output transforms)
- Chat endpoint: initial entry, follow-up conversation, error handling
"""

import json
from io import BytesIO
from app import unpack_json, CustomContentHandler


# ============================================================================
# HEALTH CHECK
# ============================================================================

class TestHealthCheck:
    def test_health_returns_initialized(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["initialized"] is True


# ============================================================================
# UNPACK JSON (CONVERSATION HISTORY PARSER)
# ============================================================================

class TestUnpackJson:
    def test_initial_entry_no_conversation(self):
        """First interaction: just a journal entry, no messages yet."""
        input_json = [{
            "journal_entry": "I felt anxious about my presentation today.",
            "ai_conversation": [],
            "entry_id": "entry-1",
            "created_at": "2025-11-10T08:00:00",
        }]

        journal, chat_history, prompt = unpack_json(input_json)

        assert journal == "I felt anxious about my presentation today."
        assert chat_history == ""
        assert prompt == ""

    def test_single_assistant_message(self):
        """After initial AI reflection: one assistant message, no user reply yet."""
        input_json = [{
            "journal_entry": "I felt anxious about my presentation today.",
            "ai_conversation": [
                {"message_id": "m1", "role": "assistant", "message": "What made you feel anxious?"},
            ],
            "entry_id": "entry-1",
            "created_at": "2025-11-10T08:00:00",
        }]

        journal, chat_history, prompt = unpack_json(input_json)

        assert journal == "I felt anxious about my presentation today."
        assert chat_history == ""
        # Last message becomes the prompt
        assert prompt == "What made you feel anxious?"

    def test_multi_turn_conversation(self):
        """Full conversation: assistant, user, assistant, user — last message is prompt."""
        input_json = [{
            "journal_entry": "I felt anxious about my presentation today.",
            "ai_conversation": [
                {"message_id": "m1", "role": "assistant", "message": "What made you feel anxious?"},
                {"message_id": "m2", "role": "user", "message": "I was afraid of being judged."},
                {"message_id": "m3", "role": "assistant", "message": "That fear is understandable."},
                {"message_id": "m4", "role": "user", "message": "How can I manage it?"},
            ],
            "entry_id": "entry-1",
            "created_at": "2025-11-10T08:00:00",
        }]

        journal, chat_history, prompt = unpack_json(input_json)

        assert journal == "I felt anxious about my presentation today."
        # Chat history is all messages except the last, formatted as ROLE: message
        assert "ASSISTANT: What made you feel anxious?" in chat_history
        assert "USER: I was afraid of being judged." in chat_history
        assert "ASSISTANT: That fear is understandable." in chat_history
        # Last message is NOT in chat_history
        assert "How can I manage it?" not in chat_history
        # Last message is the prompt
        assert prompt == "How can I manage it?"


# ============================================================================
# CUSTOM CONTENT HANDLER (SAGEMAKER INPUT/OUTPUT TRANSFORMS)
# ============================================================================

class TestCustomContentHandler:
    def setup_method(self):
        self.handler = CustomContentHandler()

    def test_transform_input(self):
        """Input prompt and model kwargs are serialized to JSON bytes."""
        result = self.handler.transform_input(
            "Tell me about CBT",
            {"max_new_tokens": 512, "temperature": 0.2},
        )

        parsed = json.loads(result.decode("utf-8"))
        assert parsed["inputs"] == "Tell me about CBT"
        assert parsed["parameters"]["max_new_tokens"] == 512
        assert parsed["parameters"]["temperature"] == 0.2

    def test_transform_output_list_format(self):
        """SageMaker returns a list of dicts with generated_text."""
        output_bytes = json.dumps([{"generated_text": "Here is my response."}]).encode("utf-8")
        result = self.handler.transform_output(BytesIO(output_bytes))
        assert result == "Here is my response."

    def test_transform_output_dict_format(self):
        """SageMaker returns a dict with generated_text."""
        output_bytes = json.dumps({"generated_text": "Here is my response."}).encode("utf-8")
        result = self.handler.transform_output(BytesIO(output_bytes))
        assert result == "Here is my response."

    def test_transform_output_dict_outputs_key(self):
        """Some models return under an 'outputs' key."""
        output_bytes = json.dumps({"outputs": "Here is my response."}).encode("utf-8")
        result = self.handler.transform_output(BytesIO(output_bytes))
        assert result == "Here is my response."


# ============================================================================
# CHAT ENDPOINT
# ============================================================================

class TestChatEndpoint:
    def test_initial_entry_no_conversation(self, client):
        """First AI reflection on a new journal entry (empty conversation)."""
        payload = [{
            "journal_entry": "I felt anxious about my presentation today.",
            "ai_conversation": [],
            "entry_id": "entry-1",
            "created_at": "2025-11-10T08:00:00",
        }]

        response = client.post("/chat", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert len(data["response"]) > 0
        assert "sources" in data

    def test_follow_up_conversation(self, client):
        """Follow-up turn with existing conversation history."""
        payload = [{
            "journal_entry": "I felt anxious about my presentation today.",
            "ai_conversation": [
                {"message_id": "m1", "role": "assistant", "message": "What made you feel anxious?"},
                {"message_id": "m2", "role": "user", "message": "I was afraid of being judged."},
            ],
            "entry_id": "entry-1",
            "created_at": "2025-11-10T08:00:00",
        }]

        response = client.post("/chat", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert len(data["response"]) > 0

    def test_empty_payload_returns_400(self, client):
        response = client.post("/chat", json=[])
        assert response.status_code == 400

    def test_sources_included_in_response(self, client):
        """Response should include retrieval sources from Bedrock KB."""
        payload = [{
            "journal_entry": "I felt anxious about my presentation today.",
            "ai_conversation": [],
            "entry_id": "entry-1",
            "created_at": "2025-11-10T08:00:00",
        }]

        response = client.post("/chat", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["sources"], list)
        assert len(data["sources"]) > 0
        assert "score" in data["sources"][0]
        assert "source" in data["sources"][0]
        assert "text_preview" in data["sources"][0]
