"""
Test suite for the refrAIme backend API.

Covers:
- Health check
- Journal entry creation
- Entry retrieval (truncated and full)
- AI reflection (initial entry and follow-up conversation)
- Entry deletion
- Account deletion with data export
- Feedback submission
- Authorization enforcement (403 on user_id mismatch, 401 on missing token)
"""


# ============================================================================
# HEALTH CHECK
# ============================================================================

class TestHealthCheck:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


# ============================================================================
# JOURNAL ENTRY CREATION
# ============================================================================

class TestSaveEntry:
    def test_save_entry_success(self, client, auth_headers, clean_db):
        payload = {"journal_entry": "Today I felt anxious about an upcoming presentation."}
        response = client.post("/sessions/test-user-1/entry", json=payload, headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert "entry_id" in data

        # Verify document was written to MongoDB
        stored = clean_db["entries"].find_one({"entry_id": data["entry_id"]})
        assert stored is not None
        assert stored["user_id"] == "test-user-1"
        assert stored["journal_entry"] == "Today I felt anxious about an upcoming presentation."
        assert stored["ai_conversation"] == []

    def test_save_entry_forbidden_for_wrong_user(self, client, auth_headers):
        """Token is for test-user-1, but path says test-user-2."""
        payload = {"journal_entry": "Should not be saved."}
        response = client.post("/sessions/test-user-2/entry", json=payload, headers=auth_headers)
        assert response.status_code == 403

    def test_save_entry_unauthorized_without_token(self, client):
        payload = {"journal_entry": "No token provided."}
        response = client.post("/sessions/test-user-1/entry", json=payload)
        assert response.status_code == 403


# ============================================================================
# ENTRY RETRIEVAL
# ============================================================================

class TestGetEntries:
    def test_get_truncated_entries(self, client, auth_headers, seed_entries):
        response = client.get("/entries/test-user-1", headers=auth_headers)

        assert response.status_code == 200
        entries = response.json()
        assert len(entries) == 2

        # Journal text should be truncated to 100 chars + "..."
        for entry in entries:
            assert entry["journal_entry"].endswith("...")
            assert len(entry["journal_entry"]) <= 104  # 100 chars + "..."
            assert "ai_conversation" not in entry

    def test_get_full_entries(self, client, auth_headers, seed_entries):
        response = client.get("/entries/test-user-1/full", headers=auth_headers)

        assert response.status_code == 200
        entries = response.json()
        assert len(entries) == 2

        # Full entries should include ai_conversation
        entry_with_convo = next(e for e in entries if e["entry_id"] == "entry-u1-e1")
        assert len(entry_with_convo["ai_conversation"]) == 1
        assert entry_with_convo["ai_conversation"][0]["role"] == "assistant"

    def test_get_entries_forbidden_for_wrong_user(self, client, auth_headers, seed_entries):
        response = client.get("/entries/test-user-2", headers=auth_headers)
        assert response.status_code == 403

    def test_get_entries_empty_for_user_with_no_entries(self, client, auth_headers):
        response = client.get("/entries/test-user-1", headers=auth_headers)
        assert response.status_code == 200
        assert response.json() == []


# ============================================================================
# AI REFLECTION (INITIAL ENTRY)
# ============================================================================

class TestAIEntry:
    def test_ai_entry_success(self, client, auth_headers, seed_entries, mock_langchain_api):
        response = client.post("/sessions/test-user-1/entry-u1-e2", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert len(data["message"]) > 0

        # Verify LangChain API was called
        mock_langchain_api.assert_called_once()

        # Verify AI message was appended to MongoDB
        stored = seed_entries["entries"].find_one({"entry_id": "entry-u1-e2"})
        assert len(stored["ai_conversation"]) == 1
        assert stored["ai_conversation"][0]["role"] == "assistant"

    def test_ai_entry_not_found(self, client, auth_headers, seed_entries, mock_langchain_api):
        response = client.post("/sessions/test-user-1/nonexistent-entry", headers=auth_headers)
        assert response.status_code == 404


# ============================================================================
# AI CONVERSATION (FOLLOW-UP)
# ============================================================================

class TestAIConversation:
    def test_ai_convo_success(self, client, auth_headers, seed_entries, mock_langchain_api):
        payload = {"content": "I think the public setting is what hurt most."}
        response = client.post(
            "/sessions/test-user-1/messages/entry-u1-e1",
            json=payload,
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "message" in data

        # Verify both user message and AI response were appended
        stored = seed_entries["entries"].find_one({"entry_id": "entry-u1-e1"})
        messages = stored["ai_conversation"]

        # Original had 1 assistant message, now should have 3: original + user + new assistant
        assert len(messages) == 3
        assert messages[1]["role"] == "user"
        assert messages[1]["message"] == "I think the public setting is what hurt most."
        assert messages[2]["role"] == "assistant"

    def test_ai_convo_forbidden_for_wrong_user(self, client, auth_headers, seed_entries, mock_langchain_api):
        payload = {"content": "Should not work."}
        response = client.post(
            "/sessions/test-user-2/messages/entry-u1-e1",
            json=payload,
            headers=auth_headers,
        )
        assert response.status_code == 403


# ============================================================================
# ENTRY DELETION
# ============================================================================

class TestDeleteEntry:
    def test_delete_entry_success(self, client, auth_headers, seed_entries):
        response = client.delete("/entries/test-user-1/entry-u1-e1", headers=auth_headers)

        assert response.status_code == 200
        assert response.json()["ok"] is True

        # Verify entry is gone from MongoDB
        stored = seed_entries["entries"].find_one({"entry_id": "entry-u1-e1"})
        assert stored is None

    def test_delete_entry_not_found(self, client, auth_headers, seed_entries):
        response = client.delete("/entries/test-user-1/nonexistent-entry", headers=auth_headers)
        assert response.status_code == 404

    def test_delete_entry_forbidden_for_wrong_user(self, client, auth_headers, seed_entries):
        response = client.delete("/entries/test-user-2/entry-u1-e1", headers=auth_headers)
        assert response.status_code == 403


# ============================================================================
# ACCOUNT DELETION
# ============================================================================

class TestDeleteAccount:
    def test_delete_account_returns_data_export(self, client, auth_headers, seed_entries):
        response = client.delete("/users/test-user-1/delete_account", headers=auth_headers)

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        assert "attachment" in response.headers.get("content-disposition", "")

        # Response body should be the exported entries as JSON
        import json
        exported = json.loads(response.content)
        assert len(exported) == 2

        # Verify entries and user are gone from MongoDB
        assert seed_entries["entries"].count_documents({"user_id": "test-user-1"}) == 0
        assert seed_entries["users"].find_one({"user_id": "test-user-1"}) is None

    def test_delete_account_no_entries(self, client, auth_headers, clean_db):
        """User exists but has no entries."""
        clean_db["users"].insert_one({"user_id": "test-user-1"})
        response = client.delete("/users/test-user-1/delete_account", headers=auth_headers)
        assert response.status_code == 404


# ============================================================================
# FEEDBACK
# ============================================================================

class TestFeedback:
    def test_submit_feedback_success(self, client, auth_headers, seed_entries):
        payload = {"likert_1": 4, "likert_2": 5, "free_text": "Very helpful reflection."}
        response = client.post(
            "/feedback/test-user-1/entry-u1-e1",
            json=payload,
            headers=auth_headers,
        )

        assert response.status_code == 200
        assert response.json()["ok"] is True

        # Verify feedback was written to MongoDB
        stored = seed_entries["feedback"].find_one({"entry_id": "entry-u1-e1"})
        assert stored is not None
        assert stored["emotionally_supportive"] == 4
        assert stored["help_reflect"] == 5
        assert stored["helpful_relevant"] == "Very helpful reflection."

    def test_submit_feedback_forbidden_for_wrong_user(self, client, auth_headers):
        payload = {"likert_1": 3, "likert_2": 3, "free_text": "Should not save."}
        response = client.post(
            "/feedback/test-user-2/entry-u1-e1",
            json=payload,
            headers=auth_headers,
        )
        assert response.status_code == 403


# ============================================================================
# AUTHORIZATION EDGE CASES
# ============================================================================

class TestAuthorization:
    def test_expired_token_returns_401(self, client):
        """A token that expired in the past should be rejected."""
        import jwt as pyjwt

        expired_payload = {
            "user_id": "test-user-1",
            "exp": datetime(2020, 1, 1),
        }
        expired_token = pyjwt.encode(expired_payload, "test-secret-key-for-ci", algorithm="HS256")
        headers = {"Authorization": f"Bearer {expired_token}"}

        response = client.get("/entries/test-user-1", headers=headers)
        assert response.status_code == 401

    def test_invalid_token_returns_401(self, client):
        headers = {"Authorization": "Bearer this-is-not-a-valid-jwt"}
        response = client.get("/entries/test-user-1", headers=headers)
        assert response.status_code == 401


# Need this import for the expired token test
from datetime import datetime
