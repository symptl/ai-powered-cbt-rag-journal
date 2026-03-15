"""
Fixtures for testing the refrAIme backend API.

Mocks:
- MongoDB via mongomock (patched before main.py imports)
- LangChain API calls (requests.post to the inference endpoint)
- JWT secret key and algorithm for test token generation
"""

import os
import sys
import jwt
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

# Set env vars BEFORE importing the app
os.environ["SITE_DOMAIN"] = "http://localhost:3000"
os.environ["WEB_CLIENT_ID"] = "test-web-client-id"
os.environ["SECRET_KEY"] = "test-secret-key-for-ci"
os.environ["ALGORITHM"] = "HS256"
os.environ["LANGCHAIN_API_URL"] = "http://fake-langchain-api"
os.environ["MONGO_URI"] = "mongodb://localhost:27017"

# Patch MongoClient with mongomock BEFORE main.py is imported
import mongomock
patch("pymongo.MongoClient", mongomock.MongoClient).start()

from fastapi.testclient import TestClient
import main


@pytest.fixture(autouse=True)
def clean_db():
    """Clear all collections before each test so tests are isolated."""
    main.db["users"].delete_many({})
    main.db["entries"].delete_many({})
    main.db["feedback"].delete_many({})
    yield main.db


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(main.app)


@pytest.fixture
def auth_headers():
    """Valid JWT bearer token for test-user-1."""
    token = _make_token("test-user-1")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def auth_headers_user2():
    """Valid JWT bearer token for test-user-2."""
    token = _make_token("test-user-2")
    return {"Authorization": f"Bearer {token}"}


def _make_token(user_id: str) -> str:
    payload = {
        "user_id": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(hours=24),
    }
    return jwt.encode(payload, os.environ["SECRET_KEY"], algorithm=os.environ["ALGORITHM"])


@pytest.fixture
def seed_entries(clean_db):
    """Pre-seed MongoDB with test entries for test-user-1."""
    clean_db["entries"].insert_many([
        {
            "user_id": "test-user-1",
            "entry_id": "entry-u1-e1",
            "created_at": datetime(2025, 11, 10, 8, 0, 0, tzinfo=timezone.utc),
            "journal_entry": "I had a really tough day at work today. My manager gave me critical feedback in front of the whole team and I felt embarrassed and angry.",
            "ai_conversation": [
                {
                    "message_id": "msg-u1-e1-a1",
                    "role": "assistant",
                    "message": "Thank you for sharing that. What was the most difficult part for you?",
                },
            ],
        },
        {
            "user_id": "test-user-1",
            "entry_id": "entry-u1-e2",
            "created_at": datetime(2025, 11, 13, 19, 0, 0, tzinfo=timezone.utc),
            "journal_entry": "Feeling better today. I talked to my manager one-on-one and explained how the public feedback made me feel.",
            "ai_conversation": [],
        },
    ])

    clean_db["users"].insert_one({
        "user_id": "test-user-1",
        "created_at": datetime(2025, 11, 1, 10, 0, 0, tzinfo=timezone.utc),
        "last_login": datetime(2025, 11, 15, 14, 30, 0, tzinfo=timezone.utc),
    })

    return clean_db


@pytest.fixture
def mock_langchain_api():
    """Mock the requests.post call to the LangChain inference API."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "response": "That sounds like a meaningful experience. What patterns do you notice?",
        "sources": [],
    }
    mock_response.status_code = 200

    with patch("main.requests.post", return_value=mock_response) as mock_post:
        yield mock_post
