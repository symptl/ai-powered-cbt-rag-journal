# Imports
import os
import jwt
import json
from uuid import uuid4
from io import StringIO
from typing import List, Optional
from datetime import datetime, timezone, timedelta
import requests

from pydantic import BaseModel
from pymongo import MongoClient
from dotenv import load_dotenv
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.encoders import jsonable_encoder


app = FastAPI()

origins = [
    os.getenv("SITE_DOMAIN")
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Classes
class AIResponse(BaseModel):
    role: str
    message: str

class Entry(BaseModel):
    journal_entry: str
    ai_conversation: Optional[List[AIResponse]] = []

class AIMessageRequest(BaseModel):
    content: str

class Feedback(BaseModel):
    likert_1: int
    likert_2: int
    free_text: str


# Load env variables
load_dotenv()


# ======== CONFIG ==========
WEB_CLIENT_ID = os.getenv("WEB_CLIENT_ID")
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
TOKEN_EXPIRE_MINUTES = 1440
LANGCHAIN_API_URL = os.getenv("LANGCHAIN_API_URL") # added to connect to prompting API

# ======== DATABASE ==========
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["refrAIme"]
users = db["users"]
entries = db["entries"]
feedback = db["feedback"]


# ======== JWT HELPERS ==========
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

security = HTTPBearer()


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ======== HEALTH CHECKPOINT ==========
@app.get("/health")
def health():
    return {"status": "ok"}


# ======== AUTH ENDPOINTS ==========
@app.post("/auth/google") # sign-in
async def google_auth(request: Request):
    data = await request.json()
    token = data.get("id_token")

    try:
        idinfo = id_token.verify_oauth2_token(token, google_requests.Request(), WEB_CLIENT_ID)
        userid = idinfo["sub"]
        # email = idinfo.get("email")
        # name = idinfo.get("name")

        # upsert user
        user = users.find_one({"user_id": userid})
  
        if not user:
            users.insert_one({
                "user_id": userid,
                "created_at": datetime.now(timezone.utc),
                "last_login": datetime.now(timezone.utc)
            })
        else:
            users.update_one(
                {"user_id": userid},
                {"$set": {"last_login": datetime.now(timezone.utc)}}
            )

        access_token = create_access_token({"user_id": userid})

        response = JSONResponse(content={
            "message": "Login successful",
            "user": {"user_id": userid, "email": "anonymized", "name": "anonymized"},
            "access_token": access_token
        })
        response.set_cookie(
            key="access_token",
            value=access_token,
            httponly=True,
            secure=True,
            samesite="lax"
        )
        return response

    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid ID token")


@app.post("/auth/signout") # sign-out
def sign_out(response: JSONResponse):
    response = JSONResponse(content={"message": "Signed out"})
    response.delete_cookie("access_token")
    return response


# ======== JOURNAL ENTRY ENDPOINT ==========
@app.post("/sessions/{user_id}/entry") # when user saves entry without reflect with AI
def save_partial_entry(user_id: str, entry: Entry, user=Depends(verify_token)):
    if user["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    entry_id = str(uuid4())

    entry_dict = entry.dict()
    entry_dict.update({
        "user_id": user_id,
        "entry_id": entry_id,
        "created_at": datetime.now(timezone.utc)
    })

    db.entries.insert_one(entry_dict)

    return {"ok": True, "entry_id": entry_id}


# ======== AI ENDPOINTS ==========
@app.post("/sessions/{user_id}/{entry_id}") # called only with journal entry (not convo)
def ai_entry(user_id: str, entry_id: str, user=Depends(verify_token)):
    if user["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    # Retrieve entry from MongoDB
    just_entry = list(db.entries.find(
                        {"user_id": user_id, "entry_id": entry_id},
                        {"_id": 0, "entry_id": 1, "created_at": 1, "journal_entry": 1, "ai_conversation": 1}
                    ))

    if not just_entry:
        raise HTTPException(status_code=404, detail="Entry not found")

    # Send to LangChain API
    ai_response = requests.post(
        f"{LANGCHAIN_API_URL}/chat",
        json=jsonable_encoder(just_entry),
        headers={"Content-Type": "application/json"},
        timeout=300
    )

    # Extract AI reponse
    ai_response_json = ai_response.json()
    ai_response_text = ai_response_json["response"]

    # Store AI message
    ai_message = {
        "message_id": str(uuid4()),
        "role": "assistant",
        "message": ai_response_text
    }

    # Update DB entry
    db.entries.update_one(
        {"entry_id": entry_id, "user_id": user_id},
        {"$push": {"ai_conversation": ai_message}}
    )

    return {"message": ai_message["message"]}


ongoing_ai_convos = {}

@app.post("/sessions/{user_id}/messages/{entry_id}") # called when user replies back
def ai_convo(request: AIMessageRequest, user_id: str, entry_id: str, user=Depends(verify_token)):

    if user["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    # Create user message
    user_message = {
        "message_id": str(uuid4()),
        "role": "user",
        "message": request.content
    }

    # Update DB entry
    db.entries.update_one(
        {"entry_id": entry_id, "user_id": user_id},
        {"$push": {"ai_conversation": user_message}}
    )

    # Get entry, chat history from DB
    entry_and_chat = list(db.entries.find(
                        {"user_id": user_id, "entry_id": entry_id},
                        {"_id": 0, "entry_id": 1, "created_at": 1, "journal_entry": 1, "ai_conversation": 1}
                    ))

    if not entry_and_chat or len(entry_and_chat) == 0:
        raise HTTPException(status_code=404, detail="Entry not found")

    # Send to LangChain API
    ai_response = requests.post(
        f"{LANGCHAIN_API_URL}/chat",
        json=jsonable_encoder(entry_and_chat),
        headers={"Content-Type": "application/json"},
        timeout=300
    )

    ai_response_json = ai_response.json()
    ai_response_text = ai_response_json["response"]

    # Create AI message
    ai_message = {
        "message_id": str(uuid4()),
        "role": "assistant",
        "message": ai_response_text
    }

    # Update DB entry
    db.entries.update_one(
        {"entry_id": entry_id, "user_id": user_id},
        {"$push": {"ai_conversation": ai_message}}
    )

    return {"message": ai_message["message"]}


# ======== JOURNAL MEMORIES ENDPOINTS ==========
@app.get("/entries/{user_id}") # in memories - if they dont expand
def get_first_100_chars(user_id: str, user=Depends(verify_token)):

    if user["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Forbidden")
    j_entries = list(db.entries.find(
                        {"user_id": user_id},
                        {"_id": 0, "entry_id": 1, "created_at": 1, "journal_entry": 1}
                    ))

    # retrieve only first 100 chars
    for e in j_entries:
        e['journal_entry'] = e['journal_entry'][:100] + "..."

    return jsonable_encoder(j_entries)


@app.get("/entries/{user_id}/full") # in memories - if they expand, shows everything
def get_all_entries(user_id: str, user=Depends(verify_token)):
    if user["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    entries_full = list(db.entries.find(
                        {"user_id": user_id},
                        {"_id": 0, "entry_id": 1, "created_at": 1, "journal_entry": 1, "ai_conversation": 1}
                    ))
    return jsonable_encoder(entries_full)


# ======== JOURNAL DELETE ENDPOINTS ==========
@app.delete("/entries/{user_id}/{entry_id}") # delete entry
def delete_entry(user_id: str, entry_id: str, user=Depends(verify_token)):
    if user["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    result = db.entries.delete_one({"entry_id": entry_id, "user_id": user_id})

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Entry not found")

    return {"ok": True, "message": "Entry deleted successfully"}


@app.delete("/users/{user_id}/delete_account") # delete entire account
def delete_account(user_id: str, user = Depends(verify_token)):
    if user["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    user_entries = list(db.entries.find(
                        {"user_id": user_id},
                        {"_id": 0, "entry_id": 1, "created_at": 1, "journal_entry": 1, "ai_conversation": 1}
                    ))

    if not user_entries:
        raise HTTPException(status_code=404, detail="No entries found for this user.")

    db.entries.delete_many({"user_id": user_id})

    result = db.users.delete_one({"user_id": user_id})

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="User not found in db")

    data_str = json.dumps(jsonable_encoder(user_entries), indent=2)

    return StreamingResponse(
            StringIO(data_str),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=refraime_entries.json"}
        )


# ======== FEEDBACK ENDPOINT ==========
@app.post("/feedback/{user_id}/{entry_id}") # get feedback from user
def get_feedback(user_id: str, entry_id: str, feedback_entry: Feedback, user=Depends(verify_token)):
    if user["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    feedback_entry_full = {
        "user_id": user_id,
        "entry_id": entry_id,
        "emotionally_supportive": feedback_entry.likert_1,
        "help_reflect": feedback_entry.likert_2,
        "helpful_relevant": feedback_entry.free_text,
        "completed_at": datetime.now(timezone.utc)
    }

    db.feedback.insert_one(feedback_entry_full)
    return {"ok": True}
