import os
import uuid
import json
import traceback
from contextlib import contextmanager
from typing import List, Optional, Dict, Any, Tuple

import psycopg
from psycopg.rows import dict_row

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from openai_client import embed_text, generate_answer
from web_search import web_search, WebSearchError

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
ESCALATION_EMAIL = os.getenv("ESCALATION_EMAIL", "bristonlowell@gmail.com")

TOP_K = int(os.getenv("TOP_K", "8"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))
WEB_RESULTS_K = int(os.getenv("WEB_RESULTS_K", "5"))

SYSTEM_PROMPT = """You are “Vinnie’s Brain,” a helpful assistant for customers and staff."""
KB_ONLY_PROMPT = """You are Vinnie’s Brain. Use ONLY internal sources."""
WEB_FALLBACK_PROMPT = """You are Vinnie’s Brain. You may use web sources cautiously."""

app = FastAPI(title="Vinnie's Brain API", version="1.0.0")


# -------------------------
# Error handler
# -------------------------
@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "trace": traceback.format_exc()},
    )


# -------------------------
# DB
# -------------------------
@contextmanager
def db():
    conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
    try:
        yield conn
    finally:
        conn.close()


def exec_one(conn, sql, params=()):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()


def exec_all(conn, sql, params=()):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def exec_no_return(conn, sql, params=()):
    with conn.cursor() as cur:
        cur.execute(sql, params)


# -------------------------
# Models
# -------------------------
class CreateSessionRequest(BaseModel):
    channel: str = "mobile"
    mode: str = Field(default="customer", pattern="^(customer|staff)$")


class CreateSessionResponse(BaseModel):
    session_id: str


class UpdateContextRequest(BaseModel):
    airstream_year: Optional[int] = None
    category: Optional[str] = None


class ChatRequest(BaseModel):
    session_id: str
    message: str
    airstream_year: Optional[int] = None


class UsedArticle(BaseModel):
    id: str
    title: str


class ChatResponse(BaseModel):
    answer: str
    confidence: float
    used_articles: List[UsedArticle] = []
    show_escalation: bool = False
    message_id: str
    clarifying_questions: List[str] = []
    safety_flags: List[str] = []


# -------------------------
# Health
# -------------------------
@app.get("/health")
def health():
    return {"ok": True}


# -------------------------
# Sessions
# -------------------------
@app.post("/v1/sessions", response_model=CreateSessionResponse)
def create_session(req: CreateSessionRequest):
    sid = str(uuid.uuid4())
    with db() as conn:
        exec_no_return(
            conn,
            """
            INSERT INTO sessions (id, channel, mode, triage_state)
            VALUES (%s, %s, %s, %s::jsonb)
            """,
            (sid, req.channel, req.mode, json.dumps({})),
        )
        conn.commit()
    return CreateSessionResponse(session_id=sid)


@app.get("/v1/sessions/{session_id}")
def session_exists(session_id: str):
    with db() as conn:
        row = exec_one(conn, "SELECT id FROM sessions WHERE id=%s", (session_id,))
        if not row:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"ok": True}


@app.post("/v1/sessions/{session_id}/context")
def update_context(session_id: str, req: UpdateContextRequest):
    with db() as conn:
        exec_no_return(
            conn,
            """
            UPDATE sessions
            SET airstream_year=%s, category=%s
            WHERE id=%s
            """,
            (req.airstream_year, req.category, session_id),
        )
        conn.commit()
    return {"ok": True}


# -------------------------
# Chat
# -------------------------
@app.post("/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    with db() as conn:
        row = exec_one(conn, "SELECT * FROM sessions WHERE id=%s", (req.session_id,))
        if not row:
            raise HTTPException(status_code=404, detail="Session not found")

        exec_no_return(
            conn,
            """
            INSERT INTO messages (id, session_id, role, content)
            VALUES (%s, %s, %s, %s)
            """,
            (str(uuid.uuid4()), req.session_id, "user", req.message),
        )

        answer = generate_answer(
            system_prompt=SYSTEM_PROMPT,
            kb_context="",
            user_message=req.message,
        )

        msg_id = str(uuid.uuid4())
        exec_no_return(
            conn,
            """
            INSERT INTO messages (id, session_id, role, content, confidence)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (msg_id, req.session_id, "assistant", answer, 0.75),
        )

        conn.commit()

        return ChatResponse(
            answer=answer,
            confidence=0.75,
            used_articles=[],
            show_escalation=True,
            message_id=msg_id,
            clarifying_questions=[],
            safety_flags=[],
        )


# -------------------------
# Escalation
# -------------------------
@app.post("/v1/escalations")
def create_escalation(payload: Dict[str, Any]):
    ticket_id = str(uuid.uuid4())
    return {"ticket_id": ticket_id}
