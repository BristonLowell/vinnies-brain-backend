import os
import uuid
import json
import smtplib
import traceback
from contextlib import contextmanager
from email.message import EmailMessage
from typing import List, Optional, Dict, Any, Tuple

import psycopg
from psycopg.rows import dict_row

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi import Request
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from openai_client import embed_text, generate_answer
from web_search import web_search, WebSearchError

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/vinniesbrain")
ESCALATION_EMAIL = os.getenv("ESCALATION_EMAIL", "bristonlowell@gmail.com")

TOP_K = int(os.getenv("TOP_K", "8"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))
WEB_RESULTS_K = int(os.getenv("WEB_RESULTS_K", "5"))

ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")

SYSTEM_PROMPT = """You are “Vinnie’s Brain,” a helpful assistant for customers and staff.
You can answer general questions normally.
For Airstream troubleshooting questions, prioritize safety and clarity.
"""

# KB-only prompt (STRICT)
KB_ONLY_PROMPT = """
You are Vinnie’s Brain.

Rules:
- Use ONLY the provided VINNIE’S BRAIN INTERNAL SOURCES.
- If the sources don’t contain the answer, say exactly: "I couldn’t find that in Vinnie’s Brain."
- If you ask a question, ask EXACTLY ONE question.
- Any question you ask must be YES/NO only.
- Keep responses short and action-oriented. No long explanations.

Cite internal sources like: [kb:<id>]
""".strip()

# Web fallback prompt (STRICT)
WEB_FALLBACK_PROMPT = """
You are Vinnie’s Brain.

You may use the provided WEB SOURCES.
Rules:
- Clearly label anything supported by web sources as NOT from Vinnie’s Brain.
- Web info can be wrong or vary by model/year; keep a cautious tone.
- If you ask a question, ask EXACTLY ONE question.
- Any question you ask must be YES/NO only.
- Keep responses short. No long explanations.

If you use web sources, include this line near the top:
"Note: This answer includes information from the web, not from Vinnie’s Brain. Confidence: Low."

Cite web sources like: [web:<n>]
""".strip()

app = FastAPI(title="Vinnie's Brain API", version="0.3.1")


# -------------------------
# Debug exception handler
# -------------------------
@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": str(exc), "trace": traceback.format_exc()})


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


def exec_one(conn, sql: str, params=()):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()


def exec_all(conn, sql: str, params=()) -> List[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def exec_no_return(conn, sql: str, params=()):
    with conn.cursor() as cur:
        cur.execute(sql, params)


def get_session(conn, session_id: str) -> Dict[str, Any]:
    row = exec_one(conn, "SELECT * FROM sessions WHERE id=%s", (session_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    return row


def insert_message(
    conn,
    session_id: str,
    role: str,
    content: str,
    used_articles: Optional[List[Dict[str, Any]]] = None,
    confidence: Optional[float] = None,
) -> str:
    mid = str(uuid.uuid4())
    exec_no_return(
        conn,
        """
        INSERT INTO messages (id, session_id, role, content, used_articles, confidence)
        VALUES (%s, %s, %s, %s, %s::jsonb, %s)
        """,
        (mid, session_id, role, content, json.dumps(used_articles or []), confidence),
    )
    return mid


# -------------------------
# Models
# -------------------------
class CreateSessionRequest(BaseModel):
    channel: str = "mobile"
    mode: str = Field(default="customer", pattern="^(customer|staff)$")
    reset_old_session_id: Optional[str] = None
    delete_old_messages: bool = True


class CreateSessionResponse(BaseModel):
    session_id: str


class UpdateContextRequest(BaseModel):
    airstream_year: Optional[int] = Field(default=None, ge=1900, le=2100)
    category: Optional[str] = None


class ChatRequest(BaseModel):
    session_id: str
    message: str
    airstream_year: Optional[int] = Field(default=None, ge=1900, le=2100)


class UsedArticle(BaseModel):
    id: str
    title: str


class ChatResponse(BaseModel):
    answer: str
    confidence: float
    used_articles: List[UsedArticle] = []
    show_escalation: bool = False
    message_id: str
    # ✅ these match your frontend optional fields, but now they’ll actually be sent
    clarifying_questions: List[str] = []
    safety_flags: List[str] = []


# -------------------------
# Helpers
# -------------------------
def detect_safety_flags(user_text: str) -> List[str]:
    t = (user_text or "").lower()
    flags = []
    if any(k in t for k in ["active drip", "dripping", "running water", "pouring in"]):
        flags.append("active_water_intrusion")
    if any(k in t for k in ["soft floor", "soft flooring", "soft wall", "swollen wall", "spongy"]):
        flags.append("structural_moisture_risk")
    if "mold" in t or "mould" in t:
        flags.append("mold_risk")
    if any(k in t for k in ["outlet", "electrical", "breaker", "short", "spark"]):
        flags.append("electrical_risk")
    return flags


def json_list(val) -> list:
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def is_greeting(text: str) -> bool:
    t = (text or "").strip().lower()
    return t in {
        "hi", "hello", "hey", "hi there", "hey there",
        "good morning", "good afternoon", "good evening"
    }


def is_airstream_question(text: str, year: Optional[int]) -> bool:
    if year is not None:
        return True
    t = (text or "").lower()
    keywords = [
        "airstream", "trailer", "rv", "camper",
        "fresh tank", "gray tank", "black tank",
        "water pump", "city water", "water heater",
        "awning", "hitch", "tongue jack",
        "underbelly", "seal", "leak", "roof", "window",
        "12v", "battery", "inverter", "converter", "shore power",
    ]
    return any(k in t for k in keywords)


def build_kb_sources_context(rows: List[Tuple[Dict[str, Any], float]]) -> Tuple[str, List[Dict[str, Any]], float]:
    if not rows:
        return "", [], 0.0

    top_score = rows[0][1]
    used_articles = [{"id": r[0]["id"], "title": r[0]["title"]} for r in rows]

    parts = []
    for (a, score) in rows:
        parts.append(
            "\n".join(
                [
                    f"ID: {a.get('id')}",
                    f"Title: {a.get('title')}",
                    f"Category: {a.get('category')}",
                    f"Years: {a.get('years_min')}-{a.get('years_max')}",
                    f"Summary: {a.get('customer_summary')}",
                    f"Clarifying Questions: {json.dumps(json_list(a.get('clarifying_questions')))}",
                    f"Steps: {json.dumps(json_list(a.get('steps')))}",
                    f"Model Year Notes: {json.dumps(json_list(a.get('model_year_notes')))}",
                    f"Stop & Escalate: {json.dumps(json_list(a.get('stop_and_escalate')))}",
                    f"Next Step: {a.get('next_step')}",
                    f"(relevance_score={score:.3f})",
                ]
            )
        )

    context_text = "VINNIE’S BRAIN INTERNAL SOURCES:\n\n" + "\n\n---\n\n".join(parts)
    return context_text, used_articles, float(top_score)


def build_web_context(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "WEB SOURCES:\n\n(none)\n"

    parts = []
    for i, r in enumerate(results, start=1):
        parts.append(
            "\n".join(
                [
                    f"[web:{i}] Title: {r.get('title','')}",
                    f"[web:{i}] URL: {r.get('url','')}",
                    f"[web:{i}] Snippet: {r.get('content','')}",
                ]
            )
        )
    return "WEB SOURCES:\n\n" + "\n\n---\n\n".join(parts)


def enforce_one_question(answer: str) -> Tuple[str, List[str]]:
    """
    If the model output includes any question marks, we force it to ask ONLY ONE question
    and return ONLY that single question (no info dump).
    """
    a = (answer or "").strip()
    if "?" not in a:
        return a, []

    # Take only up to the first question mark
    first_q = a[: a.find("?") + 1].strip()

    # Remove leading filler lines if present and keep it clean
    # e.g. "Note: ...\n\nQuestion?" -> keep both only if the note is required (web fallback).
    # We'll keep required web note if it's present, otherwise only the question.
    lower = a.lower()
    if "includes information from the web" in lower:
        # Keep the first "Note: ..." line plus the question
        lines = [ln.strip() for ln in a.splitlines() if ln.strip()]
        note_line = None
        for ln in lines[:3]:
            if "includes information from the web" in ln.lower():
                note_line = ln
                break
        if note_line and note_line not in first_q:
            forced = note_line + "\n\n" + first_q
            return forced, [first_q]

    return first_q, [first_q]


# -------------------------
# Retrieval (pgvector)
# -------------------------
def retrieve_articles(
    conn,
    query_embedding: List[float],
    airstream_year: Optional[int],
    top_k: int,
) -> List[Tuple[Dict[str, Any], float]]:
    if airstream_year is not None:
        sql = """
          SELECT
            id, title, category, severity, years_min, years_max, customer_summary,
            clarifying_questions, steps, model_year_notes, stop_and_escalate, next_step,
            decision_tree,
            1 - (embedding <=> %s::vector) AS score
          FROM kb_articles
          WHERE embedding IS NOT NULL
            AND years_min <= %s AND years_max >= %s
          ORDER BY embedding <=> %s::vector
          LIMIT %s
        """
        params = (query_embedding, airstream_year, airstream_year, query_embedding, top_k)
    else:
        sql = """
          SELECT
            id, title, category, severity, years_min, years_max, customer_summary,
            clarifying_questions, steps, model_year_notes, stop_and_escalate, next_step,
            decision_tree,
            1 - (embedding <=> %s::vector) AS score
          FROM kb_articles
          WHERE embedding IS NOT NULL
          ORDER BY embedding <=> %s::vector
          LIMIT %s
        """
        params = (query_embedding, query_embedding, top_k)

    rows = exec_all(conn, sql, params)
    return [(r, float(r["score"])) for r in rows]


# -------------------------
# Routes
# -------------------------
@app.get("/v1/sessions/{session_id}")
def session_exists(session_id: str):
    with db() as conn:
        row = exec_one(conn, "SELECT id FROM sessions WHERE id=%s", (session_id,))
        if not row:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"ok": True}


def reset_existing_session(conn, session_id: str, delete_messages: bool = True) -> None:
    row = exec_one(conn, "SELECT id FROM sessions WHERE id=%s", (session_id,))
    if not row:
        return
    exec_no_return(
        conn,
        "UPDATE sessions SET triage_state='{}'::jsonb, airstream_year=NULL, category=NULL WHERE id=%s",
        (session_id,),
    )
    if delete_messages:
        exec_no_return(conn, "DELETE FROM messages WHERE session_id=%s", (session_id,))


@app.post("/v1/sessions", response_model=CreateSessionResponse)
def create_session(req: CreateSessionRequest):
    sid = str(uuid.uuid4())
    with db() as conn:
        if req.reset_old_session_id:
            reset_existing_session(conn, req.reset_old_session_id, delete_messages=req.delete_old_messages)
        exec_no_return(conn, "INSERT INTO sessions (id, mode, channel) VALUES (%s, %s, %s)", (sid, req.mode, req.channel))
        conn.commit()
    return CreateSessionResponse(session_id=sid)


@app.post("/v1/sessions/{session_id}/context")
def update_context(session_id: str, req: UpdateContextRequest):
    with db() as conn:
        _ = get_session(conn, session_id)
        exec_no_return(conn, "UPDATE sessions SET airstream_year=%s, category=%s WHERE id=%s", (req.airstream_year, req.category, session_id))
        conn.commit()
    return {"ok": True}


@app.post("/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    with db() as conn:
        sess = get_session(conn, req.session_id)
        insert_message(conn, req.session_id, "user", req.message)

        year = req.airstream_year or sess.get("airstream_year")
        user_text = (req.message or "").strip()

        safety_flags = detect_safety_flags(user_text)
        show_escalation = any(
            f in safety_flags for f in ["active_water_intrusion", "structural_moisture_risk", "mold_risk", "electrical_risk"]
        )

        # Friendly greeting behavior
        if is_greeting(user_text):
            answer = "Hi 👋 What can I help with?"
            msg_id = insert_message(conn, req.session_id, "assistant", answer, used_articles=[], confidence=1.0)
            conn.commit()
            return ChatResponse(
                answer=answer,
                confidence=1.0,
                used_articles=[],
                show_escalation=False,
                message_id=msg_id,
                clarifying_questions=[],
                safety_flags=[],
            )

        airstreamish = is_airstream_question(user_text, year)

        # 1) Non-Airstream: answer normally, but keep it short and single-question-safe
        if not airstreamish:
            raw = generate_answer(
                system_prompt=SYSTEM_PROMPT + "\n\nKeep it concise. If you ask a question, ask only ONE.",
                kb_context="(No internal sources used.)",
                user_message=user_text,
            )
            answer, clarifying = enforce_one_question(raw)
            msg_id = insert_message(conn, req.session_id, "assistant", answer, used_articles=[], confidence=0.85)
            conn.commit()
            return ChatResponse(
                answer=answer,
                confidence=0.85,
                used_articles=[],
                show_escalation=False,
                message_id=msg_id,
                clarifying_questions=clarifying,
                safety_flags=safety_flags,
            )

        # 2) Airstream-related: KB first
        q_emb = embed_text(user_text)
        retrieved = retrieve_articles(conn, q_emb, year, top_k=TOP_K)
        kb_context, used_articles, kb_score = build_kb_sources_context(retrieved)

        if retrieved and kb_score >= CONFIDENCE_THRESHOLD:
            raw = generate_answer(
                system_prompt=KB_ONLY_PROMPT,
                kb_context=kb_context,
                user_message=user_text,
            )
            answer, clarifying = enforce_one_question(raw)
            msg_id = insert_message(conn, req.session_id, "assistant", answer, used_articles=used_articles, confidence=kb_score)
            conn.commit()
            return ChatResponse(
                answer=answer,
                confidence=kb_score,
                used_articles=[UsedArticle(**ua) for ua in used_articles],
                show_escalation=show_escalation,
                message_id=msg_id,
                clarifying_questions=clarifying,
                safety_flags=safety_flags,
            )

        # 3) KB not confident -> Web fallback (clearly labeled + low confidence)
        web_results = []
        try:
            web_results = web_search(f"Airstream {year or ''} {user_text}".strip(), max_results=WEB_RESULTS_K)
        except WebSearchError:
            web_results = []
        except Exception:
            web_results = []

        web_context = build_web_context(web_results)

        combined_context = "\n\n".join(
            [
                "VINNIE’S BRAIN INTERNAL CHECK:\n"
                + (f"Top internal match score: {kb_score:.3f}\n" if retrieved else "No internal matches found.\n"),
                kb_context if kb_context else "(No internal sources found.)",
                web_context,
            ]
        )

        raw = generate_answer(
            system_prompt=WEB_FALLBACK_PROMPT,
            kb_context=combined_context,
            user_message=user_text,
        )

        # Force disclaimer if model forgets
        disclaimer = "Note: This answer includes information from the web, not from Vinnie’s Brain. Confidence: Low."
        if "includes information from the web" not in (raw or "").lower():
            raw = disclaimer + "\n\n" + (raw or "")

        answer, clarifying = enforce_one_question(raw)

        confidence = 0.35
        msg_id = insert_message(conn, req.session_id, "assistant", answer, used_articles=used_articles, confidence=confidence)
        conn.commit()
        return ChatResponse(
            answer=answer,
            confidence=confidence,
            used_articles=[UsedArticle(**ua) for ua in used_articles],
            show_escalation=show_escalation,
            message_id=msg_id,
            clarifying_questions=clarifying,
            safety_flags=safety_flags,
        )
