import os
import uuid
import json
import traceback
from contextlib import contextmanager
from typing import List, Optional, Dict, Any, Tuple
from urllib.parse import urlencode

import psycopg
from psycopg.rows import dict_row

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from openai_client import embed_text, generate_answer
from web_search import web_search, WebSearchError

# Prefer requests, but fall back to urllib if needed
try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore
    import urllib.request

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/vinniesbrain")

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
OWNER_SUPABASE_USER_ID = os.getenv("OWNER_SUPABASE_USER_ID", "")

EXPO_PUSH_ENABLED = os.getenv("EXPO_PUSH_ENABLED", "true").lower() == "true"

TOP_K = int(os.getenv("TOP_K", "8"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))
WEB_RESULTS_K = int(os.getenv("WEB_RESULTS_K", "5"))

SYSTEM_PROMPT = """You are “Vinnie’s Brain,” a helpful assistant for customers and staff.
You can answer general questions normally.
For Airstream troubleshooting questions, prioritize safety and clarity.
"""

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

app = FastAPI(title="Vinnie's Brain API", version="1.1.0")


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
# Supabase REST helpers (service role)
# -------------------------
def _sb_headers():
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise HTTPException(status_code=500, detail="Supabase is not configured (SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY).")

    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Prefer": "return=representation",
    }


def sb_get(table: str, query: Dict[str, str]):
    url = f"{SUPABASE_URL}/rest/v1/{table}?{urlencode(query)}"
    headers = _sb_headers()
    if requests:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"Supabase GET {table} failed: {r.status_code} {r.text}")
        return r.json()
    else:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))


def sb_post(table: str, payload: Any):
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    headers = _sb_headers()
    data = json.dumps(payload).encode("utf-8")

    if requests:
        r = requests.post(url, headers=headers, data=data, timeout=15)
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"Supabase POST {table} failed: {r.status_code} {r.text}")
        return r.json()
    else:
        req = urllib.request.Request(url, headers=headers, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))


def sb_upsert(table: str, payload: Any, on_conflict: str):
    url = f"{SUPABASE_URL}/rest/v1/{table}?on_conflict={on_conflict}"
    headers = _sb_headers()
    data = json.dumps(payload).encode("utf-8")

    if requests:
        r = requests.post(url, headers=headers, data=data, timeout=15)
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"Supabase UPSERT {table} failed: {r.status_code} {r.text}")
        return r.json()
    else:
        req = urllib.request.Request(url, headers=headers, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))


# -------------------------
# Expo push
# -------------------------
def send_expo_push(to_token: str, title: str, body: str, data: Optional[dict] = None) -> None:
    if not EXPO_PUSH_ENABLED:
        return
    if not to_token:
        return

    url = "https://exp.host/--/api/v2/push/send"
    payload = {
        "to": to_token,
        "title": title,
        "body": body,
        "data": data or {},
        "sound": "default",
    }

    if requests:
        r = requests.post(url, headers={"Content-Type": "application/json", "Accept": "application/json"}, json=payload, timeout=15)
        # Don't hard-fail your app flow if push fails
        if r.status_code >= 400:
            print("Expo push failed:", r.status_code, r.text)
    else:
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, headers={"Content-Type": "application/json", "Accept": "application/json"}, data=data_bytes, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                _ = resp.read()
        except Exception as e:
            print("Expo push failed:", str(e))


def get_owner_push_token(owner_id: str) -> Optional[str]:
    if not owner_id:
        return None
    rows = sb_get("owner_push_tokens", {"owner_id": f"eq.{owner_id}", "select": "expo_push_token", "limit": "1"})
    if isinstance(rows, list) and rows:
        return rows[0].get("expo_push_token")
    return None


# -------------------------
# triage_state (cached web follow-up)
# -------------------------
def get_triage_state(conn, session_id: str) -> dict:
    row = exec_one(conn, "SELECT triage_state FROM sessions WHERE id=%s", (session_id,))
    if not row:
        return {}
    val = row.get("triage_state") if isinstance(row, dict) else row[0]
    return val or {}


def set_triage_state(conn, session_id: str, state: dict) -> None:
    exec_no_return(
        conn,
        "UPDATE sessions SET triage_state=%s::jsonb WHERE id=%s",
        (json.dumps(state or {}), session_id),
    )


# -------------------------
# Conversation history (so yes/no follow-ups make sense)
# -------------------------
def get_recent_messages(conn, session_id: str, limit: int = 18) -> List[Dict[str, Any]]:
    return exec_all(
        conn,
        """
        SELECT role, content
        FROM messages
        WHERE session_id=%s
        ORDER BY created_at DESC
        LIMIT %s
        """,
        (session_id, limit),
    )


def format_conversation(conn, session_id: str, limit: int = 18) -> str:
    rows = list(reversed(get_recent_messages(conn, session_id, limit)))
    lines: List[str] = []
    for r in rows:
        role = (r.get("role") or "").strip()
        content = (r.get("content") or "").strip()
        if not role or not content:
            continue
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


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
    clarifying_questions: List[str] = []
    safety_flags: List[str] = []


class LiveChatSendRequest(BaseModel):
    session_id: str
    body: str


class OwnerPushTokenRequest(BaseModel):
    owner_id: str
    expo_push_token: str


# -------------------------
# Helpers
# -------------------------
def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


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


def is_greeting(text: str) -> bool:
    t = (text or "").strip().lower()
    return t in {"hi", "hello", "hey", "hi there", "hey there", "good morning", "good afternoon", "good evening"}


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


def parse_yes_no(text: str) -> Optional[str]:
    t = (text or "").strip().lower()
    yes = {"yes", "y", "yeah", "yep", "true", "correct"}
    no = {"no", "n", "nope", "false", "incorrect"}
    if t in yes:
        return "yes"
    if t in no:
        return "no"
    return None


def is_simple_followup(text: str) -> bool:
    t = (text or "").strip().lower()
    return parse_yes_no(t) in {"yes", "no"} or t in {"skip", "not sure", "unsure", "idk"}


def should_show_help_for_airstream(airstreamish: bool, safety: bool, confidence: float, web_mode: bool = False) -> bool:
    # HARD-LOCK: only show help for Airstream issues
    if not airstreamish:
        return False
    return safety or web_mode or (confidence < CONFIDENCE_THRESHOLD)


def enforce_one_question(answer: str) -> Tuple[str, List[str]]:
    a = (answer or "").strip()
    if "?" not in a:
        return a, []
    first_q = a[: a.find("?") + 1].strip()

    lower = a.lower()
    if "includes information from the web" in lower:
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


# -------------------------
# Retrieval (pgvector)
# -------------------------
def retrieve_articles(conn, query_embedding: List[float], airstream_year: Optional[int], top_k: int) -> List[Tuple[Dict[str, Any], float]]:
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
    return [(r, clamp01(float(r["score"]))) for r in rows]


# -------------------------
# Supabase live chat logic
# -------------------------
def get_or_create_conversation_for_session(session_id: str) -> str:
    # session_id is a UUID string; we reuse it as customer_id in supabase
    rows = sb_get(
        "conversations",
        {
            "select": "id",
            "customer_id": f"eq.{session_id}",
            "status": "eq.open",
            "order": "created_at.desc",
            "limit": "1",
        },
    )
    if isinstance(rows, list) and rows:
        return rows[0]["id"]

    payload = [{
        "customer_id": session_id,
        "assigned_owner_id": OWNER_SUPABASE_USER_ID or None,
        "status": "open",
    }]
    created = sb_post("conversations", payload)
    if not created or not isinstance(created, list):
        raise HTTPException(status_code=502, detail="Failed to create supabase conversation.")
    return created[0]["id"]


def supabase_insert_message(conversation_id: str, sender_id: str, sender_role: str, body: str) -> dict:
    payload = [{
        "conversation_id": conversation_id,
        "sender_id": sender_id,
        "sender_role": sender_role,
        "body": body,
    }]
    created = sb_post("messages", payload)
    if not created or not isinstance(created, list):
        raise HTTPException(status_code=502, detail="Failed to insert supabase message.")
    return created[0]


# -------------------------
# Routes
# -------------------------
@app.get("/health")
def health():
    return {"ok": True, "service": "vinnies-brain-backend"}


@app.post("/v1/sessions", response_model=CreateSessionResponse)
def create_session(req: CreateSessionRequest):
    sid = str(uuid.uuid4())
    with db() as conn:
        exec_no_return(
            conn,
            """
            INSERT INTO sessions (id, channel, mode, airstream_year, category, triage_state)
            VALUES (%s, %s, %s, NULL, NULL, %s::jsonb)
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
        get_session(conn, session_id)
        exec_no_return(
            conn,
            "UPDATE sessions SET airstream_year=%s, category=%s WHERE id=%s",
            (req.airstream_year, req.category, session_id),
        )
        conn.commit()
    return {"ok": True}


@app.post("/v1/owner/push-token")
def register_owner_push_token(req: OwnerPushTokenRequest):
    # Upsert owner push token into supabase
    # Requires table owner_push_tokens
    _ = sb_upsert(
        "owner_push_tokens",
        [{
            "owner_id": req.owner_id,
            "expo_push_token": req.expo_push_token,
        }],
        on_conflict="owner_id",
    )
    return {"ok": True}


@app.post("/v1/livechat/send")
def livechat_send(req: LiveChatSendRequest):
    # Creates a supabase conversation if needed and inserts a customer message.
    conversation_id = get_or_create_conversation_for_session(req.session_id)
    msg = supabase_insert_message(conversation_id, req.session_id, "customer", req.body)

    # Push notify owner (if token exists)
    token = get_owner_push_token(OWNER_SUPABASE_USER_ID) if OWNER_SUPABASE_USER_ID else None
    if token:
        send_expo_push(
            token,
            title="New chat message",
            body=req.body[:120],
            data={"conversation_id": conversation_id, "session_id": req.session_id},
        )

    return {"ok": True, "conversation_id": conversation_id, "message": msg}


@app.get("/v1/livechat/history/{session_id}")
def livechat_history(session_id: str):
    # Return last messages for that session's open conversation
    conv_id = get_or_create_conversation_for_session(session_id)
    rows = sb_get(
        "messages",
        {
            "select": "id,conversation_id,sender_role,body,created_at",
            "conversation_id": f"eq.{conv_id}",
            "order": "created_at.asc",
            "limit": "200",
        },
    )
    return {"conversation_id": conv_id, "messages": rows or []}

@app.get("/v1/debug/tavily")
def debug_tavily():
    try:
        r = web_search("Airstream winterize", max_results=2)
        return {"ok": True, "count": len(r), "sample": r[:1]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})



@app.post("/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    with db() as conn:
        sess = get_session(conn, req.session_id)
        insert_message(conn, req.session_id, "user", req.message)

        year = req.airstream_year or sess.get("airstream_year")
        user_text = (req.message or "").strip()

        # Greeting
        if is_greeting(user_text):
            answer = "Hi 👋 What can I help with?"
            conf = 1.0
            msg_id = insert_message(conn, req.session_id, "assistant", answer, used_articles=[], confidence=conf)
            conn.commit()
            return ChatResponse(
                answer=answer,
                confidence=conf,
                used_articles=[],
                show_escalation=False,
                message_id=msg_id,
                clarifying_questions=[],
                safety_flags=[],
            )

        airstreamish = is_airstream_question(user_text, year)

        safety_flags = detect_safety_flags(user_text) if airstreamish else []
        safety_escalation = any(
            f in safety_flags for f in ["active_water_intrusion", "structural_moisture_risk", "mold_risk", "electrical_risk"]
        )

        # Cached web follow-up (no web_search)
        state = get_triage_state(conn, req.session_id) or {}
        if state.get("web_followup_active") and is_simple_followup(user_text):
            cached_combined = state.get("cached_combined_context") or ""
            cached_used = state.get("cached_used_articles") or []
            cached_conf = clamp01(float(state.get("cached_confidence") or 0.35))

            conversation = format_conversation(conn, req.session_id, limit=18)
            raw = generate_answer(
                system_prompt=WEB_FALLBACK_PROMPT,
                kb_context=cached_combined + "\n\nCONVERSATION:\n" + conversation,
                user_message=user_text,
            )
            answer, clarifying = enforce_one_question(raw)

            if clarifying:
                state["web_followup_active"] = True
                state["last_clarifying_question"] = clarifying[0]
            else:
                state = {}

            set_triage_state(conn, req.session_id, state)

            msg_id = insert_message(conn, req.session_id, "assistant", answer, used_articles=cached_used, confidence=cached_conf)
            conn.commit()
            return ChatResponse(
                answer=answer,
                confidence=cached_conf,
                used_articles=[UsedArticle(**ua) for ua in cached_used] if cached_used else [],
                show_escalation=should_show_help_for_airstream(airstreamish, safety_escalation, cached_conf, web_mode=True),
                message_id=msg_id,
                clarifying_questions=clarifying,
                safety_flags=safety_flags,
            )

        # If new message, clear cached follow-up
        if state.get("web_followup_active") and not is_simple_followup(user_text):
            set_triage_state(conn, req.session_id, {})

        # Non-Airstream: normal answer, NEVER show Request Help
        if not airstreamish:
            conversation = format_conversation(conn, req.session_id, limit=18)
            raw = generate_answer(
                system_prompt=SYSTEM_PROMPT + "\n\nKeep it concise. If you ask a question, ask only ONE.",
                kb_context="CONVERSATION:\n" + conversation + "\n\n(No internal sources used.)",
                user_message=user_text,
            )
            answer, clarifying = enforce_one_question(raw)
            conf = 0.85
            msg_id = insert_message(conn, req.session_id, "assistant", answer, used_articles=[], confidence=conf)
            conn.commit()
            return ChatResponse(
                answer=answer,
                confidence=conf,
                used_articles=[],
                show_escalation=False,
                message_id=msg_id,
                clarifying_questions=clarifying,
                safety_flags=[],
            )

        # Airstream: KB first
        q_emb = embed_text(user_text)
        retrieved = retrieve_articles(conn, q_emb, year, top_k=TOP_K)
        kb_context, used_articles, kb_score = build_kb_sources_context(retrieved)
        kb_score = clamp01(kb_score)

        if retrieved and kb_score >= CONFIDENCE_THRESHOLD:
            conversation = format_conversation(conn, req.session_id, limit=18)
            raw = generate_answer(
                system_prompt=KB_ONLY_PROMPT,
                kb_context=kb_context + "\n\nCONVERSATION:\n" + conversation,
                user_message=user_text,
            )
            answer, clarifying = enforce_one_question(raw)

            msg_id = insert_message(conn, req.session_id, "assistant", answer, used_articles=used_articles, confidence=kb_score)
            conn.commit()
            return ChatResponse(
                answer=answer,
                confidence=kb_score,
                used_articles=[UsedArticle(**ua) for ua in used_articles],
                show_escalation=should_show_help_for_airstream(True, safety_escalation, kb_score, web_mode=False),
                message_id=msg_id,
                clarifying_questions=clarifying,
                safety_flags=safety_flags,
            )

        # Airstream: web fallback
        web_results: List[Dict[str, Any]] = []
        try:
            web_results = web_search(f"Airstream {year or ''} {user_text}".strip(), max_results=WEB_RESULTS_K)
        except WebSearchError as e:
            print("web_search failed:", str(e))
            web_results = []
        except Exception as e:
            print("web_search unexpected error:", str(e))
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

        conversation = format_conversation(conn, req.session_id, limit=18)

        raw = generate_answer(
            system_prompt=WEB_FALLBACK_PROMPT,
            kb_context=combined_context + "\n\nCONVERSATION:\n" + conversation,
            user_message=user_text,
        )

        disclaimer = "Note: This answer includes information from the web, not from Vinnie’s Brain. Confidence: Low."
        if "includes information from the web" not in (raw or "").lower():
            raw = disclaimer + "\n\n" + (raw or "")

        answer, clarifying = enforce_one_question(raw)
        confidence = 0.35

        if clarifying:
            set_triage_state(conn, req.session_id, {
                "web_followup_active": True,
                "cached_combined_context": combined_context,
                "cached_used_articles": used_articles,
                "cached_confidence": confidence,
                "last_clarifying_question": clarifying[0],
            })
        else:
            set_triage_state(conn, req.session_id, {})

        msg_id = insert_message(conn, req.session_id, "assistant", answer, used_articles=used_articles, confidence=confidence)
        conn.commit()
        return ChatResponse(
            answer=answer,
            confidence=confidence,
            used_articles=[UsedArticle(**ua) for ua in used_articles],
            show_escalation=should_show_help_for_airstream(True, safety_escalation, confidence, web_mode=True),
            message_id=msg_id,
            clarifying_questions=clarifying,
            safety_flags=safety_flags,
        )


@app.post("/v1/escalations")
def create_escalation(payload: Dict[str, Any]):
    # Keep as simple ticket creation (you can later email/store)
    ticket_id = str(uuid.uuid4())
    return {"ticket_id": ticket_id}
    
