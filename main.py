import os
import uuid
import json
import traceback
from contextlib import contextmanager
from typing import List, Optional, Dict, Any, Tuple
from urllib.parse import urlencode

import psycopg
from psycopg.rows import dict_row

from fastapi import FastAPI, HTTPException, Request, Header
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
OWNER_SUPABASE_USER_ID = os.getenv("OWNER_SUPABASE_USER_ID", "").strip()

ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", os.getenv("ADMIN_KEY", "")).strip()

# =========================
# App + DB helpers
# =========================
app = FastAPI()


@contextmanager
def db():
    conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
    try:
        yield conn
    finally:
        conn.close()


def exec_one(conn, sql: str, params: Tuple = ()):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()


def exec_all(conn, sql: str, params: Tuple = ()):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def exec_no_return(conn, sql: str, params: Tuple = ()):
    with conn.cursor() as cur:
        cur.execute(sql, params)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


# =========================
# Supabase REST helpers
# =========================
def _sb_headers(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise HTTPException(status_code=500, detail="Supabase is not configured (SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY).")
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
    }
    if extra:
        headers.update(extra)
    return headers


def sb_get(table: str, params: Dict[str, str]) -> Any:
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    qs = urlencode(params)
    full = f"{url}?{qs}" if qs else url

    if requests is not None:
        r = requests.get(full, headers=_sb_headers(), timeout=15)
        if not r.ok:
            raise HTTPException(status_code=502, detail=f"Supabase GET {table} failed: {r.status_code} {r.text}")
        return r.json()
    else:
        req = urllib.request.Request(full, headers=_sb_headers(), method="GET")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))


def sb_post(table: str, payload: Any, prefer: str = "return=representation") -> Any:
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    headers = _sb_headers({"Prefer": prefer})

    if requests is not None:
        r = requests.post(url, headers=headers, json=payload, timeout=15)
        if not r.ok:
            raise HTTPException(status_code=502, detail=f"Supabase POST {table} failed: {r.status_code} {r.text}")
        return r.json()
    else:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, headers=headers, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))


def sb_upsert(table: str, payload: Any, on_conflict: str):
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    headers = _sb_headers({"Prefer": "resolution=merge-duplicates,return=representation"})
    if on_conflict:
        url = f"{url}?on_conflict={on_conflict}"

    if requests is not None:
        r = requests.post(url, headers=headers, json=payload, timeout=15)
        if not r.ok:
            raise HTTPException(status_code=502, detail=f"Supabase UPSERT {table} failed: {r.status_code} {r.text}")
        return r.json()
    else:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, headers=headers, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))


# =========================
# Models
# =========================
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
    clarifying_questions: List[str]
    safety_flags: List[str]
    confidence: float
    used_articles: List[UsedArticle]
    show_escalation: bool
    message_id: str


class EscalationRequest(BaseModel):
    session_id: str
    name: str
    phone: str
    email: str
    message: str
    preferred_contact: Optional[str] = "text"
    reset_old: bool = False


class EscalationResponse(BaseModel):
    ticket_id: str


class LiveChatSendRequest(BaseModel):
    session_id: str
    body: str


class AdminLiveChatSendRequest(BaseModel):
    conversation_id: str
    body: str


class OwnerPushTokenRequest(BaseModel):
    owner_id: str
    expo_push_token: str


# =========================
# Safety helpers
# =========================
def detect_safety_flags(user_text: str) -> List[str]:
    t = (user_text or "").lower()
    flags = []
    if any(k in t for k in ["propane", "gas leak", "smell gas", "carbon monoxide", "co2 alarm", "co alarm"]):
        flags.append("gas_or_co_risk")
    if any(k in t for k in ["fire", "smoke", "burning", "flames"]):
        flags.append("fire_risk")
    if any(k in t for k in ["mold", "black mold"]):
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
    return any(k in t for k in ["airstream", "trailer", "rv", "camping", "leveling", "leak", "water", "furnace", "ac"])


def require_admin(x_admin_key: str) -> None:
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=500, detail="ADMIN_API_KEY is not set on the server.")
    if (x_admin_key or "").strip() != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


# =========================
# DB: sessions + chat log
# =========================
def get_session(conn, session_id: str) -> Dict[str, Any]:
    row = exec_one(conn, "SELECT * FROM sessions WHERE id=%s", (session_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    return row


def log_message(conn, session_id: str, role: str, text: str):
    exec_no_return(
        conn,
        "INSERT INTO chat_messages (id, session_id, role, content) VALUES (%s, %s, %s, %s)",
        (str(uuid.uuid4()), session_id, role, text),
    )


def get_recent_messages(conn, session_id: str, limit: int = 20):
    return exec_all(
        conn,
        "SELECT role, content AS text FROM chat_messages WHERE session_id=%s ORDER BY created_at ASC LIMIT %s",
        (session_id, limit),
    )



# =========================
# RAG helpers (existing)
# =========================
def embed(text: str) -> List[float]:
    return embed_text(text)


def rank_kb_articles(conn, query_embedding: List[float], year: Optional[int], category: Optional[str], top_k: int = 6):
    # ✅ FIX: your kb_articles table doesn't have "body"; use customer_summary as body
    sql = """
    SELECT id, title, customer_summary AS body,
           1 - (embedding <=> %s::vector) AS score
      FROM kb_articles
     WHERE 1=1
    """
    params: List[Any] = [query_embedding]

    if year is not None:
        sql += " AND years_min <= %s AND years_max >= %s"
        params.extend([year, year])

    if category:
        sql += " AND category = %s"
        params.append(category)

    sql += " ORDER BY embedding <=> %s::vector ASC LIMIT %s"
    params.append(query_embedding)
    params.append(top_k)

    rows = exec_all(conn, sql, tuple(params))
    return [(r, clamp01(float(r["score"]))) for r in rows]


# =========================
# Supabase live chat logic
# =========================
def get_or_create_conversation_for_session(session_id: str) -> str:
    rows = sb_get(
        "conversations",
        {"select": "id", "customer_id": f"eq.{session_id}", "limit": "1"},
    )
    if rows and isinstance(rows, list) and rows[0].get("id"):
        return rows[0]["id"]

    created = sb_post(
        "conversations",
        [{"customer_id": session_id}],
    )
    if not created or not isinstance(created, list) or not created[0].get("id"):
        raise HTTPException(status_code=502, detail="Failed to create supabase conversation.")
    return created[0]["id"]


def supabase_insert_message(conversation_id: str, sender_id: str, sender_role: str, body: str) -> dict:
    payload = [
        {
            "conversation_id": conversation_id,
            "sender_id": sender_id,
            "sender_role": sender_role,
            "body": body,
        }
    ]
    created = sb_post("messages", payload)
    if not created or not isinstance(created, list):
        raise HTTPException(status_code=502, detail="Failed to insert supabase message.")
    return created[0]


def get_owner_push_token(owner_id: str) -> Optional[str]:
    if not owner_id:
        return None
    rows = sb_get("owner_push_tokens", {"select": "expo_push_token", "owner_id": f"eq.{owner_id}", "limit": "1"})
    if rows and isinstance(rows, list):
        return rows[0].get("expo_push_token")
    return None


def send_expo_push(expo_push_token: str, title: str, body: str, data: Optional[Dict[str, Any]] = None):
    if not expo_push_token:
        return
    payload = {"to": expo_push_token, "title": title, "body": body, "data": data or {}}
    if requests is not None:
        try:
            requests.post("https://exp.host/--/api/v2/push/send", json=payload, timeout=10)
        except Exception:
            pass


# =========================
# Routes
# =========================
@app.get("/health")
def health():
    return {"ok": True, "service": "vinnies-brain-backend"}


@app.post("/v1/sessions", response_model=CreateSessionResponse)
def create_session(req: CreateSessionRequest):
    sid = str(uuid.uuid4())
    with db() as conn:
        if req.reset_old_session_id:
            # Optionally wipe old session messages if you want.
            if req.delete_old_messages:
                exec_no_return(conn, "DELETE FROM chat_messages WHERE session_id=%s", (req.reset_old_session_id,))
            exec_no_return(conn, "DELETE FROM sessions WHERE id=%s", (req.reset_old_session_id,))

        exec_no_return(conn, "INSERT INTO sessions (id, channel, mode) VALUES (%s, %s, %s)", (sid, req.channel, req.mode))
        conn.commit()
    return {"session_id": sid}


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
        exec_no_return(conn, "UPDATE sessions SET airstream_year=%s, category=%s WHERE id=%s", (req.airstream_year, req.category, session_id))
        conn.commit()
    return {"ok": True}


@app.post("/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    with db() as conn:
        sess = get_session(conn, req.session_id)

        year = req.airstream_year or sess.get("airstream_year")
        category = sess.get("category")

        flags = detect_safety_flags(req.message)

        if is_greeting(req.message):
            return ChatResponse(
                answer="Hey! What Airstream issue are you dealing with today?",
                clarifying_questions=[],
                safety_flags=flags,
                confidence=0.7,
                used_articles=[],
                show_escalation=False,
                message_id=str(uuid.uuid4()),
            )

        if not is_airstream_question(req.message, year):
            return ChatResponse(
                answer="I can help with Airstream troubleshooting. What system are you working on (water, electrical, appliances, leaks, etc.)?",
                clarifying_questions=[],
                safety_flags=flags,
                confidence=0.4,
                used_articles=[],
                show_escalation=False,
                message_id=str(uuid.uuid4()),
            )

        q_emb = embed(req.message)
        ranked = rank_kb_articles(conn, q_emb, year, category, top_k=6)

        used_articles = [{"id": r["id"], "title": r["title"]} for r, _ in ranked]
        context_chunks = []
        for r, score in ranked:
            if score < 0.25:
                continue
            context_chunks.append(f"TITLE: {r['title']}\nBODY:\n{r['body']}")

        history = get_recent_messages(conn, req.session_id, limit=20)

        try:
            answer, clarifying, confidence = generate_answer(
                user_message=req.message,
                context="\n\n---\n\n".join(context_chunks),
                safety_flags=flags,
                airstream_year=year,
                category=category,
                history=history,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM error: {e}")

        message_id = str(uuid.uuid4())
        log_message(conn, req.session_id, "user", req.message)
        log_message(conn, req.session_id, "assistant", answer)
        conn.commit()

        show_escalation = True

        return ChatResponse(
            answer=answer,
            clarifying_questions=clarifying,
            safety_flags=flags,
            confidence=confidence,
            used_articles=[UsedArticle(**a) for a in used_articles],
            show_escalation=show_escalation,
            message_id=message_id,
        )


@app.post("/v1/escalations", response_model=EscalationResponse)
def create_escalation(req: EscalationRequest):
    ticket_id = str(uuid.uuid4())
    # Keep your existing escalation handling here (email/ticket/etc.)
    return {"ticket_id": ticket_id}


@app.post("/v1/owner/push-token")
def register_owner_push_token(req: OwnerPushTokenRequest):
    _ = sb_upsert(
        "owner_push_tokens",
        [{"owner_id": req.owner_id, "expo_push_token": req.expo_push_token}],
        on_conflict="owner_id",
    )
    return {"ok": True}


@app.post("/v1/livechat/send")
def livechat_send(req: LiveChatSendRequest):
    conversation_id = get_or_create_conversation_for_session(req.session_id)
    msg = supabase_insert_message(conversation_id, req.session_id, "customer", req.body)

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
    conv_id = get_or_create_conversation_for_session(session_id)
    rows = sb_get(
        "messages",
        {
            "select": "id,conversation_id,sender_id,sender_role,body,created_at",
            "conversation_id": f"eq.{conv_id}",
            "order": "created_at.asc",
            "limit": "500",
        },
    )
    return {"conversation_id": conv_id, "messages": rows or []}


# =========================
# Admin live chat (INBOX + REPLY)
# =========================
@app.get("/v1/admin/livechat/conversations")
def admin_livechat_conversations(x_admin_key: str = Header(default="", alias="X-Admin-Key")):
    require_admin(x_admin_key)

    msg_rows = sb_get(
        "messages",
        {
            "select": "id,conversation_id,sender_role,body,created_at",
            "order": "created_at.desc",
            "limit": "300",
        },
    )

    conv_ids: List[str] = []
    last_by_conv: Dict[str, Dict[str, Any]] = {}

    for r in (msg_rows or []):
        cid = r.get("conversation_id")
        if not cid or cid in last_by_conv:
            continue
        last_by_conv[cid] = {
            "sender_role": r.get("sender_role"),
            "body": r.get("body"),
            "created_at": r.get("created_at"),
        }
        conv_ids.append(cid)
        if len(conv_ids) >= 50:
            break

    if not conv_ids:
        return {"conversations": []}

    conv_rows = sb_get(
        "conversations",
        {
            "select": "id,customer_id,created_at",
            "id": f"in.({','.join(conv_ids)})",
        },
    )
    by_id = {c.get("id"): c for c in (conv_rows or [])}

    conversations: List[Dict[str, Any]] = []
    for cid in conv_ids:
        c = by_id.get(cid) or {}
        conversations.append(
            {
                "conversation_id": cid,
                "customer_id": c.get("customer_id") or "",
                "last_message": last_by_conv.get(cid),
            }
        )

    return {"conversations": conversations}


@app.get("/v1/admin/livechat/history/{conversation_id}")
def admin_livechat_history(
    conversation_id: str, x_admin_key: str = Header(default="", alias="X-Admin-Key")
):
    require_admin(x_admin_key)

    rows = sb_get(
        "messages",
        {
            "select": "id,conversation_id,sender_id,sender_role,body,created_at",
            "conversation_id": f"eq.{conversation_id}",
            "order": "created_at.asc",
            "limit": "500",
        },
    )
    return {"conversation_id": conversation_id, "messages": rows or []}


@app.post("/v1/admin/livechat/send")
def admin_livechat_send(
    req: AdminLiveChatSendRequest, x_admin_key: str = Header(default="", alias="X-Admin-Key")
):
    require_admin(x_admin_key)

    sender_id = OWNER_SUPABASE_USER_ID or "owner"
    msg = supabase_insert_message(req.conversation_id, sender_id, "owner", req.body)
    return {"ok": True, "conversation_id": req.conversation_id, "message": msg}


@app.exception_handler(Exception)
async def function_exception_handler(request: Request, exc: Exception):
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
