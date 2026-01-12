import os
import uuid
import json
import traceback
from contextlib import contextmanager
from typing import List, Optional, Dict, Any, Tuple, Set
from urllib.parse import urlencode
from collections import deque

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
        raise HTTPException(
            status_code=500,
            detail="Supabase is not configured (SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY).",
        )
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


# ✅ Schema-aligned admin article payload (matches your admin.tsx POST)
class AdminArticleRequest(BaseModel):
    title: str
    category: str = "General"
    severity: str = "Medium"
    years_min: int = 2010
    years_max: int = 2025

    customer_summary: str

    # These exist in your table, but when decision_tree is present they will be derived
    clarifying_questions: Optional[Any] = None  # jsonb
    steps: Optional[Any] = None  # jsonb
    model_year_notes: Optional[Any] = None  # jsonb
    stop_and_escalate: Optional[Any] = None  # jsonb
    next_step: Optional[str] = None  # text

    retrieval_text: Optional[str] = None  # text

    # Canonical authored logic:
    decision_tree: Optional[Dict[str, Any]] = None  # jsonb


# =========================
# Decision-tree derivation helpers
# =========================
END_TARGETS = {"end_done", "end_escalate", "end_not_applicable"}


def _dt_get_nodes(tree: Dict[str, Any]) -> Dict[str, Any]:
    nodes = tree.get("nodes")
    if not isinstance(nodes, dict):
        return {}
    return nodes


def _dt_get_start(tree: Dict[str, Any]) -> str:
    start = tree.get("start")
    return start if isinstance(start, str) else ""


def _dt_node_question_text(node: Dict[str, Any]) -> str:
    title = (node.get("title") or "").strip()
    body = (node.get("body") or "").strip()
    if title and body:
        return f"{title} — {body}"
    return title or body


def derive_from_decision_tree(tree: Dict[str, Any]) -> Dict[str, Any]:
    """
    Canonical rule:
      If decision_tree exists, it becomes the SINGLE source of truth.

    Returns:
      clarifying_questions: List[str]
      steps: List[str]
      next_step: Optional[str]
    """
    nodes = _dt_get_nodes(tree)
    start = _dt_get_start(tree)

    if not start or start not in nodes:
        # Can't derive reliably
        return {"clarifying_questions": [], "steps": [], "next_step": None}

    # 1) Clarifying questions: root question (single best first question)
    root_node = nodes.get(start) or {}
    root_q = _dt_node_question_text(root_node).strip()
    clarifying_questions: List[str] = [root_q] if root_q else []

    # 2) Steps: flatten reachable nodes (BFS), keeping them stable-ish for retrieval
    steps: List[str] = []
    visited: Set[str] = set()
    q = deque([start])

    while q:
        nid = q.popleft()
        if nid in visited:
            continue
        visited.add(nid)

        node = nodes.get(nid) or {}
        qtext = _dt_node_question_text(node).strip()
        if qtext:
            steps.append(qtext)

        opts = node.get("options") or []
        if isinstance(opts, list):
            for o in opts:
                if not isinstance(o, dict):
                    continue
                goto = (o.get("goto") or "").strip()
                if not goto or goto in END_TARGETS:
                    continue
                if goto in nodes and goto not in visited:
                    q.append(goto)

    # 3) next_step: simple deterministic fallback based on whether tree contains an escalate end
    next_step: Optional[str] = None
    saw_escalate = False
    saw_done = False

    for nid, node in nodes.items():
        opts = node.get("options") or []
        if not isinstance(opts, list):
            continue
        for o in opts:
            if not isinstance(o, dict):
                continue
            goto = (o.get("goto") or "").strip()
            if goto == "end_escalate":
                saw_escalate = True
            elif goto == "end_done":
                saw_done = True

    if saw_escalate:
        next_step = "request_help"
    elif saw_done:
        next_step = "issue_resolved"

    return {"clarifying_questions": clarifying_questions, "steps": steps, "next_step": next_step}


def build_retrieval_text(payload: Dict[str, Any]) -> str:
    parts: List[str] = []
    parts.append(f"Title: {payload.get('title','')}")
    parts.append(f"Category: {payload.get('category','')}")
    parts.append(f"Severity: {payload.get('severity','')}")
    parts.append(f"Years: {payload.get('years_min','')}-{payload.get('years_max','')}")
    parts.append(f"Customer Summary: {payload.get('customer_summary','')}")

    ns = (payload.get("next_step") or "").strip()
    if ns:
        parts.append(f"Next Step: {ns}")

    cq = payload.get("clarifying_questions")
    if isinstance(cq, list) and cq:
        parts.append("Clarifying Questions: " + " | ".join(str(x) for x in cq if str(x).strip()))

    st = payload.get("steps")
    if isinstance(st, list) and st:
        parts.append("Steps: " + " | ".join(str(x) for x in st if str(x).strip()))
    elif st is not None:
        # keep structure searchable
        parts.append("Steps(JSON): " + json.dumps(st))

    my = payload.get("model_year_notes")
    if my is not None:
        parts.append("Model Year Notes(JSON): " + json.dumps(my))

    sae = payload.get("stop_and_escalate")
    if sae is not None:
        parts.append("Stop and Escalate(JSON): " + json.dumps(sae))

    dt = payload.get("decision_tree")
    if dt is not None:
        parts.append("Decision Tree(JSON): " + json.dumps(dt))

    return "\n".join(parts).strip()


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
    # Vector search (requires embeddings to be present)
    sql = """
    SELECT id, title, customer_summary AS body, retrieval_text,
           1 - (embedding <=> %s::vector) AS score
      FROM kb_articles
     WHERE embedding IS NOT NULL
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


def keyword_kb_articles(conn, query_text: str, year: Optional[int], category: Optional[str], top_k: int = 6):
    # Text match search (works even if embeddings are NULL)
    q = (query_text or "").strip()
    if not q:
        return []

    sql = """
    SELECT id, title, customer_summary AS body, retrieval_text
      FROM kb_articles
     WHERE (
        title ILIKE %s
        OR COALESCE(retrieval_text, '') ILIKE %s
        OR COALESCE(customer_summary, '') ILIKE %s
     )
    """
    params: List[Any] = [f"%{q}%", f"%{q}%", f"%{q}%"]

    if year is not None:
        sql += " AND years_min <= %s AND years_max >= %s"
        params.extend([year, year])

    if category:
        sql += " AND category = %s"
        params.append(category)

    sql += " LIMIT %s"
    params.append(top_k)

    rows = exec_all(conn, sql, tuple(params))
    # Give keyword hits a strong pseudo-score so they reliably make it into context
    return [(r, 0.95) for r in rows]


# =========================
# KB admin helpers
# =========================
_KB_COLUMNS_CACHE: Optional[set] = None


def _get_kb_columns(conn) -> set:
    global _KB_COLUMNS_CACHE
    if _KB_COLUMNS_CACHE is not None:
        return _KB_COLUMNS_CACHE

    rows = exec_all(
        conn,
        "SELECT column_name FROM information_schema.columns WHERE table_name='kb_articles'",
        (),
    )
    _KB_COLUMNS_CACHE = {r["column_name"] for r in (rows or []) if r.get("column_name")}
    return _KB_COLUMNS_CACHE


def _vector_literal(vec: List[float]) -> str:
    # pgvector accepts a string like: [0.1,0.2,0.3]
    return "[" + ",".join(f"{float(x):.8f}" for x in vec) + "]"


def kb_insert_article(conn, req: AdminArticleRequest) -> Dict[str, Any]:
    """
    IMPORTANT:
      If req.decision_tree is present, we derive clarifying_questions/steps/next_step from it,
      and ignore client-provided duplicates.
    """
    cols = _get_kb_columns(conn)
    article_id = str(uuid.uuid4())

    # Base payload
    payload: Dict[str, Any] = {
        "id": article_id,
        "title": (req.title or "").strip(),
        "category": (req.category or "").strip(),
        "severity": (req.severity or "").strip(),
        "years_min": int(req.years_min),
        "years_max": int(req.years_max),
        "customer_summary": (req.customer_summary or "").strip(),
        "decision_tree": req.decision_tree,
        "model_year_notes": req.model_year_notes if req.model_year_notes is not None else {},
        "stop_and_escalate": req.stop_and_escalate if req.stop_and_escalate is not None else {},
    }

    # Derive (decision_tree -> canonical)
    if isinstance(req.decision_tree, dict) and req.decision_tree:
        derived = derive_from_decision_tree(req.decision_tree)
        payload["clarifying_questions"] = derived.get("clarifying_questions") or []
        payload["steps"] = derived.get("steps") or []
        payload["next_step"] = derived.get("next_step")
    else:
        # Backward compat: accept client-provided values if no decision tree
        payload["clarifying_questions"] = req.clarifying_questions
        payload["steps"] = req.steps
        payload["next_step"] = (req.next_step or "").strip() or None

    # Retrieval text: if provided, keep it, but ensure tree is included; else auto-build
    provided_rt = (req.retrieval_text or "").strip()
    payload_for_rt = {
        **payload,
        "clarifying_questions": payload.get("clarifying_questions"),
        "steps": payload.get("steps"),
        "next_step": payload.get("next_step"),
    }
    auto_rt = build_retrieval_text(payload_for_rt)
    payload["retrieval_text"] = provided_rt or auto_rt

    # Compute embedding from retrieval_text (stable + includes the tree)
    doc_text = payload["retrieval_text"] or auto_rt
    emb = embed_text(doc_text)
    emb_literal = _vector_literal(emb)

    insert_cols: List[str] = []
    insert_vals: List[str] = []
    params: List[Any] = []

    def add(col: str, val: Any):
        if col in cols:
            insert_cols.append(col)
            insert_vals.append("%s")
            params.append(val)

    def add_json(col: str, val: Any):
        if col in cols:
            insert_cols.append(col)
            insert_vals.append("%s")
            params.append(json.dumps(val) if val is not None else None)

    # Required-ish text fields
    add("id", payload["id"])
    add("title", payload["title"])
    add("category", payload["category"])
    add("severity", payload["severity"])
    add("years_min", payload["years_min"])
    add("years_max", payload["years_max"])
    add("customer_summary", payload["customer_summary"])

    # Derived/optional schema fields
    add_json("clarifying_questions", payload.get("clarifying_questions"))
    add_json("steps", payload.get("steps"))
    add_json("model_year_notes", payload.get("model_year_notes") if payload.get("model_year_notes") is not None else {})
    add_json("stop_and_escalate", payload.get("stop_and_escalate") if payload.get("stop_and_escalate") is not None else {})

    add("next_step", payload.get("next_step"))
    add("retrieval_text", payload.get("retrieval_text"))

    # Decision tree is jsonb
    add_json("decision_tree", payload.get("decision_tree"))

    # embedding (pgvector)
    if "embedding" in cols:
        insert_cols.append("embedding")
        insert_vals.append("%s::vector")
        params.append(emb_literal)

    if not insert_cols:
        raise HTTPException(status_code=500, detail="kb_articles table has no insertable columns (unexpected).")

    sql = f"INSERT INTO kb_articles ({', '.join(insert_cols)}) VALUES ({', '.join(insert_vals)}) RETURNING id, title"
    row = exec_one(conn, sql, tuple(params))
    return {"id": (row or {}).get("id", article_id), "title": (row or {}).get("title", payload["title"])}


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
        exec_no_return(
            conn,
            "UPDATE sessions SET airstream_year=%s, category=%s WHERE id=%s",
            (req.airstream_year, req.category, session_id),
        )
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

        # ✅ DB-first: try keyword/title match first (works even if embeddings are NULL)
        ranked = keyword_kb_articles(conn, req.message, year, category, top_k=6)

        # If no keyword hits, try vector search (requires embeddings)
        if not ranked:
            q_emb = embed(req.message)
            ranked = rank_kb_articles(conn, q_emb, year, category, top_k=6)

        # Build KB context (only from rows that pass the threshold)
        context_chunks: List[str] = []
        used_articles: List[Dict[str, str]] = []
        for r, score in ranked:
            if score < 0.15:
                continue

            used_articles.append({"id": r["id"], "title": r["title"]})

            rt = (r.get("retrieval_text") or "").strip()
            body = (r.get("body") or "").strip()

            chunk = f"TITLE: {r['title']}\n"
            if rt:
                chunk += f"RETRIEVAL_TEXT:\n{rt}\n"
            chunk += f"BODY:\n{body}"
            context_chunks.append(chunk)

        from_kb = len(context_chunks) > 0

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
            if not from_kb:
                answer = "⚠️ This information is NOT from Vinnies Brain’s database.\n\n" + answer
                confidence = min(confidence, 0.45)
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
# Admin KB article create
# =========================
@app.post("/v1/admin/articles")
def admin_create_article(
    req: AdminArticleRequest, x_admin_key: str = Header(default="", alias="X-Admin-Key")
):
    require_admin(x_admin_key)

    with db() as conn:
        try:
            out = kb_insert_article(conn, req)
            conn.commit()
            return {"ok": True, "article": out}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create article: {e}")


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


# @app.exception_handler(Exception)
# async def function_exception_handler(request: Request, exc: Exception):
#     traceback.print_exc()
#     return JSONResponse(status_code=500, content={"detail": str(exc)})
