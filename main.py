import os
import uuid
import json
import traceback
import logging
import time
from contextlib import contextmanager
from typing import List, Optional, Dict, Any, Tuple, Set
from urllib.parse import urlencode
from collections import deque, defaultdict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from openai_client import embed_text, generate_answer

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

# =========================
# Runtime hardening
# =========================
logger = logging.getLogger("vinniesbrain")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

POOL_MIN_SIZE = int(os.getenv("DB_POOL_MIN", "1"))
POOL_MAX_SIZE = int(os.getenv("DB_POOL_MAX", "6"))

DB_POOL: ConnectionPool = ConnectionPool(
    conninfo=DATABASE_URL,
    min_size=POOL_MIN_SIZE,
    max_size=POOL_MAX_SIZE,
    timeout=15,
)

# naive in-memory rate limiting for /v1/chat (IP + session)
_RATE_WINDOW_SEC = int(os.getenv("CHAT_RATE_WINDOW_SEC", "60"))
_RATE_MAX_REQ = int(os.getenv("CHAT_RATE_MAX_REQ", "45"))
_rate_buckets: Dict[str, deque] = defaultdict(deque)


def _now() -> float:
    return time.time()


def _client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _rate_allow(bucket: str) -> bool:
    ts = _now()
    dq = _rate_buckets[bucket]
    cutoff = ts - _RATE_WINDOW_SEC
    while dq and dq[0] < cutoff:
        dq.popleft()
    if len(dq) >= _RATE_MAX_REQ:
        return False
    dq.append(ts)
    return True


@app.middleware("http")
async def request_id_and_rate_limit(request: Request, call_next):
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = rid

    # Rate limit only the chat endpoint
    if request.url.path == "/v1/chat":
        try:
            body = await request.body()
            request._body = body  # restore for downstream reads
        except Exception:
            body = b""

        sid = ""
        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
            sid = str(payload.get("session_id") or "")
        except Exception:
            sid = ""

        bucket = f"{_client_ip(request)}:{sid or 'nosession'}"
        if not _rate_allow(bucket):
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many messages. Please wait a moment and try again.", "request_id": rid},
                headers={"X-Request-Id": rid},
            )

    start = _now()
    try:
        resp = await call_next(request)
    except Exception:
        logger.exception("Unhandled error rid=%s path=%s", rid, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Server error", "request_id": rid},
            headers={"X-Request-Id": rid},
        )
    finally:
        dur_ms = int((_now() - start) * 1000)
        logger.info("rid=%s %s %s %sms", rid, request.method, request.url.path, dur_ms)

    resp.headers["X-Request-Id"] = rid
    return resp


@contextmanager
def db():
    conn = DB_POOL.getconn()
    try:
        conn.row_factory = dict_row
        yield conn
    finally:
        DB_POOL.putconn(conn)


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
# Support routing helpers
# =========================
# Business hours are evaluated server-side in Pacific Time so iOS/Android behave identically.
SUPPORT_TZ = os.getenv("SUPPORT_TZ", "America/Los_Angeles")
SUPPORT_OPEN_HOUR = int(os.getenv("SUPPORT_OPEN_HOUR", "8"))   # 8am
SUPPORT_CLOSE_HOUR = int(os.getenv("SUPPORT_CLOSE_HOUR", "17"))  # 5pm
# 0=Mon ... 6=Sun
SUPPORT_DAYS_OPEN = os.getenv("SUPPORT_DAYS_OPEN", "0,1,2,3,4")  # Mon–Fri
SUPPORT_EMAIL_TO = os.getenv("SUPPORT_EMAIL_TO", "info@vinnies.net")


def _pt_now() -> datetime:
    return datetime.now(ZoneInfo(SUPPORT_TZ))


def is_business_hours_now() -> bool:
    now = _pt_now()
    try:
        open_days = {int(x.strip()) for x in SUPPORT_DAYS_OPEN.split(",") if x.strip() != ""}
    except Exception:
        open_days = {0, 1, 2, 3, 4}

    if now.weekday() not in open_days:
        return False

    # open at SUPPORT_OPEN_HOUR:00 inclusive, close at SUPPORT_CLOSE_HOUR:00 exclusive
    return SUPPORT_OPEN_HOUR <= now.hour < SUPPORT_CLOSE_HOUR


def next_business_open_iso() -> str:
    """Returns an ISO string (PT) for the next opening time."""
    now = _pt_now()
    try:
        open_days = sorted({int(x.strip()) for x in SUPPORT_DAYS_OPEN.split(",") if x.strip() != ""})
        if not open_days:
            open_days = [0, 1, 2, 3, 4]
    except Exception:
        open_days = [0, 1, 2, 3, 4]

    # search day by day up to 14 days
    for i in range(0, 14):
        d = now + timedelta(days=i)
        if d.weekday() not in open_days:
            continue
        candidate = d.replace(hour=SUPPORT_OPEN_HOUR, minute=0, second=0, microsecond=0)
        if candidate > now:
            return candidate.isoformat()
    # fallback: tomorrow at open
    d = now + timedelta(days=1)
    return d.replace(hour=SUPPORT_OPEN_HOUR, minute=0, second=0, microsecond=0).isoformat()


def build_escalation_email_subject(session_id: str) -> str:
    return f"Vinnies Brain Escalation — Session {session_id}"


def build_escalation_email_body(*, req: 'EscalationRequest', transcript: str, business_hours: bool) -> str:
    lines: List[str] = []
    lines.append("Vinnies Brain — Escalation")
    lines.append("")
    lines.append(f"Session ID: {req.session_id}")
    lines.append(f"Business hours at submit: {'YES' if business_hours else 'NO'}")
    lines.append("")
    lines.append("Customer info")
    lines.append(f"Name: {req.name}")
    if req.email:
        lines.append(f"Email: {req.email}")
    if req.phone:
        lines.append(f"Phone: {req.phone}")
    lines.append(f"Preferred contact: {req.preferred_contact}")
    lines.append("")
    lines.append("Issue summary")
    lines.append(req.message or "")
    lines.append("")
    lines.append("AI troubleshooting transcript")
    lines.append(transcript or "(no transcript found)")
    return "\n".join(lines).strip()

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


def sb_delete(table: str, params: Dict[str, str], prefer: str = "return=minimal") -> Any:
    """Supabase REST DELETE helper. Filters are passed via query params."""
    url = f"{SUPABASE_URL}/rest/v1/{table}"
    qs = urlencode(params or {})
    full = f"{url}?{qs}" if qs else url
    headers = _sb_headers({"Prefer": prefer})

    if requests is not None:
        r = requests.delete(full, headers=headers, timeout=15)
        if not r.ok:
            raise HTTPException(status_code=502, detail=f"Supabase DELETE {table} failed: {r.status_code} {r.text}")
        try:
            return r.json()
        except Exception:
            return {"ok": True}
    else:
        req = urllib.request.Request(full, headers=headers, method="DELETE")
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8")
            if not raw:
                return {"ok": True}
            try:
                return json.loads(raw)
            except Exception:
                return {"ok": True}


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
    escalation_id: Optional[str] = None
    routing: str = "chat"  # chat | email | both
    business_hours: bool = False
    conversation_id: Optional[str] = None

    # If the client wants the USER to send an email from their own mail app,
    # we return a prefilled subject/body. The client can open a mailto: link.
    email_to: Optional[str] = None
    email_subject: Optional[str] = None
    email_body: Optional[str] = None



class LiveChatSendRequest(BaseModel):
    session_id: str
    body: str


class AdminLiveChatSendRequest(BaseModel):
    conversation_id: str
    body: str


class OwnerPushTokenRequest(BaseModel):
    owner_id: str
    expo_push_token: str


class AdminArticleRequest(BaseModel):
    title: str
    category: str = "General"
    severity: str = "Medium"
    years_min: int = 2010
    years_max: int = 2025

    customer_summary: str

    clarifying_questions: Optional[Any] = None  # jsonb
    steps: Optional[Any] = None  # jsonb
    model_year_notes: Optional[Any] = None  # jsonb
    stop_and_escalate: Optional[Any] = None  # jsonb
    next_step: Optional[str] = None  # text

    retrieval_text: Optional[str] = None  # text

    decision_tree: Optional[Dict[str, Any]] = None  # jsonb


# NEW (login-ready): claim guest sessions after login
class ClaimSessionsRequest(BaseModel):
    session_ids: List[str] = Field(default_factory=list)


# =========================
# Decision-tree helpers
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
    nodes = _dt_get_nodes(tree)
    start = _dt_get_start(tree)

    if not start or start not in nodes:
        return {"clarifying_questions": [], "steps": [], "next_step": None}

    root_node = nodes.get(start) or {}
    root_q = _dt_node_question_text(root_node).strip()
    clarifying_questions: List[str] = [root_q] if root_q else []

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

    next_step: Optional[str] = None
    saw_escalate = False
    saw_done = False

    for _, node in nodes.items():
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
_SESS_COLUMNS_CACHE: Optional[set] = None


def _get_sessions_columns(conn) -> set:
    global _SESS_COLUMNS_CACHE
    if _SESS_COLUMNS_CACHE is not None:
        return _SESS_COLUMNS_CACHE
    rows = exec_all(
        conn,
        "SELECT column_name FROM information_schema.columns WHERE table_name='sessions'",
        (),
    )
    _SESS_COLUMNS_CACHE = {r["column_name"] for r in (rows or []) if r.get("column_name")}
    return _SESS_COLUMNS_CACHE


def sessions_supports_pinning(conn) -> bool:
    cols = _get_sessions_columns(conn)
    return {"active_article_id", "active_node_id", "active_tree"}.issubset(cols)


def sessions_supports_active_question(conn) -> bool:
    cols = _get_sessions_columns(conn)
    return "active_question_text" in cols


# NEW: login-ready sessions ownership
def sessions_supports_user_id(conn) -> bool:
    cols = _get_sessions_columns(conn)
    return "user_id" in cols


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


def get_recent_messages(conn, session_id: str, limit: int = 200):
    return exec_all(
        conn,
        "SELECT role, content AS text, created_at FROM chat_messages WHERE session_id=%s ORDER BY created_at ASC LIMIT %s",
        (session_id, limit),
    )

def format_transcript(messages: List[Dict[str, Any]], max_chars: int = 9000) -> str:
    """Human-readable transcript for escalation emails. Trims to keep mailto links usable."""
    lines: List[str] = []
    for m in messages or []:
        role = (m.get("role") or "").strip().lower()
        text = (m.get("text") or "").strip()
        if not role or not text:
            continue
        tag = "YOU" if role == "user" else ("VINNIES" if role == "assistant" else role.upper())
        lines.append(f"{tag}: {text}")

    out = "\n\n".join(lines).strip()
    if len(out) <= max_chars:
        return out

    # Keep the tail (most recent context tends to matter most)
    tail = out[-max_chars:]
    return "(transcript trimmed)\n...\n" + tail



# =========================
# Subtle check-in helpers (NEW)
# =========================
_CHECKIN_MARKERS = [
    "when you get a chance",
    "let me know what happened",
    "tell me what you see",
    "what did you notice",
    "did that change anything",
    "did that help",
    "after you try that",
]


def _looks_like_multi_step(answer_text: str) -> bool:
    t = (answer_text or "")
    if len(t) >= 900:
        return True
    stepish = 0
    for line in t.splitlines():
        s = line.strip()
        if not s:
            continue
        if s[:3].isdigit() and s[3:4] in {".", ")", ":"}:
            stepish += 1
        if s.lower().startswith(("step ", "steps:", "try this:", "do this:")):
            stepish += 1
    return stepish >= 2


def _recent_assistant_texts(history: List[Dict[str, Any]], max_n: int = 10) -> List[str]:
    out: List[str] = []
    for m in reversed(history or []):
        if len(out) >= max_n:
            break
        if (m.get("role") or "").strip() == "assistant":
            out.append((m.get("text") or "").strip())
    return out


def _already_checked_in_recently(history: List[Dict[str, Any]]) -> bool:
    for t in _recent_assistant_texts(history, max_n=8):
        tl = t.lower()
        if any(k in tl for k in _CHECKIN_MARKERS):
            return True
    return False


def _assistant_question_count_recent(history: List[Dict[str, Any]], max_n: int = 8) -> int:
    c = 0
    for t in _recent_assistant_texts(history, max_n=max_n):
        c += t.count("?")
    return c


def maybe_append_subtle_checkin(
    answer: str,
    clarifying: List[str],
    confidence: float,
    from_kb: bool,
    history: List[Dict[str, Any]],
) -> str:
    if not from_kb:
        return answer

    if (clarifying or []) and any((q or "").strip() for q in (clarifying or [])):
        return answer

    if confidence < 0.72:
        return answer

    if _already_checked_in_recently(history):
        return answer

    if _assistant_question_count_recent(history, max_n=6) >= 3:
        return answer

    if not _looks_like_multi_step(answer):
        return answer

    checkin_line = "When you get a chance to try that, tell me what happened (even if it didn’t change anything)."
    return (answer.rstrip() + "\n\n" + checkin_line).strip()


# =========================
# Pinned-flow helpers
# =========================
YES_NO_MAP = {
    "y": "yes",
    "yes": "yes",
    "yeah": "yes",
    "yep": "yes",
    "yup": "yes",
    "sure": "yes",
    "ok": "yes",
    "okay": "yes",
    "n": "no",
    "no": "no",
    "nope": "no",
    "nah": "no",
}


def normalize_yes_no(text: str) -> Optional[str]:
    t = (text or "").strip().lower()
    if t in YES_NO_MAP:
        return YES_NO_MAP[t]
    return None


def _opt_answer_key(opt: Dict[str, Any]) -> str:
    for k in ("answer", "value", "label", "text"):
        v = opt.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip().lower()
    return ""


def advance_node(tree: Dict[str, Any], node_id: str, user_answer: str) -> str:
    try:
        nodes = tree.get("nodes") or {}
        if not isinstance(nodes, dict):
            return node_id
        node = nodes.get(node_id) or {}
        opts = node.get("options") or []
        if not isinstance(opts, list):
            return node_id

        ua = normalize_yes_no(user_answer)
        if not ua:
            return node_id

        for opt in opts:
            if not isinstance(opt, dict):
                continue
            key = _opt_answer_key(opt)
            goto = (opt.get("goto") or "").strip()
            if not goto:
                continue
            if key == ua:
                return goto

        for opt in opts:
            if not isinstance(opt, dict):
                continue
            key = _opt_answer_key(opt)
            goto = (opt.get("goto") or "").strip()
            if not goto:
                continue
            if ua == "yes" and key in {"true", "t", "1"}:
                return goto
            if ua == "no" and key in {"false", "f", "0"}:
                return goto

        return node_id
    except Exception:
        return node_id


def node_text(tree: Dict[str, Any], node_id: str) -> str:
    nodes = _dt_get_nodes(tree)
    node = nodes.get(node_id) or {}
    return _dt_node_question_text(node).strip()


def should_reset_flow(message: str) -> bool:
    t = (message or "").lower()
    return any(p in t for p in ["new issue", "different issue", "different problem", "switch topic", "switch topics", "reset"])


def rewrite_short_answer(user_text: str, active_question_text: Optional[str]) -> str:
    yn = normalize_yes_no(user_text)
    q = (active_question_text or "").strip()
    if yn and q:
        return f'Answer to: "{q}" -> {yn}'
    return user_text


# =========================
# RAG helpers
# =========================
def embed(text: str) -> List[float]:
    return embed_text(text)


def rank_kb_articles(conn, query_embedding: List[float], year: Optional[int], category: Optional[str], top_k: int = 6):
    sql = """
    SELECT id, title,
           customer_summary AS body,
           retrieval_text,
           decision_tree,
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
    q = (query_text or "").strip()
    if not q:
        return []

    sql = """
    SELECT id, title,
           customer_summary AS body,
           retrieval_text,
           decision_tree
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
    return [(r, 0.95) for r in rows]


# =========================
# KB admin helpers (unchanged)
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
    return "[" + ",".join(f"{float(x):.8f}" for x in vec) + "]"


def kb_insert_article(conn, req: AdminArticleRequest) -> Dict[str, Any]:
    cols = _get_kb_columns(conn)
    article_id = str(uuid.uuid4())

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

    if isinstance(req.decision_tree, dict) and req.decision_tree:
        derived = derive_from_decision_tree(req.decision_tree)
        payload["clarifying_questions"] = derived.get("clarifying_questions") or []
        payload["steps"] = derived.get("steps") or []
        payload["next_step"] = derived.get("next_step")
    else:
        payload["clarifying_questions"] = req.clarifying_questions
        payload["steps"] = req.steps
        payload["next_step"] = (req.next_step or "").strip() or None

    provided_rt = (req.retrieval_text or "").strip()
    payload_for_rt = {
        **payload,
        "clarifying_questions": payload.get("clarifying_questions"),
        "steps": payload.get("steps"),
        "next_step": payload.get("next_step"),
    }
    auto_rt = build_retrieval_text(payload_for_rt)
    payload["retrieval_text"] = provided_rt or auto_rt

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

    add("id", payload["id"])
    add("title", payload["title"])
    add("category", payload["category"])
    add("severity", payload["severity"])
    add("years_min", payload["years_min"])
    add("years_max", payload["years_max"])
    add("customer_summary", payload["customer_summary"])

    add_json("clarifying_questions", payload.get("clarifying_questions"))
    add_json("steps", payload.get("steps"))
    add_json("model_year_notes", payload.get("model_year_notes") if payload.get("model_year_notes") is not None else {})
    add_json("stop_and_escalate", payload.get("stop_and_escalate") if payload.get("stop_and_escalate") is not None else {})
    add("next_step", payload.get("next_step"))
    add("retrieval_text", payload.get("retrieval_text"))
    add_json("decision_tree", payload.get("decision_tree"))

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
# Supabase live chat logic (unchanged)
# =========================
def get_or_create_conversation_for_session(session_id: str) -> str:
    rows = sb_get(
        "conversations",
        {"select": "id", "customer_id": f"eq.{session_id}", "limit": "1"},
    )
    if rows and isinstance(rows, list) and rows[0].get("id"):
        return rows[0]["id"]

    created = sb_post("conversations", [{"customer_id": session_id}])
    if not created or not isinstance(created, list) or not created[0].get("id"):
        raise HTTPException(status_code=502, detail="Failed to create supabase conversation.")
    return created[0]["id"]


def supabase_insert_message(conversation_id: str, sender_id: str, sender_role: str, body: str) -> dict:
    payload = [{"conversation_id": conversation_id, "sender_id": sender_id, "sender_role": sender_role, "body": body}]
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
# Escalations persistence
# =========================
def ensure_escalations_table(conn):
    exec_no_return(
        conn,
        """
        CREATE TABLE IF NOT EXISTS escalations (
          id TEXT PRIMARY KEY,
          session_id TEXT NOT NULL,
          name TEXT NOT NULL,
          phone TEXT NOT NULL,
          email TEXT NOT NULL,
          message TEXT NOT NULL,
          preferred_contact TEXT,
          context_json JSONB,
          created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
    )


@app.on_event("startup")
def _startup():
    with db() as conn:
        ensure_escalations_table(conn)
        conn.commit()


@app.on_event("shutdown")
def _shutdown():
    try:
        DB_POOL.close()
    except Exception:
        pass


# =========================
# Routes
# =========================
@app.get("/health")
def health():
    return {"ok": True, "service": "vinnies-brain-backend"}

@app.get("/v1/support/status")
def support_status():
    """Lets the mobile app decide whether to show Live Chat vs Email options."""
    open_now = is_business_hours_now()
    return {
        "business_hours": open_now,
        "timezone": SUPPORT_TZ,
        "open_hour": SUPPORT_OPEN_HOUR,
        "close_hour": SUPPORT_CLOSE_HOUR,
        "next_open": None if open_now else next_business_open_iso(),
        "support_email": SUPPORT_EMAIL_TO,
    }


@app.post("/v1/sessions", response_model=CreateSessionResponse)
def create_session(req: CreateSessionRequest, request: Request):
    """
    Creates a new session. If the client includes X-User-Id (UUID),
    we attach the session to that user (login-ready).
    """
    sid = str(uuid.uuid4())

    # TEMP bridge until real auth exists:
    # Later you will derive user_id from a verified JWT.
    user_id = (request.headers.get("X-User-Id") or "").strip() or None

    with db() as conn:
        if req.reset_old_session_id:
            if req.delete_old_messages:
                exec_no_return(conn, "DELETE FROM chat_messages WHERE session_id=%s", (req.reset_old_session_id,))
            exec_no_return(conn, "DELETE FROM sessions WHERE id=%s", (req.reset_old_session_id,))

        if user_id and sessions_supports_user_id(conn):
            exec_no_return(
                conn,
                "INSERT INTO sessions (id, channel, mode, user_id) VALUES (%s, %s, %s, %s)",
                (sid, req.channel, req.mode, user_id),
            )
        else:
            exec_no_return(conn, "INSERT INTO sessions (id, channel, mode) VALUES (%s, %s, %s)", (sid, req.channel, req.mode))

        conn.commit()
    return {"session_id": sid}


@app.get("/v1/sessions")
def list_sessions(request: Request):
    """
    Login-ready 'Previous Issues' endpoint.
    Uses X-User-Id for now (temporary until JWT auth is implemented).
    """
    user_id = (request.headers.get("X-User-Id") or "").strip()
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated (missing X-User-Id)")

    with db() as conn:
        if not sessions_supports_user_id(conn):
            raise HTTPException(status_code=500, detail="sessions.user_id column missing (run migration)")

        rows = exec_all(
            conn,
            """
            SELECT
              s.id AS session_id,
              lm.created_at AS last_message_at,
              lm.content AS preview
            FROM sessions s
            LEFT JOIN LATERAL (
              SELECT created_at, content
              FROM chat_messages
              WHERE session_id = s.id
              ORDER BY created_at DESC
              LIMIT 1
            ) lm ON TRUE
            WHERE s.user_id = %s::uuid
            ORDER BY lm.created_at DESC NULLS LAST
            LIMIT 50
            """,
            (user_id,),
        )

    # trim preview
    out = []
    for r in rows or []:
        p = (r.get("preview") or "").strip()
        r["preview"] = (p[:120] + "…") if len(p) > 120 else p
        out.append(r)

    return {"sessions": out}


@app.post("/v1/sessions/claim")
def claim_sessions(req: ClaimSessionsRequest, request: Request):
    """
    Called once after login to attach guest sessions to the logged-in user.
    Only claims sessions that are currently unowned (user_id IS NULL).
    """
    user_id = (request.headers.get("X-User-Id") or "").strip()
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated (missing X-User-Id)")

    session_ids = [s.strip() for s in (req.session_ids or []) if isinstance(s, str) and s.strip()]
    if not session_ids:
        return {"ok": True, "claimed": 0}

    with db() as conn:
        if not sessions_supports_user_id(conn):
            raise HTTPException(status_code=500, detail="sessions.user_id column missing (run migration)")

        exec_no_return(
            conn,
            """
            UPDATE sessions
               SET user_id = %s::uuid
             WHERE id = ANY(%s::text[])
               AND user_id IS NULL
            """,
            (user_id, session_ids),
        )
        conn.commit()

    return {"ok": True, "claimed": len(session_ids)}


@app.get("/v1/sessions/{session_id}")
def session_exists(session_id: str):
    with db() as conn:
        row = exec_one(conn, "SELECT id FROM sessions WHERE id=%s", (session_id,))
        if not row:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"ok": True}

@app.get("/v1/sessions/{session_id}/history")
def session_history(session_id: str):
    """Customer-safe transcript endpoint (used to prefill an escalation email)."""
    with db() as conn:
        _ = get_session(conn, session_id)
        msgs = get_recent_messages(conn, session_id, limit=250)
        out = []
        for m in (msgs or []):
            role = (m.get("role") or "").strip()
            text = (m.get("text") or "").strip()
            if not role or not text:
                continue
            out.append({"role": role, "text": text, "created_at": m.get("created_at")})
        return {"session_id": session_id, "messages": out}


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


# -------------------------
# Admin: AI troubleshooting progress for a session
# -------------------------
def _admin_ai_history_payload(conn, session_id: str) -> Dict[str, Any]:
    sess = get_session(conn, session_id)
    msgs = get_recent_messages(conn, session_id, limit=500)

    out_msgs = []
    for m in (msgs or []):
        role = (m.get("role") or "").strip()
        text = (m.get("text") or "").strip()
        if not role or not text:
            continue
        out_msgs.append({"role": role, "text": text, "created_at": m.get("created_at")})

    payload: Dict[str, Any] = {"session_id": session_id, "messages": out_msgs}

    if sessions_supports_pinning(conn):
        payload["active_article_id"] = sess.get("active_article_id")
        payload["active_node_id"] = sess.get("active_node_id")
        tree = sess.get("active_tree")
        if isinstance(tree, str):
            try:
                tree = json.loads(tree)
            except Exception:
                tree = None
        payload["active_tree_present"] = bool(tree)
        if isinstance(tree, dict) and sess.get("active_node_id"):
            payload["active_node_text"] = node_text(tree, sess.get("active_node_id"))

    if sessions_supports_active_question(conn):
        payload["active_question_text"] = sess.get("active_question_text")

    return payload


@app.get("/admin/ai-history/{session_id}")
@app.get("/v1/admin/ai-history/{session_id}")
def admin_ai_history(session_id: str, x_admin_key: str = Header(default="", alias="X-Admin-Key")):
    require_admin(x_admin_key)
    with db() as conn:
        return _admin_ai_history_payload(conn, session_id)


# -------------------------
# Chat (Pinned-flow + active_question_text)
# -------------------------
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

        pin_supported = sessions_supports_pinning(conn)
        aq_supported = sessions_supports_active_question(conn)

        active_article_id = sess.get("active_article_id") if pin_supported else None
        active_node_id = sess.get("active_node_id") if pin_supported else None
        active_tree = sess.get("active_tree") if pin_supported else None
        active_question_text = sess.get("active_question_text") if aq_supported else None

        # reset flow if user explicitly asks to switch/reset
        if should_reset_flow(req.message):
            updates = []
            if pin_supported:
                updates.append("active_article_id=NULL, active_node_id=NULL, active_tree=NULL")
            if aq_supported:
                updates.append("active_question_text=NULL")
            if updates:
                exec_no_return(conn, f"UPDATE sessions SET {', '.join(updates)} WHERE id=%s", (req.session_id,))
                conn.commit()
            active_article_id = None
            active_node_id = None
            active_tree = None
            active_question_text = None

        if isinstance(active_tree, str):
            try:
                active_tree = json.loads(active_tree)
            except Exception:
                active_tree = None

        yn = normalize_yes_no(req.message)

        # If user says yes/no and we have a stored question, rewrite it so the model knows what it's answering.
        rewritten_user_message = rewrite_short_answer(req.message, active_question_text)

        use_pinned = bool(pin_supported and active_article_id and isinstance(active_tree, dict) and active_node_id and yn)

        used_articles: List[Dict[str, str]] = []
        context_chunks: List[str] = []
        from_kb = False

        if use_pinned:
            next_node = advance_node(active_tree, active_node_id, req.message)

            # update session node
            exec_no_return(conn, "UPDATE sessions SET active_node_id=%s WHERE id=%s", (next_node, req.session_id))

            # pinned article only
            row = exec_one(
                conn,
                """
                SELECT id, title, customer_summary AS body, retrieval_text, decision_tree
                  FROM kb_articles
                 WHERE id=%s
                """,
                (active_article_id,),
            )

            if row:
                used_articles.append({"id": row["id"], "title": row["title"]})
                rt = (row.get("retrieval_text") or "").strip()
                body = (row.get("body") or "").strip()
                tree = row.get("decision_tree")

                chunk = f"TITLE: {row['title']}\n"
                if rt:
                    chunk += f"RETRIEVAL_TEXT:\n{rt}\n"
                chunk += f"BODY:\n{body}\n\n"

                if isinstance(tree, dict) and next_node:
                    qtext = node_text(tree, next_node)
                    if qtext:
                        chunk += f"ACTIVE_TROUBLESHOOTING_NODE:\n{next_node}\nQUESTION:\n{qtext}\n"

                # Also explicitly include the last asked question (if any)
                if (active_question_text or "").strip():
                    chunk += f'\nLAST_AI_QUESTION:\n{active_question_text.strip()}\n'

                context_chunks.append(chunk)
                from_kb = True

        else:
            ranked = keyword_kb_articles(conn, req.message, year, category, top_k=6)
            if not ranked:
                q_emb = embed(req.message)
                ranked = rank_kb_articles(conn, q_emb, year, category, top_k=6)

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

            # Pin the top article/tree when available
            if pin_supported and from_kb and ranked:
                top = ranked[0][0]
                tree = top.get("decision_tree")
                if isinstance(tree, dict) and tree.get("start") and tree.get("nodes"):
                    exec_no_return(
                        conn,
                        """
                        UPDATE sessions
                           SET active_article_id=%s,
                               active_tree=%s,
                               active_node_id=%s
                         WHERE id=%s
                        """,
                        (top["id"], json.dumps(tree), tree.get("start"), req.session_id),
                    )

        history = get_recent_messages(conn, req.session_id, limit=50)

        try:
            answer, clarifying, confidence = generate_answer(
                user_message=rewritten_user_message,
                context="\n\n---\n\n".join(context_chunks),
                safety_flags=flags,
                airstream_year=year,
                category=category,
                history=history,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM error: {e}")

        # Subtle check-in (NEW): occasionally append a gentle “let me know what happened after trying that”
        answer = maybe_append_subtle_checkin(
            answer=answer,
            clarifying=clarifying,
            confidence=confidence,
            from_kb=from_kb,
            history=history,
        )

        # If the AI asked a clarifying question, store it as active_question_text
        if aq_supported:
            q_to_store = (clarifying[0] if (isinstance(clarifying, list) and len(clarifying) > 0) else "").strip()
            exec_no_return(
                conn,
                "UPDATE sessions SET active_question_text=%s WHERE id=%s",
                (q_to_store or None, req.session_id),
            )

        message_id = str(uuid.uuid4())
        log_message(conn, req.session_id, "user", req.message)
        log_message(conn, req.session_id, "assistant", answer)
        conn.commit()

        return ChatResponse(
            answer=answer,
            clarifying_questions=clarifying,
            safety_flags=flags,
            confidence=confidence,
            used_articles=[UsedArticle(**a) for a in used_articles],
            show_escalation=True,
            message_id=message_id,
        )


@app.post("/v1/escalations", response_model=EscalationResponse)
def create_escalation(req: EscalationRequest, request: Request):
    ticket_id = str(uuid.uuid4())
    open_now = is_business_hours_now()

    pref = (req.preferred_contact or "").strip().lower()
    wants_email = pref == "email" or bool((req.email or "").strip())

    # Routing rules:
    # - After hours: email only
    # - During business hours: default to live chat, but email is available if user chose Email
    routing = "email" if (not open_now) else ("email" if wants_email else "chat")

    conversation_id: Optional[str] = None
    transcript = ""
    ctx: Dict[str, Any] = {}

    with db() as conn:
        sess = get_session(conn, req.session_id)
        history = get_recent_messages(conn, req.session_id, limit=250)
        transcript = format_transcript(history, max_chars=9000)

        ctx = {
            "airstream_year": sess.get("airstream_year"),
            "category": sess.get("category"),
            "history": history,
        }

        # ✅ Keep existing Postgres insert (does not break current functionality)
        try:
            exec_no_return(
                conn,
                """
                INSERT INTO escalations (id, session_id, name, phone, email, message, preferred_contact, context_json)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    ticket_id,
                    req.session_id,
                    req.name,
                    req.phone,
                    req.email,
                    req.message,
                    req.preferred_contact,
                    json.dumps(ctx),
                ),
            )
        except Exception:
            # Some deployments may not have the Postgres escalations table; Supabase is the new source of truth.
            pass

        # If we're open and routing to chat, ensure a live chat conversation exists so the owner can jump in.
        if routing == "chat":
            try:
                conversation_id = get_or_create_conversation_for_session(req.session_id)
            except Exception:
                conversation_id = None

        # Optional reset: start a fresh troubleshooting state after escalation
        if req.reset_old:
            try:
                updates = []
                if sessions_supports_pinning(conn):
                    updates.append("active_article_id=NULL, active_node_id=NULL, active_tree=NULL")
                if sessions_supports_active_question(conn):
                    updates.append("active_question_text=NULL")
                if updates:
                    exec_no_return(conn, f"UPDATE sessions SET {', '.join(updates)} WHERE id=%s", (req.session_id,))
            except Exception:
                pass

        conn.commit()

    # ✅ Supabase escalation row (admin dashboard source of truth)
    escalation_row = {
        "id": ticket_id,
        "session_id": req.session_id,
        "name": (req.name or "").strip(),
        "phone": (req.phone or "").strip(),
        "email": (req.email or "").strip(),
        "message": (req.message or "").strip(),
        "preferred_contact": (req.preferred_contact or "").strip(),
        "status": "open",
        "routing": routing,
        "business_hours": open_now,
        "conversation_id": conversation_id,
        "context_json": ctx,
        "transcript": transcript,
    }

    try:
        _ = sb_post("escalations", [escalation_row])
    except Exception:
        # Don't block the customer UX if Supabase has a hiccup; the ticket_id still exists.
        pass

    # Notify owner via push for faster response (optional; safe if token missing)
    try:
        token = get_owner_push_token(OWNER_SUPABASE_USER_ID) if OWNER_SUPABASE_USER_ID else None
        if token:
            send_expo_push(
                token,
                title="New escalation",
                body=(req.message or "").strip()[:120] or "New escalation received",
                data={"session_id": req.session_id, "ticket_id": ticket_id, "conversation_id": conversation_id},
            )
    except Exception:
        pass

    email_subject = build_escalation_email_subject(req.session_id)
    email_body = build_escalation_email_body(req=req, transcript=transcript, business_hours=open_now)

    return {
        "ticket_id": ticket_id,
        "escalation_id": ticket_id,
        "routing": routing,
        "business_hours": open_now,
        "conversation_id": conversation_id,
        "email_to": SUPPORT_EMAIL_TO,
        "email_subject": email_subject,
        "email_body": email_body,
    }



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


@app.post("/v1/admin/articles")
def admin_create_article(req: AdminArticleRequest, x_admin_key: str = Header(default="", alias="X-Admin-Key")):
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


@app.get("/v1/admin/livechat/conversations")
def admin_livechat_conversations(x_admin_key: str = Header(default="", alias="X-Admin-Key")):
    require_admin(x_admin_key)

    msg_rows = sb_get(
        "messages",
        {"select": "id,conversation_id,sender_role,body,created_at", "order": "created_at.desc", "limit": "300"},
    )

    conv_ids: List[str] = []
    last_by_conv: Dict[str, Dict[str, Any]] = {}

    for r in (msg_rows or []):
        cid = r.get("conversation_id")
        if not cid or cid in last_by_conv:
            continue
        last_by_conv[cid] = {"sender_role": r.get("sender_role"), "body": r.get("body"), "created_at": r.get("created_at")}
        conv_ids.append(cid)
        if len(conv_ids) >= 50:
            break

    if not conv_ids:
        return {"conversations": []}

    conv_rows = sb_get("conversations", {"select": "id,customer_id,created_at", "id": f"in.({','.join(conv_ids)})"})
    by_id = {c.get("id"): c for c in (conv_rows or [])}

    conversations: List[Dict[str, Any]] = []
    for cid in conv_ids:
        c = by_id.get(cid) or {}
        conversations.append({"conversation_id": cid, "customer_id": c.get("customer_id") or "", "last_message": last_by_conv.get(cid)})

    return {"conversations": conversations}


@app.get("/v1/admin/livechat/history/{conversation_id}")
def admin_livechat_history(conversation_id: str, x_admin_key: str = Header(default="", alias="X-Admin-Key")):
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
def admin_livechat_send(req: AdminLiveChatSendRequest, x_admin_key: str = Header(default="", alias="X-Admin-Key")):
    require_admin(x_admin_key)
    sender_id = OWNER_SUPABASE_USER_ID or "owner"
    msg = supabase_insert_message(req.conversation_id, sender_id, "owner", req.body)
    return {"ok": True, "conversation_id": req.conversation_id, "message": msg}


# -------------------------
# Admin: Quality control (all AI sessions)
# -------------------------
@app.get("/v1/admin/sessions")
def admin_list_all_sessions(x_admin_key: str = Header(default="", alias="X-Admin-Key")):
    require_admin(x_admin_key)

    with db() as conn:
        cols = _get_sessions_columns(conn)

        select_parts = ["s.id AS session_id"]
        if "user_id" in cols:
            select_parts.append("s.user_id AS user_id")
        if "channel" in cols:
            select_parts.append("s.channel AS channel")
        if "mode" in cols:
            select_parts.append("s.mode AS mode")
        if "airstream_year" in cols:
            select_parts.append("s.airstream_year AS airstream_year")
        if "category" in cols:
            select_parts.append("s.category AS category")
        if "created_at" in cols:
            select_parts.append("s.created_at AS created_at")

        select_parts.append("lm.created_at AS last_message_at")
        select_parts.append("lm.content AS preview")

        sql = f"""
        SELECT {', '.join(select_parts)}
        FROM sessions s
        LEFT JOIN LATERAL (
          SELECT created_at, content
          FROM chat_messages
          WHERE session_id = s.id
          ORDER BY created_at DESC
          LIMIT 1
        ) lm ON TRUE
        ORDER BY lm.created_at DESC NULLS LAST
        LIMIT 300
        """

        rows = exec_all(conn, sql, ())

    out = []
    for r in (rows or []):
        ptxt = (r.get("preview") or "").strip()
        r["preview"] = (ptxt[:140] + "…") if len(ptxt) > 140 else ptxt
        out.append(r)

    return {"sessions": out}


@app.delete("/v1/admin/sessions/{session_id}")
def admin_delete_session(session_id: str, x_admin_key: str = Header(default="", alias="X-Admin-Key")):
    require_admin(x_admin_key)

    with db() as conn:
        _ = get_session(conn, session_id)  # ensure exists

        exec_no_return(conn, "DELETE FROM chat_messages WHERE session_id=%s", (session_id,))
        exec_no_return(conn, "DELETE FROM sessions WHERE id=%s", (session_id,))
        conn.commit()

    return {"ok": True}


# -------------------------
# Admin: delete a live chat conversation (Supabase)
# -------------------------
@app.delete("/v1/admin/livechat/conversations/{conversation_id}")
def admin_delete_livechat_conversation(conversation_id: str, x_admin_key: str = Header(default="", alias="X-Admin-Key")):
    require_admin(x_admin_key)

    _ = sb_delete("messages", {"conversation_id": f"eq.{conversation_id}"}, prefer="return=minimal")
    _ = sb_delete("conversations", {"id": f"eq.{conversation_id}"}, prefer="return=minimal")

    return {"ok": True}
