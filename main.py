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
import hashlib
from datetime import datetime, timezone


import smtplib
from email.message import EmailMessage


import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from openai_client import embed_text, generate_answer, generate_checkpoint_summary

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
PUSH_TOKEN_SALT = os.getenv("PUSH_TOKEN_SALT", "vinniesbrain_push_tokens")

# =========================
# RevenueCat (Subscriptions)
# =========================
REVENUECAT_WEBHOOK_AUTH = os.getenv("REVENUECAT_WEBHOOK_AUTH", "").strip()
REVENUECAT_SECRET_API_KEY_V1 = os.getenv("REVENUECAT_SECRET_API_KEY_V1", "").strip()
REVENUECAT_ENTITLEMENT_ID = os.getenv("REVENUECAT_ENTITLEMENT_ID", "Vinnies Brain Pro").strip()

# Toggle for gating paid subscription checks.
# Set SUBSCRIPTION_GATE_ENABLED=0 to disable (recommended while wiring up Android / profiles).
SUBSCRIPTION_GATE_ENABLED = os.getenv("SUBSCRIPTION_GATE_ENABLED", "0").strip() == "1"


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
POOL_MAX_SIZE = int(os.getenv("DB_POOL_MAX", "15"))

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
    """Get a DB connection from the pool.

    Supabase/Postgres can drop idle TCP connections. If we return a dead
    connection to the pool, the next request can crash with:
      psycopg.OperationalError: server closed the connection unexpectedly

    So:
    - If we hit a connection-level OperationalError, we DISCARD the conn.
    - We rollback best-effort so pooled conns don't come back INTRANS.
    """

    conn = DB_POOL.getconn()
    bad = False
    try:
        conn.row_factory = dict_row
        yield conn
    except psycopg.OperationalError:
        bad = True
        raise
    finally:
        # Rollback best-effort (skip if broken/closed).
        try:
            if not bad and getattr(conn, "closed", 0) == 0:
                conn.rollback()
        except Exception:
            pass

        # Return to pool OR discard.
        try:
            if bad or getattr(conn, "closed", 0) != 0:
                # psycopg_pool supports close=True; if not, fallback to conn.close().
                try:
                    DB_POOL.putconn(conn, close=True)  # type: ignore[arg-type]
                except TypeError:
                    try:
                        conn.close()
                    except Exception:
                        pass
                except Exception:
                    try:
                        conn.close()
                    except Exception:
                        pass
            else:
                DB_POOL.putconn(conn)
        except Exception:
            try:
                conn.close()
            except Exception:
                pass


def run_db_transaction(fn, attempts: int = 2):
    """Run a DB transaction with a small retry for transient disconnects."""
    last_err = None
    for i in range(max(1, attempts)):
        try:
            with db() as conn:
                out = fn(conn)
                conn.commit()
                return out
        except psycopg.OperationalError as e:
            last_err = e
            logger.warning("DB OperationalError (attempt %s/%s): %s", i + 1, attempts, e)
            continue
    raise last_err  # type: ignore[misc]

# -------------------------
# Telemetry (optional)
# -------------------------
def _telemetry_insert(conn, session_id: str, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
    """Best-effort telemetry. Will not raise if table doesn't exist."""
    payload = payload or {}
    try:
        exec_no_return(
            conn,
            "INSERT INTO telemetry_events (session_id, event_type, payload) VALUES (%s, %s, %s)",
            (session_id, event_type, json.dumps(payload)),
        )
    except Exception:
        return



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
SUPPORT_EMAIL_TO = os.getenv("ESCALATION_EMAIL") or os.getenv("SUPPORT_EMAIL_TO", "BristonLowell@gmail.com")

# Gmail SMTP (optional). If configured, the server will send escalation emails automatically.
GMAIL_SMTP_USER = os.getenv("GMAIL_SMTP_USER", "").strip()
GMAIL_SMTP_APP_PASSWORD = os.getenv("GMAIL_SMTP_APP_PASSWORD", "").strip()

def send_escalation_email(to_email: str, subject: str, body: str) -> bool:
    """Best-effort Gmail SMTP send.

    Returns True if sent, False otherwise. This is intentionally non-blocking for
    customer UX: failures are logged and the API still returns success.
    """
    if not to_email:
        return False
    if not GMAIL_SMTP_USER or not GMAIL_SMTP_APP_PASSWORD:
        return False

    msg = EmailMessage()
    msg["From"] = GMAIL_SMTP_USER
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(GMAIL_SMTP_USER, GMAIL_SMTP_APP_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        try:
            print(f"SMTP send failed: {e}")
        except Exception:
            pass
        return False


@app.post("/v1/billing/revenuecat/webhook")
async def revenuecat_webhook(request: Request, authorization: str = Header(default="", alias="Authorization")):
    """
    RevenueCat recommends setting an Authorization header value in their dashboard
    and validating it in your server:contentReference[oaicite:12]{index=12}.
    """
    if not REVENUECAT_WEBHOOK_AUTH:
        raise HTTPException(status_code=500, detail="REVENUECAT_WEBHOOK_AUTH is not set on the server.")

    if (authorization or "").strip() != REVENUECAT_WEBHOOK_AUTH:
        raise HTTPException(status_code=401, detail="Unauthorized webhook")

    body = await request.json()

    # app_user_id should be the stable ID you log into RevenueCat with.
    # Best practice for your setup: use the Supabase user id (UUID) as RevenueCat app_user_id.
    app_user_id = (body.get("app_user_id") or "").strip()
    if not app_user_id:
        return {"ok": True, "ignored": True}

    # Fetch canonical status from RevenueCat API (don’t trust event ordering alone)
    sub = rc_get_subscriber(app_user_id)
    is_active, expires_at = rc_entitlement_status(sub)

    # Store in Supabase profiles
    upsert_profile_subscription(
        user_id=app_user_id,
        is_subscribed=is_active,
        expires_at_iso=expires_at,
        rc_app_user_id=app_user_id,
    )

    return {"ok": True}


@app.get("/v1/billing/status")
def billing_status(request: Request):
    """
    App can call this to quickly learn if the current user is subscribed.
    Uses your existing temporary auth bridge: X-User-Id:contentReference[oaicite:13]{index=13}.
    """
    user_id = (request.headers.get("X-User-Id") or "").strip()
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated (missing X-User-Id)")

    rows = sb_get("profiles", {"select": "id,is_subscribed,subscription_expires_at,rc_app_user_id", "id": f"eq.{user_id}", "limit": "1"})
    row = (rows or [{}])[0] if isinstance(rows, list) and rows else {}
    return {
    "user_id": user_id,
    "is_subscribed": bool(row.get("is_subscribed")),
    "subscription_expires_at": row.get("subscription_expires_at"),
}



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
            
from datetime import datetime, timezone

def _parse_rc_expires_at(expires_str: str | None):
    """
    Convert RevenueCat ISO8601 expiry string into UTC ISO string.
    Returns None if missing or invalid.
    """
    if not expires_str:
        return None

    try:
        # Convert trailing Z → +00:00 for Python
        if expires_str.endswith("Z"):
            expires_str = expires_str[:-1] + "+00:00"

        dt = datetime.fromisoformat(expires_str)

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return None

def rc_get_subscriber(app_user_id: str) -> dict:
    """
    Calls RevenueCat API v1 to fetch subscriber status.
    Auth is via Authorization: Bearer <secret_api_key>.

    """
    if not requests:
        raise HTTPException(status_code=500, detail="requests library not available on server.")
    if not REVENUECAT_SECRET_API_KEY_V1:
        raise HTTPException(status_code=500, detail="REVENUECAT_SECRET_API_KEY_V1 is not set on the server.")

    url = f"https://api.revenuecat.com/v1/subscribers/{app_user_id}"
    headers = {
        "Authorization": f"Bearer {REVENUECAT_SECRET_API_KEY_V1}",
        "Content-Type": "application/json",
    }
    r = requests.get(url, headers=headers, timeout=15)
    if not r.ok:
        raise HTTPException(status_code=502, detail=f"RevenueCat API failed: {r.status_code} {r.text}")
    return r.json()

def rc_entitlement_status(subscriber_payload: dict) -> tuple[bool, str | None]:
    """
    Returns (is_active, expires_at_iso).
    RevenueCat returns entitlements in subscriber payload.
    """
    sub = (subscriber_payload or {}).get("subscriber") or {}
    ents = sub.get("entitlements") or {}
    ent = ents.get(REVENUECAT_ENTITLEMENT_ID) or {}
    expires_at = _parse_rc_expires_at(ent.get("expires_date") or "")
    is_active = bool(ent.get("active"))
    return is_active, expires_at

def upsert_profile_subscription(user_id: str, is_subscribed: bool, expires_at_iso: str | None, rc_app_user_id: str | None = None):
    """
    Stores subscription state in Supabase profiles.
    Assumes you already created the columns in Step 2.
    """
    payload = [{
        "id": user_id,
        "is_subscribed": bool(is_subscribed),
        "subscription_expires_at": expires_at_iso,
        "rc_app_user_id": (rc_app_user_id or user_id),
        "subscription_source": "revenuecat",
        "subscription_updated_at": datetime.now(timezone.utc).isoformat(),
    }]
    sb_upsert("profiles", payload, on_conflict="id")



# =========================
# Subscription enforcement helper (NEW)
# =========================
from datetime import datetime, timezone

def enforce_subscription_if_enabled(user_id: str) -> None:
    """Enforce paid subscription only when SUBSCRIPTION_GATE_ENABLED=1.

    Uses profiles.user_id + is_pro + pro_expires_at.
    """
    if not SUBSCRIPTION_GATE_ENABLED:
        return

    uid = str(user_id or "").strip()
    if not uid:
        return

    rows = sb_get(
        "profiles",
        {
            "select": "user_id,is_pro,pro_expires_at",
            "user_id": f"eq.{uid}",   # ✅ correct column
            "limit": "1",
        },
    )

    # If the profile row doesn't exist yet, treat as not pro (or you can auto-create)
    row = rows[0] if isinstance(rows, list) and rows else None
    if not row:
        raise HTTPException(status_code=402, detail="Subscription required")

    is_pro = bool(row.get("is_pro"))

    # Optional: handle expiry if you use it
    expires_at = row.get("pro_expires_at")
    if expires_at:
        try:
            # Supabase returns ISO8601; datetime.fromisoformat handles "YYYY-MM-DDTHH:MM:SS+00:00"
            exp = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            if exp <= datetime.now(timezone.utc):
                is_pro = False
        except Exception:
            # If parsing fails, fall back to is_pro flag only
            pass

    if not is_pro:
        raise HTTPException(status_code=402, detail="Subscription required")




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


class CheckpointSummary(BaseModel):
    known: List[str] = []
    ruled_out: List[str] = []
    likely_causes: List[str] = []
    next_checks: List[str] = []


class ChatResponse(BaseModel):
    answer: str
    checkpoint_summary: Optional[CheckpointSummary] = None
    clarifying_questions: List[str]
    safety_flags: List[str]
    confidence: float
    used_articles: List[UsedArticle]
    show_escalation: bool
    message_id: str


class EscalationRequest(BaseModel):
    session_id: str
    name: Optional[str] = ""
    phone: Optional[str] = ""
    email: Optional[str] = ""
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

    # True if the server successfully emailed the escalation (requires Gmail SMTP env vars)
    emailed: Optional[bool] = None



class LiveChatSendRequest(BaseModel):
    session_id: str
    body: str


class LiveChatOpenedRequest(BaseModel):
    session_id: str


class AdminLiveChatSendRequest(BaseModel):
    conversation_id: str
    body: str


class OwnerPushTokenRequest(BaseModel):
    owner_id: str
    expo_push_token: str


class AdminPushTokenRequest(BaseModel):
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


def get_relevant_facts(conn, year: int | None, user_message: str):
    cur = conn.cursor()

    cur.execute("""
        select fact_text
        from kb_facts
        where
            (years_min is null or years_min <= %s)
        and
            (years_max is null or years_max >= %s)
        order by created_at desc
        limit 10
    """, (year, year))

    rows = cur.fetchall()
    return [r[0] for r in rows]


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
    """Rewrite very short replies so the model reliably anchors them to the last asked question.

    This is a pragmatic fix for "yes", "no", "only while driving", "sometimes", etc. where
    the user is clearly replying to the AI's most recent question.
    """
    q = (active_question_text or "").strip()
    if not q:
        return user_text

    # Strip markdown bold wrapper if UI stored it that way.
    if q.startswith("**") and q.endswith("**") and len(q) > 4:
        q = q[2:-2].strip()

    # If user is explicitly switching/resetting, do not anchor.
    if should_reset_flow(user_text):
        return user_text

    # If user is asking a new question (has '?') we also don't force-anchor.
    if "?" in (user_text or ""):
        return user_text

    yn = normalize_yes_no(user_text)
    if yn:
        return f'Answer to: "{q}" -> {yn}'

    # Anchor other short/ambiguous replies.
    compact = (user_text or "").strip()
    word_count = len(compact.split())
    if len(compact) <= 28 or word_count <= 4:
        return f'User is answering the previous question: "{q}". User reply: {compact}'

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
        
def _hash_admin_key(raw_key: str) -> str:
    """
    Hash the admin key before storing/looking up push tokens.
    This prevents your real admin key from being stored in Supabase.
    """
    key = (raw_key or "").strip()
    if not key:
        return ""

    # If no salt is set, fallback to hashing raw key (still better than storing plaintext),
    # but you SHOULD set PUSH_TOKEN_SALT in Render.
    salt = PUSH_TOKEN_SALT or ""
    material = f"{salt}:{key}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def get_admin_push_tokens_for_notifications() -> List[str]:
    """
    Returns ALL Expo push tokens registered for admin notifications.

    Why list?
    - Supports multiple admin devices
    - Avoids subtle mismatches where the server's ADMIN_API_KEY differs from the key used to register
      the token (X-Admin-Key), which would otherwise result in *no* notifications.
    """
    rows = sb_get("admin_push_tokens", {"select": "expo_push_token", "order": "updated_at.desc", "limit": "50"})
    out: List[str] = []
    for r in (rows or []):
        tok = (r or {}).get("expo_push_token")
        if isinstance(tok, str) and tok.strip():
            out.append(tok.strip())

    # de-dupe while preserving order
    seen: Set[str] = set()
    uniq: List[str] = []
    for t in out:
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    return uniq


def get_admin_push_token_for_notifications() -> Optional[str]:
    """
    Back-compat helper: returns the most recently registered admin token (if any).
    """
    toks = get_admin_push_tokens_for_notifications()
    return toks[0] if toks else None


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

    # ✅ Subscription gate (optional)
    if user_id:
        enforce_subscription_if_enabled(user_id)

    def _tx(conn):
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

        return {"session_id": sid}

    # Retry once if Supabase/Postgres dropped the pooled connection.
    return run_db_transaction(_tx, attempts=2)


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
    """
    IMPORTANT: This endpoint MUST NOT hold a DB pool connection while calling OpenAI/Supabase/other network APIs.
    We split into phases:
      1) DB read + quick session updates
      2) Slow network calls (OpenAI)
      3) DB write (log messages, store active_question_text)
    """

    # -------------------------
    # Phase 1: DB READ (fast)
    # -------------------------
    sess: Dict[str, Any] = {}
    year: Optional[int] = None
    category: Optional[str] = None
    flags: List[str] = []
    pin_supported = False
    aq_supported = False
    active_article_id = None
    active_node_id = None
    active_tree: Any = None
    active_question_text: Optional[str] = None
    pending_q: Optional[str] = None

    used_articles: List[Dict[str, str]] = []
    context_chunks: List[str] = []
    from_kb = False

    history: List[Dict[str, Any]] = []

    authoritative_facts: List[str] = []

    # We may need vector search; embedding must be computed outside DB
    need_vector_search = False

    # Store a couple things for later DB-write phase
    q_to_store: Optional[str] = None

    # Message id created early (stable for this response)
    message_id = str(uuid.uuid4())

    with db() as conn:
        sess = get_session(conn, req.session_id)

        # ✅ Subscription gate (optional) — keep here (Supabase REST call happens inside enforce_subscription_if_enabled,
        # but we are NOT holding DB during that call in this block; however enforce reads Supabase (network).
        # To keep DB clean, we only check the session user_id here and defer the network gate check outside DB.
        # We'll do that right after this DB phase.
        user_id = str(sess.get("user_id") or "").strip()

        year = req.airstream_year or sess.get("airstream_year")
        category = sess.get("category")

        flags = detect_safety_flags(req.message)

        # Greetings / not-airstream can return fast without OpenAI
        if is_greeting(req.message):
            # log user msg + assistant reply quickly
            log_message(conn, req.session_id, "user", req.message)
            log_message(conn, req.session_id, "assistant", "Hey! What Airstream issue are you dealing with today?")
            conn.commit()

            return ChatResponse(
                answer="Hey! What Airstream issue are you dealing with today?",
                clarifying_questions=[],
                safety_flags=flags,
                confidence=0.7,
                used_articles=[],
                show_escalation=False,
                message_id=message_id,
            )

        if not is_airstream_question(req.message, year):
            # log user msg + assistant reply quickly
            msg = "I can help with Airstream troubleshooting. What system are you working on (water, electrical, appliances, leaks, etc.)?"
            log_message(conn, req.session_id, "user", req.message)
            log_message(conn, req.session_id, "assistant", msg)
            conn.commit()

            return ChatResponse(
                answer=msg,
                clarifying_questions=[],
                safety_flags=flags,
                confidence=0.4,
                used_articles=[],
                show_escalation=False,
                message_id=message_id,
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

        # Parse stored tree (if any)
        if isinstance(active_tree, str):
            try:
                active_tree = json.loads(active_tree)
            except Exception:
                active_tree = None

        yn = normalize_yes_no(req.message)

        # If user says yes/no and we have a stored question, rewrite it so the model knows what it's answering.
        rewritten_user_message = rewrite_short_answer(req.message, active_question_text)

        use_pinned = bool(pin_supported and active_article_id and isinstance(active_tree, dict) and active_node_id and yn)

        # 🔥 Fetch authoritative micro-facts from DB (fast)
        try:
            authoritative_facts = get_relevant_facts(conn, year, req.message)
        except Exception:
            authoritative_facts = []

        # Step B: compute pending_q from the last stored assistant question
        pending_q = (active_question_text or "").strip()
        if pending_q.startswith("**") and pending_q.endswith("**") and len(pending_q) > 4:
            pending_q = pending_q[2:-2].strip()
        if not pending_q:
            pending_q = None

        # Build context (DB-only work)
        if use_pinned:
            next_node = advance_node(active_tree, active_node_id, req.message)

            # update session node
            exec_no_return(conn, "UPDATE sessions SET active_node_id=%s WHERE id=%s", (next_node, req.session_id))

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

                if (active_question_text or "").strip():
                    chunk += f'\nLAST_AI_QUESTION:\n{active_question_text.strip()}\n'

                context_chunks.append(chunk)
                from_kb = True

            conn.commit()

        else:
            # Keyword search first (no OpenAI embedding)
            ranked = keyword_kb_articles(conn, req.message, year, category, top_k=6)

            if ranked:
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
                conn.commit()
            else:
                # No keyword results; we may do vector search, but embedding must happen OUTSIDE DB
                need_vector_search = True
                conn.commit()

        # Pull recent messages (DB)
        history = get_recent_messages(conn, req.session_id, limit=50)

    # -------------------------
    # Phase 1b: Optional subscription gate (NETWORK) — outside DB
    # -------------------------
    # We re-check session user_id without holding DB (enforce uses Supabase REST).
    try:
        uid = str(sess.get("user_id") or "").strip()
        if uid:
            enforce_subscription_if_enabled(uid)
    except HTTPException:
        # re-raise subscription required etc.
        raise
    except Exception:
        # do not crash if subscription check fails unexpectedly
        pass

    # -------------------------
    # Phase 2: Optional vector search (needs OpenAI embedding) — NO DB HELD
    # -------------------------
    if need_vector_search:
        try:
            q_emb = embed(req.message)  # OpenAI embeddings (network)
        except Exception:
            q_emb = []

        if q_emb:
            with db() as conn:
                ranked2 = rank_kb_articles(conn, q_emb, year, category, top_k=6)

                for r, score in ranked2:
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
                if sessions_supports_pinning(conn) and from_kb and ranked2:
                    top = ranked2[0][0]
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
                        conn.commit()

    # Prepare rewritten_user_message again (we computed it inside DB phase, but keep it stable)
    rewritten_user_message = rewrite_short_answer(req.message, active_question_text)

    # -------------------------
    # Phase 2: LLM call (NO DB HELD)
    # -------------------------
    try:
        answer, clarifying, confidence = generate_answer(
            user_message=rewritten_user_message,
            context="\n\n---\n\n".join(context_chunks),
            safety_flags=flags,
            airstream_year=year,
            category=category,
            history=history,
            pending_question=pending_q or None,
            authoritative_facts=authoritative_facts,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    # Subtle check-in (only when from_kb)
    answer = maybe_append_subtle_checkin(
        answer=answer,
        clarifying=clarifying,
        confidence=confidence,
        from_kb=from_kb,
        history=history,
    )

    # Checkpoint summary (every 3 assistant messages per your code)
    checkpoint_summary = None
    try:
        assistant_count = sum(
            1 for h in (history or []) if (h.get("role") or "").strip().lower() == "assistant"
        )
        if (assistant_count + 1) % 3 == 0:
            checkpoint_summary = generate_checkpoint_summary(
                history=history,
                airstream_year=year,
                category=category,
            )
    except Exception:
        checkpoint_summary = None

    # Store last asked question (without ** wrappers)
    if aq_supported:
        q_to_store = (clarifying[0] if (isinstance(clarifying, list) and len(clarifying) > 0) else "").strip()
        if q_to_store.startswith("**") and q_to_store.endswith("**") and len(q_to_store) > 4:
            q_to_store = q_to_store[2:-2].strip()
        if not q_to_store:
            q_to_store = None

    # -------------------------
    # Phase 3: DB WRITE (fast)
    # -------------------------
    with db() as conn:
        # Update active_question_text (if supported)
        if sessions_supports_active_question(conn):
            exec_no_return(
                conn,
                "UPDATE sessions SET active_question_text=%s WHERE id=%s",
                (q_to_store, req.session_id),
            )

        # Log messages + telemetry
        log_message(conn, req.session_id, "user", req.message)
        _telemetry_insert(conn, req.session_id, "chat_user_message", {"len": len((req.message or ""))})

        log_message(conn, req.session_id, "assistant", answer)
        _telemetry_insert(conn, req.session_id, "chat_assistant_message", {"len": len((answer or "")), "from_kb": from_kb})

        if checkpoint_summary is not None:
            _telemetry_insert(conn, req.session_id, "checkpoint_generated", {})

        conn.commit()

    return ChatResponse(
        answer=answer,
        checkpoint_summary=CheckpointSummary(**checkpoint_summary) if isinstance(checkpoint_summary, dict) else None,
        clarifying_questions=clarifying,
        safety_flags=flags,
        confidence=confidence,
        used_articles=[UsedArticle(**a) for a in used_articles],
        show_escalation=True,
        message_id=message_id,
    )




@app.post("/v1/escalations", response_model=EscalationResponse)
def create_escalation(req: EscalationRequest):
    """Create an escalation ticket and persist it to Supabase (table: escalations).

    Important: we do NOT block the customer flow if Supabase insert fails.
    We return a ticket_id regardless so the frontend confirmation still works.

    NOTE: Supabase schema (public.escalations) uses:
      - issue_summary (NOT NULL) instead of "message"
      - contact (instead of separate email/phone columns)
    """
    ticket_id = str(uuid.uuid4())

    # Determine routing based on preferred_contact (keep loose to match existing client behavior)
    pc = (req.preferred_contact or "").strip().lower()
    if pc == "email":
        routing = "email"
    elif pc in {"both", "chat+email", "chat_and_email"}:
        routing = "both"
    else:
        routing = "chat"

    open_now = is_business_hours_now()

    conversation_id: Optional[str] = None
    if routing in {"chat", "both"}:
        try:
            conversation_id = get_or_create_conversation_for_session(req.session_id)
        except Exception:
            conversation_id = None

    # Pull a little context (optional) from local chat log
    airstream_year: Optional[int] = None
    excerpt: Optional[str] = None
    try:
        with db() as conn:
            srow = exec_one(conn, "SELECT airstream_year FROM sessions WHERE id=%s", (req.session_id,))
            if srow and srow.get("airstream_year") is not None:
                try:
                    airstream_year = int(srow.get("airstream_year"))
                except Exception:
                    airstream_year = None

            mrows = exec_all(
                conn,
                "SELECT role, content, created_at FROM chat_messages WHERE session_id=%s ORDER BY created_at DESC LIMIT 12",
                (req.session_id,),
            )
            if mrows:
                lines = []
                for r in reversed(mrows):
                    role = (r.get("role") or "").strip() or "user"
                    content = (r.get("content") or "").strip()
                    if content:
                        lines.append(f"{role}: {content}")
                joined = "\n".join(lines).strip()
                if joined:
                    excerpt = joined[:4000]
    except Exception:
        pass

    # Supabase escalations row (match your actual table schema)
    contact = ""
    if (req.email or "").strip():
        contact = (req.email or "").strip()
    elif (req.phone or "").strip():
        contact = (req.phone or "").strip()

    full_row = {
        "id": ticket_id,  # uuid string
        "session_id": req.session_id,  # uuid string
        "airstream_year": airstream_year,
        "issue_summary": (req.message or "").strip(),  # NOT NULL
        "name": (req.name or "").strip() or None,
        "contact": contact or None,
        "preferred_contact": (req.preferred_contact or "").strip() or None,
        "conversation_excerpt": excerpt,
        "status": "new",  # matches your table default
        "routing": routing,
        "business_hours": open_now,
        "conversation_id": conversation_id,
        "context_json": {
            "source": "mobile",
            "routing": routing,
            "business_hours": open_now,
        },
    }

    try:
        _ = sb_post("escalations", [full_row])
    except Exception as e:
        # Log for debugging; do not block customer UX
        try:
            print(f"Supabase escalation insert failed: {e}")
        except Exception:
            pass

    # Optional reset: start fresh after escalation
    if req.reset_old:
        try:
            with db() as conn:
                updates = []
                if sessions_supports_pinning(conn):
                    updates.append("active_article_id=NULL, active_node_id=NULL, active_tree=NULL")
                if sessions_supports_active_question(conn):
                    updates.append("active_question_text=NULL")
                if updates:
                    exec_no_return(conn, f"UPDATE sessions SET {', '.join(updates)} WHERE id=%s", (req.session_id,))
                    conn.commit()
        except Exception:
            pass

    email_subject = build_escalation_email_subject(req.session_id)
    email_subject = build_escalation_email_subject(req.session_id)

    transcript_block = ""
    if excerpt and excerpt.strip():
        transcript_block = (
            "\n\n"
            "Conversation transcript (most recent)\n"
            "------------------------------\n"
            f"{excerpt.strip()}"
        )

    email_body = (
        "Vinnies Brain — Escalation\n"
        f"Session ID: {req.session_id}\n"
        f"Airstream year: {airstream_year if airstream_year is not None else ''}\n"
        f"Business hours at submit: {'YES' if open_now else 'NO'}\n\n"
        "Customer info\n"
        f"Name: {req.name}\n"
        + (f"Email: {req.email}\n" if (req.email or '').strip() else "")
        + (f"Phone: {req.phone}\n" if (req.phone or '').strip() else "")
        + f"Preferred contact: {req.preferred_contact}\n\n"
        "Issue summary\n"
        + (req.message or "").strip()
        + transcript_block
    ).strip()


    emailed = send_escalation_email(SUPPORT_EMAIL_TO, email_subject, email_body)

    return {
        "ticket_id": ticket_id,
        "escalation_id": ticket_id,
        "routing": routing,
        "business_hours": open_now,
        "conversation_id": conversation_id,
        "email_to": SUPPORT_EMAIL_TO,
        "email_subject": email_subject,
        "email_body": email_body,
        "emailed": emailed,
    }


@app.post("/v1/owner/push-token")
def register_owner_push_token(req: OwnerPushTokenRequest):
    _ = sb_upsert(
        "owner_push_tokens",
        [{"owner_id": req.owner_id, "expo_push_token": req.expo_push_token}],
        on_conflict="owner_id",
    )
    return {"ok": True}


@app.post("/v1/admin/push-token")
def register_admin_push_token(
    req: AdminPushTokenRequest,
    x_admin_key: str = Header(default="", alias="X-Admin-Key"),
):
    """
    Register this admin device for Expo push notifications.

    Uses X-Admin-Key auth and stores the token keyed by a HASH of the admin key
    (not tied to Supabase auth users / OWNER_SUPABASE_USER_ID).
    """
    require_admin(x_admin_key)

    token = (req.expo_push_token or "").strip()
    if not token:
        raise HTTPException(status_code=400, detail="Missing expo_push_token")

    key_hash = _hash_admin_key(x_admin_key)
    if not key_hash:
        raise HTTPException(status_code=500, detail="Failed to hash admin key (check PUSH_TOKEN_SALT)")

    payload = [{
        "key_hash": key_hash,
        "expo_push_token": token,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }]

    _ = sb_upsert("admin_push_tokens", payload, on_conflict="key_hash")
    return {"ok": True}

def _parse_iso_dt(s: str) -> Optional[datetime]:
    try:
        if not s:
            return None
        # Supabase often returns Z for UTC
        s2 = s.replace("Z", "+00:00")
        return datetime.fromisoformat(s2)
    except Exception:
        return None


def _is_probable_duplicate_message(
    conversation_id: str,
    sender_role: str,
    sender_id: str,
    body: str,
    within_seconds: int = 5,
) -> Optional[Dict[str, Any]]:
    """
    Best-effort idempotency for live chat sends.

    If the client retries (or a tap is registered multiple times) we can end up inserting duplicates.
    We treat a message as duplicate if the most recent message for the same conversation/sender/body
    is very recent (within `within_seconds`).
    """
    try:
        rows = sb_get(
            "messages",
            {
                "select": "id,conversation_id,sender_id,sender_role,body,created_at",
                "conversation_id": f"eq.{conversation_id}",
                "sender_role": f"eq.{sender_role}",
                "sender_id": f"eq.{sender_id}",
                "body": f"eq.{body}",
                "order": "created_at.desc",
                "limit": "1",
            },
        )
        if not rows:
            return None

        last = rows[0] or {}
        dt = _parse_iso_dt(str(last.get("created_at") or ""))
        if not dt:
            return None

        now = datetime.now(timezone.utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        if abs((now - dt).total_seconds()) <= within_seconds:
            return last
    except Exception:
        return None

    return None


@app.post("/v1/livechat/opened")
def livechat_opened(req: LiveChatOpenedRequest):
    conversation_id = get_or_create_conversation_for_session(req.session_id)

    system_body = "Hello! Please tell us your name and we will be with you shortly."

    # Only insert if there are zero messages in this conversation
    existing = sb_get(
        "messages",
        {
            "select": "id",
            "conversation_id": f"eq.{conversation_id}",
            "order": "created_at.asc",
            "limit": "1",
        },
    )

    if not existing:
        supabase_insert_message(conversation_id, req.session_id, "system", system_body)

    rows = sb_get(
        "messages",
        {
            "select": "id,conversation_id,sender_id,sender_role,body,created_at",
            "conversation_id": f"eq.{conversation_id}",
            "order": "created_at.asc",
            "limit": "500",
        },
    )

    return {"ok": True, "conversation_id": conversation_id, "messages": rows or []}


@app.post("/v1/livechat/send")
def livechat_send(req: LiveChatSendRequest):
    conversation_id = get_or_create_conversation_for_session(req.session_id)

    body = (req.body or "").strip()
    if not body:
        raise HTTPException(status_code=400, detail="Missing body")

    # Best-effort idempotency: prevent accidental duplicate inserts (e.g., client retries)
    existing = _is_probable_duplicate_message(
        conversation_id=conversation_id,
        sender_role="customer",
        sender_id=req.session_id,
        body=body,
        within_seconds=6,
    )
    if existing:
        msg = existing
    else:
        msg = supabase_insert_message(conversation_id, req.session_id, "customer", body)

    # Notify ALL registered admin devices
    for token in get_admin_push_tokens_for_notifications():
        send_expo_push(
            token,
            title="New chat message",
            body=body[:120],
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



@app.get("/v1/admin/escalations")
def admin_list_escalations(
    status: str = "",
    x_admin_key: str = Header(default="", alias="X-Admin-Key"),
):
    require_admin(x_admin_key)

    # Your Supabase table: public.escalations (issue_summary/contact/etc.)
    params: Dict[str, str] = {
        "select": "id,session_id,airstream_year,issue_summary,location,trigger,name,contact,preferred_contact,conversation_excerpt,status,created_at,routing,business_hours,conversation_id,handled_at,transcript,context_json",
        "order": "created_at.desc",
        "limit": "300",
    }
    if status.strip():
        params["status"] = f"eq.{status.strip()}"

    rows = sb_get("escalations", params)

    # Back-compat mapping so existing admin-inbox.tsx can keep using message/email/phone fields
    out: List[Dict[str, Any]] = []
    for r in (rows or []):
        issue = (r.get("issue_summary") or "").strip()
        contact = (r.get("contact") or "").strip()

        r["message"] = r.get("message") or issue
        if contact:
            if "@" in contact and not r.get("email"):
                r["email"] = contact
            elif not r.get("phone"):
                r["phone"] = contact

        r["message_preview"] = (issue[:140] + "…") if len(issue) > 140 else issue
        out.append(r)

    return {"escalations": out}




class AdminUpdateEscalationRequest(BaseModel):
    status: Optional[str] = None  # open | in_progress | closed


@app.post("/v1/admin/escalations/{escalation_id}")
def admin_update_escalation(
    escalation_id: str,
    req: AdminUpdateEscalationRequest,
    x_admin_key: str = Header(default="", alias="X-Admin-Key"),
):
    require_admin(x_admin_key)

    payload: Dict[str, Any] = {"id": escalation_id}
    if req.status and req.status.strip():
        payload["status"] = req.status.strip()
        if req.status.strip() in {"in_progress", "closed"}:
            payload["handled_at"] = datetime.utcnow().isoformat()

    _ = sb_upsert("escalations", [payload], on_conflict="id")
    return {"ok": True}



@app.get("/v1/admin/livechat/conversations")
def admin_livechat_conversations(x_admin_key: str = Header(default="", alias="X-Admin-Key")):
    require_admin(x_admin_key)

    # 1) Pull the most recent conversations directly so a newly-opened chat shows up immediately,
    #    even before the customer sends a message.
    conv_rows = sb_get(
        "conversations",
        {
            "select": "id,customer_id,created_at",
            "order": "created_at.desc",
            "limit": "60",
        },
    )

    if not conv_rows:
        return {"conversations": []}

    conv_ids = [c.get("id") for c in conv_rows if c.get("id")]
    conv_ids = [c for c in conv_ids if isinstance(c, str) and c.strip()]

    if not conv_ids:
        return {"conversations": []}

    # 2) Fetch recent messages for ONLY these conversations (fast + deterministic)
    #    We'll compute "last message" per conversation.
    msg_rows = sb_get(
        "messages",
        {
            "select": "id,conversation_id,sender_role,body,created_at",
            "conversation_id": f"in.({','.join(conv_ids)})",
            "order": "created_at.desc",
            "limit": "600",
        },
    )

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

    # 3) Build response in the same order as conversations table
    out: List[Dict[str, Any]] = []
    for c in conv_rows:
        cid = c.get("id")
        if not cid:
            continue
        out.append(
            {
                "conversation_id": cid,
                "customer_id": c.get("customer_id") or "",
                "created_at": c.get("created_at"),
                "last_message": last_by_conv.get(cid),  # can be None (still shows up!)
            }
        )

    return {"conversations": out}


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


