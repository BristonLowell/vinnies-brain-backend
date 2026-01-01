import os
import uuid
import json
import smtplib
from fastapi import Header
import re
# import psycopg2.extras
from email.message import EmailMessage
from typing import List, Optional, Dict, Any, Tuple
from typing import Any


from psycopg import connect
from psycopg.rows import dict_row

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import traceback
import psycopg
from psycopg.rows import dict_row
from contextlib import contextmanager
from openai_client import embed_text, generate_answer

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/vinniesbrain"
)
ESCALATION_EMAIL = os.getenv("ESCALATION_EMAIL", "bristonlowell@gmail.com")

TOP_K = int(os.getenv("TOP_K", "5"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))

SYSTEM_PROMPT = """You are “Vinnie’s Brain,” a customer-first troubleshooting assistant for Airstream trailers from model years 2010–2025.

PRIMARY GOAL
Help customers troubleshoot safely and clearly using ONLY the provided knowledge base context (retrieved articles). Do not guess. Do not invent steps, products, or causes that are not supported by the retrieved context.

STRICT GROUNDING RULE
- If the retrieved context does not contain a verified answer for the user’s situation, say:
  “I don’t have a verified Airstream-specific answer for that yet in Vinnie’s Brain.”
  Then offer escalation to: bristonlowell@gmail.com.
- Never provide definitive diagnoses without support in context.
- Never provide instructions that could cause damage or safety risk.

TONE
Technical-but-clear, calm, concise. Use short paragraphs and numbered steps.

MANDATORY RESPONSE STRUCTURE
1) One-sentence summary of what may be happening (must be consistent with retrieved context).
2) Ask up to 2–3 clarifying questions ONLY if needed to choose the right path (e.g., active leak vs stains, location, rain vs washing, model year if missing).
3) Provide numbered steps from the knowledge base.
4) Provide “Stop & contact us” conditions (especially for water intrusion, soft floors, electrical exposure).
5) Provide next step: either monitor, or escalate to bristonlowell@gmail.com.

SAFETY & SCOPE
- Prioritize water intrusion risks over cosmetic issues.
- If user mentions active dripping, soft walls/floors, mold, electrical fixtures getting wet, or repeated recurrence: instruct them to stop and contact bristonlowell@gmail.com.
- Do not recommend harsh chemicals, acids, or abrasive pads unless explicitly supported by retrieved context (generally avoid).
- Do not discuss pricing, warranty determinations, or repairs requiring disassembly beyond simple inspection.

OUTPUT REQUIREMENTS
- Use numbered steps for actions.
- Keep answers customer-safe and avoid internal-only jargon.
- If asked about years outside 2010–2025, state the supported range and offer escalation.
"""

app = FastAPI(title="Vinnie's Brain API", version="0.1.0")


# -------------------------
# Debug exception handler (returns traceback as JSON)
# -------------------------
@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "trace": traceback.format_exc()},
    )


@app.get("/v1/sessions/{session_id}")
def session_exists(session_id: str):
    with db() as conn:
        sess = get_session(conn, session_id)  # your existing function
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"ok": True}


# -------------------------
# Models
# -------------------------
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")

class AdminArticleUpsertRequest(BaseModel):
    id: Optional[str] = None
    title: str
    category: str
    severity: str
    years_min: int
    years_max: int
    customer_summary: str
    clarifying_questions: List[str] = []
    steps: List[str] = []
    model_year_notes: List[str] = []
    stop_and_escalate: List[str] = []
    next_step: str

def require_admin(x_admin_key: str | None):
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=500, detail="ADMIN_API_KEY not set")
    if not x_admin_key or x_admin_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

def make_retrieval_text_from_admin(a: AdminArticleUpsertRequest) -> str:
    return "\n".join([
        f"Title: {a.title}",
        f"Category: {a.category}",
        f"Severity: {a.severity}",
        f"Applies: {a.years_min}-{a.years_max}",
        f"Summary: {a.customer_summary}",
        "Clarifying Questions: " + " | ".join(a.clarifying_questions),
        "Steps: " + " | ".join(a.steps),
        "Model Year Notes: " + " | ".join(a.model_year_notes),
        "Stop & Escalate: " + " | ".join(a.stop_and_escalate),
        f"Next Step: {a.next_step}",
    ])


class CreateSessionRequest(BaseModel):
    channel: str = "mobile"
    mode: str = Field(default="customer", pattern="^(customer|staff)$")


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
    clarifying_questions: List[str] = []
    safety_flags: List[str] = []
    confidence: float
    used_articles: List[UsedArticle] = []
    show_escalation: bool = False
    message_id: str


class EscalationRequest(BaseModel):
    session_id: str
    airstream_year: Optional[int] = None
    issue_summary: str
    location: Optional[str] = None
    trigger: Optional[str] = None
    name: Optional[str] = None
    contact: Optional[str] = None
    preferred_contact: Optional[str] = None


class EscalationResponse(BaseModel):
    ticket_id: str


class FeedbackRequest(BaseModel):
    session_id: str
    message_id: Optional[str] = None
    rating: str = Field(pattern="^(up|down)$")
    note: Optional[str] = None


# -------------------------
# DB helpers (psycopg2 requires cursor.execute)
# -------------------------


def extract_steps(article: dict) -> list[str]:
    steps = article.get("troubleshooting_steps")

    # If already a list, use it
    if isinstance(steps, list):
        return [s.strip() for s in steps if s and s.strip()]

    # If it's a text blob, split lines and clean bullets/numbers
    if isinstance(steps, str):
        lines = [l.strip() for l in steps.splitlines() if l.strip()]
        cleaned = [re.sub(r"^[-•\d\)\.\s]+", "", l).strip() for l in lines]
        return [c for c in cleaned if len(c) > 2]

    return []


def build_context(original_issue: str, questions: list[str], answers: dict) -> str:
    parts = [f"Issue: {original_issue}"]
    for i, q in enumerate(questions):
        a = answers.get(str(i))
        if a:
            parts.append(f"Clarifying question: {q}\nUser answer: {a}")
    return "\n\n".join(parts)

def get_triage_state(conn, session_id: str) -> dict:
    row = conn.execute(
        "SELECT triage_state FROM sessions WHERE id=%s",
        (session_id,),
    ).fetchone()
    if not row:
        return {}
    # row can be a dict-row or tuple depending on your db() row_factory
    val = row["triage_state"] if isinstance(row, dict) else row[0]
    return val or {}

def set_triage_state(conn, session_id: str, state: dict):
    conn.execute(
        "UPDATE sessions SET triage_state=%s::jsonb WHERE id=%s",
        (json.dumps(state), session_id),
    )

def build_clarify_context(original_issue: str, questions: list[str], answers: dict) -> str:
    parts = [f"Issue: {original_issue}"]
    for i, q in enumerate(questions):
        a = answers.get(str(i), "").strip()
        if a:
            parts.append(f"Clarifying question: {q}\nUser answer: {a}")
    return "\n\n".join(parts)



@app.post("/v1/admin/kb/upsert")
def admin_kb_upsert(
    req: AdminArticleUpsertRequest,
    x_admin_key: str | None = Header(default=None),
):
    require_admin(x_admin_key)

    article_id = req.id or str(uuid.uuid4())
    retrieval_text = make_retrieval_text_from_admin(req)
    emb = embed_text(retrieval_text)

    with db() as conn:
        conn.execute(
            """
            INSERT INTO kb_articles
              (id, title, category, severity, years_min, years_max, customer_summary,
               clarifying_questions, steps, model_year_notes, stop_and_escalate, next_step,
               retrieval_text, embedding)
            VALUES
              (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s::jsonb,%s,%s,%s::vector)
            ON CONFLICT (id) DO UPDATE SET
              title=EXCLUDED.title,
              category=EXCLUDED.category,
              severity=EXCLUDED.severity,
              years_min=EXCLUDED.years_min,
              years_max=EXCLUDED.years_max,
              customer_summary=EXCLUDED.customer_summary,
              clarifying_questions=EXCLUDED.clarifying_questions,
              steps=EXCLUDED.steps,
              model_year_notes=EXCLUDED.model_year_notes,
              stop_and_escalate=EXCLUDED.stop_and_escalate,
              next_step=EXCLUDED.next_step,
              retrieval_text=EXCLUDED.retrieval_text,
              embedding=EXCLUDED.embedding,
              updated_at=now()
            """,
            (
                article_id,
                req.title, req.category, req.severity,
                req.years_min, req.years_max,
                req.customer_summary,
                json.dumps(req.clarifying_questions),
                json.dumps(req.steps),
                json.dumps(req.model_year_notes),
                json.dumps(req.stop_and_escalate),
                req.next_step,
                retrieval_text,
                emb,
            ),
        )
        conn.commit()

    return {"ok": True, "id": article_id}


@contextmanager
def db():
    conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
    try:
        yield conn
    finally:
        conn.close()



def exec_one(conn, sql: str, params=()):
    cur = conn.cursor()
    cur.execute(sql, params)
    row = cur.fetchone()
    cur.close()
    return row


def exec_all(conn, sql: str, params=()) -> List[Dict[str, Any]]:
    """
    Execute a query and return all rows as dictionaries.
    Requires psycopg v3 connection created with row_factory=dict_row.
    """
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def exec_no_return(conn, sql: str, params=()):
    cur = conn.cursor()
    cur.execute(sql, params)
    cur.close()


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
# Email (Gmail SMTP)
# -------------------------
def send_escalation_email(to_email: str, subject: str, body: str) -> None:
    smtp_user = os.getenv("GMAIL_SMTP_USER")
    smtp_pass = os.getenv("GMAIL_SMTP_APP_PASSWORD")
    if not smtp_user or not smtp_pass:
        raise RuntimeError("Missing GMAIL_SMTP_USER or GMAIL_SMTP_APP_PASSWORD in .env")

    msg = EmailMessage()
    msg["From"] = smtp_user
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.ehlo()
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)


# -------------------------
# Retrieval helpers
# -------------------------
def build_kb_context(articles: List[Dict[str, Any]]) -> str:
    parts = [
        "VERIFIED KNOWLEDGE BASE CONTEXT (use this only):",
        "- If the answer is not present here, you must say you do not have a verified answer.",
        ""
    ]
    for i, a in enumerate(articles, start=1):
        parts.append(f"[ARTICLE {i}]")
        parts.append(f"Title: {a['title']}")
        parts.append(f"Applies: {a['years_min']}–{a['years_max']}")
        parts.append(f"Category: {a['category']}")
        parts.append(f"Severity: {a['severity']}")
        parts.append(f"Summary: {a['customer_summary']}")
        parts.append(f"Clarifying Questions: {a['clarifying_questions']}")
        parts.append(f"Steps: {a['steps']}")
        parts.append(f"Model Year Notes: {a['model_year_notes']}")
        parts.append(f"Stop & Escalate: {a['stop_and_escalate']}")
        parts.append(f"Next Step: {a['next_step']}")
        parts.append("")
    return "\n".join(parts)


def retrieve_articles(
    conn,
    query_embedding: List[float],
    airstream_year: Optional[int],
    top_k: int
) -> List[Tuple[Dict[str, Any], float]]:
    if airstream_year is not None:
        sql = """
          SELECT
            id, title, category, severity, years_min, years_max, customer_summary,
            clarifying_questions, steps, model_year_notes, stop_and_escalate, next_step,
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
            1 - (embedding <=> %s::vector) AS score
          FROM kb_articles
          WHERE embedding IS NOT NULL
          ORDER BY embedding <=> %s::vector
          LIMIT %s
        """
        params = (query_embedding, query_embedding, top_k)

    rows = exec_all(conn, sql, params)
    return [(r, float(r["score"])) for r in rows]


def detect_safety_flags(user_text: str) -> List[str]:
    t = user_text.lower()
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

def get_triage_state(conn, session_id: str) -> dict:
    row = conn.execute(
        "SELECT triage_state FROM sessions WHERE id=%s",
        (session_id,),
    ).fetchone()
    if not row:
        return {}
    val = row["triage_state"] if isinstance(row, dict) else row[0]
    return val or {}

def set_triage_state(conn, session_id: str, state: dict):
    conn.execute(
        "UPDATE sessions SET triage_state=%s::jsonb WHERE id=%s",
        (json.dumps(state), session_id),
    )

def json_list(val) -> list:
    """
    Handles JSONB returned as:
      - Python list (ideal)
      - JSON string like '["a","b"]' (your current case)
      - None
    """
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

def format_clarify(q_index: int, total: int, question: str) -> str:
    return f"Quick question ({q_index+1}/{total}): {question}"

def format_step(step_index: int, total: int, step_text: str) -> str:
    return (
        f"Step {step_index+1}/{total}: {step_text}\n\n"
        "What did you find / what changed?"
    )



# -------------------------
# Routes
# -------------------------
@app.post("/v1/sessions", response_model=CreateSessionResponse)
def create_session(req: CreateSessionRequest):
    sid = str(uuid.uuid4())
    with db() as conn:
        exec_no_return(
            conn,
            "INSERT INTO sessions (id, mode, channel) VALUES (%s, %s, %s)",
            (sid, req.mode, req.channel),
        )
        conn.commit()
    return CreateSessionResponse(session_id=sid)


@app.post("/v1/sessions/{session_id}/context")
def update_context(session_id: str, req: UpdateContextRequest):
    with db() as conn:
        _ = get_session(conn, session_id)
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
        insert_message(conn, req.session_id, "user", req.message)

        year = req.airstream_year or sess.get("airstream_year")
        safety_flags = detect_safety_flags(req.message)

        state = get_triage_state(conn, req.session_id) or {}
        user_text = (req.message or "").strip()
        user_skip = user_text.lower() in ("skip", "skip question", "not sure", "idk", "unsure")

        # Escalate if safety flags are present
        show_escalation = any(
            f in safety_flags
            for f in ["active_water_intrusion", "structural_moisture_risk", "mold_risk", "electrical_risk"]
        )

        # ---------------------------------------------------------
        # A) COACH MODE: return ONE step at a time
        # ---------------------------------------------------------
        if state.get("stage") == "coach":
            if show_escalation:
                answer = (
                    "Before we continue: this could involve a safety risk.\n\n"
                    f"Please request help at {ESCALATION_EMAIL} and include your Airstream year and where the issue is."
                )
                msg_id = insert_message(conn, req.session_id, "assistant", answer, used_articles=[], confidence=1.0)
                conn.commit()
                return ChatResponse(
                    answer=answer,
                    clarifying_questions=[],
                    safety_flags=safety_flags,
                    confidence=1.0,
                    used_articles=[],
                    show_escalation=True,
                    message_id=msg_id,
                )

            # Optional: user can skip ahead to wrap-up
            if user_skip:
                set_triage_state(conn, req.session_id, {})
                answer = (
                    "No problem — here’s the next best move:\n\n"
                    f"{state.get('next_step') or 'If this persists, request help so a tech can trace the leak path.'}\n\n"
                    f"If you want, tell me: clean water or musty, and whether it happens after rain or after using water systems."
                )
                msg_id = insert_message(conn, req.session_id, "assistant", answer, used_articles=[], confidence=1.0)
                conn.commit()
                return ChatResponse(
                    answer=answer,
                    clarifying_questions=[],
                    safety_flags=safety_flags,
                    confidence=1.0,
                    used_articles=[],
                    show_escalation=False,
                    message_id=msg_id,
                )

            steps = state.get("steps", [])
            step_index = int(state.get("step_index", 0))
            article_id = state.get("article_id")
            article_title = state.get("article_title", "")

            if not steps or step_index >= len(steps):
                # wrap-up
                set_triage_state(conn, req.session_id, {})
                answer = (
                    "We’ve covered the standard troubleshooting steps.\n\n"
                    f"Next step: {state.get('next_step') or 'Request help so we can trace the leak path and inspect hidden areas.'}\n\n"
                    f"If it’s still wet/musty or returns after the next rain, contact us at {ESCALATION_EMAIL}."
                )
                msg_id = insert_message(conn, req.session_id, "assistant", answer, used_articles=[], confidence=1.0)
                conn.commit()
                return ChatResponse(
                    answer=answer,
                    clarifying_questions=[],
                    safety_flags=safety_flags,
                    confidence=1.0,
                    used_articles=[],
                    show_escalation=True,
                    message_id=msg_id,
                )

            answer = format_step(step_index, len(steps), steps[step_index])

            # advance step index
            state["step_index"] = step_index + 1
            set_triage_state(conn, req.session_id, state)

            used_articles = []
            if article_id and article_title:
                used_articles = [{"id": article_id, "title": article_title}]

            msg_id = insert_message(conn, req.session_id, "assistant", answer, used_articles=used_articles, confidence=1.0)
            conn.commit()

            return ChatResponse(
                answer=answer,
                clarifying_questions=[],
                safety_flags=safety_flags,
                confidence=1.0,
                used_articles=[UsedArticle(**ua) for ua in used_articles] if used_articles else [],
                show_escalation=False,
                message_id=msg_id,
            )

        # ---------------------------------------------------------
        # B) CLARIFY MODE: ask ONE question at a time
        # ---------------------------------------------------------
        if state.get("stage") == "clarify":
            questions = state.get("questions", [])
            q_index = int(state.get("q_index", 0))
            answers = state.get("answers", {})

            # store answer for the question we just asked
            if not user_skip and q_index < len(questions):
                answers[str(q_index)] = user_text

            q_index += 1
            state["answers"] = answers
            state["q_index"] = q_index

            # ask next question (only one)
            if q_index < len(questions):
                next_q = questions[q_index]
                set_triage_state(conn, req.session_id, state)

                assistant_text = format_clarify(q_index, len(questions), next_q)
                msg_id = insert_message(conn, req.session_id, "assistant", assistant_text, used_articles=[], confidence=1.0)
                conn.commit()

                return ChatResponse(
                    answer=assistant_text,
                    clarifying_questions=[next_q],
                    safety_flags=safety_flags,
                    confidence=1.0,
                    used_articles=[],
                    show_escalation=False,
                    message_id=msg_id,
                )

            # clarification done -> re-retrieve with combined context, then start coach mode
            original_issue = state.get("original_issue", "")
            combined = original_issue
            if answers:
                combined += "\n\n" + "\n".join([f"{k}: {v}" for k, v in answers.items() if v])

            q_emb = embed_text(combined)
            retrieved = retrieve_articles(conn, q_emb, year, top_k=TOP_K)
            articles = [r[0] for r in retrieved]
            top_score = retrieved[0][1] if retrieved else 0.0

            if top_score < CONFIDENCE_THRESHOLD or not articles:
                set_triage_state(conn, req.session_id, {})
                answer = (
                    "I don’t have a verified Airstream-specific answer for that yet in Vinnie’s Brain.\n\n"
                    f"Because this could involve hidden damage or safety risk, please contact us at {ESCALATION_EMAIL} "
                    "and include your Airstream year, where the issue is happening, and when it occurs (rain/washing/travel)."
                )
                msg_id = insert_message(conn, req.session_id, "assistant", answer, used_articles=[], confidence=top_score)
                conn.commit()
                return ChatResponse(
                    answer=answer,
                    clarifying_questions=[],
                    safety_flags=safety_flags,
                    confidence=top_score,
                    used_articles=[],
                    show_escalation=True,
                    message_id=msg_id,
                )

            top_article = articles[0]
            steps = json_list(top_article.get("steps"))
            questions = json_list(top_article.get("clarifying_questions"))
            stop_escalate = json_list(top_article.get("stop_and_escalate"))

            # start coach mode; step_index begins at 0
            coach_state = {
                "stage": "coach",
                "article_id": top_article.get("id"),
                "article_title": top_article.get("title"),
                "original_issue": original_issue or combined,
                "steps": steps,
                "step_index": 0,
                "next_step": top_article.get("next_step"),
                "stop_and_escalate": stop_escalate,
            }
            set_triage_state(conn, req.session_id, coach_state)

            # Return ONLY step 1 (not a big answer)
            if steps:
                first = format_step(0, len(steps), steps[0])
                coach_state["step_index"] = 1
                set_triage_state(conn, req.session_id, coach_state)

                used_articles = [{"id": top_article["id"], "title": top_article["title"]}]
                msg_id = insert_message(conn, req.session_id, "assistant", first, used_articles=used_articles, confidence=top_score)
                conn.commit()

                return ChatResponse(
                    answer=first,
                    clarifying_questions=[],
                    safety_flags=safety_flags,
                    confidence=top_score,
                    used_articles=[UsedArticle(**ua) for ua in used_articles],
                    show_escalation=False,
                    message_id=msg_id,
                )

            # If no steps exist, give a short “customer summary” and a single question
            set_triage_state(conn, req.session_id, {})
            answer = (
                f"{top_article.get('customer_summary')}\n\n"
                "Quick question: does this happen after rain/washing, or after using the water system?"
            )
            used_articles = [{"id": top_article["id"], "title": top_article["title"]}]
            msg_id = insert_message(conn, req.session_id, "assistant", answer, used_articles=used_articles, confidence=top_score)
            conn.commit()

            return ChatResponse(
                answer=answer,
                clarifying_questions=[],
                safety_flags=safety_flags,
                confidence=top_score,
                used_articles=[UsedArticle(**ua) for ua in used_articles],
                show_escalation=False,
                message_id=msg_id,
            )

        # ---------------------------------------------------------
        # C) NORMAL START: retrieve and decide clarify vs coach
        # ---------------------------------------------------------
        q_emb = embed_text(user_text)
        retrieved = retrieve_articles(conn, q_emb, year, top_k=TOP_K)
        articles = [r[0] for r in retrieved]
        top_score = retrieved[0][1] if retrieved else 0.0

        if top_score < CONFIDENCE_THRESHOLD or not articles:
            answer = (
                "I don’t have a verified Airstream-specific answer for that yet in Vinnie’s Brain.\n\n"
                f"Because this could involve hidden damage or safety risk, please contact us at {ESCALATION_EMAIL} "
                "and include your Airstream year, where the issue is happening, and when it occurs (rain/washing/travel)."
            )
            msg_id = insert_message(conn, req.session_id, "assistant", answer, used_articles=[], confidence=top_score)
            conn.commit()
            return ChatResponse(
                answer=answer,
                clarifying_questions=[],
                safety_flags=safety_flags,
                confidence=top_score,
                used_articles=[],
                show_escalation=True,
                message_id=msg_id,
            )

        top_article = articles[0]
        qs = json_list(top_article.get("clarifying_questions"))[:3]
        steps = json_list(top_article.get("steps"))
        stop_escalate = json_list(top_article.get("stop_and_escalate"))


        # If we have clarifying questions, start clarify flow
        if qs:
            new_state = {
                "stage": "clarify",
                "article_id": top_article.get("id"),
                "article_title": top_article.get("title"),
                "original_issue": user_text,
                "questions": qs,
                "q_index": 0,
                "answers": {},
            }
            set_triage_state(conn, req.session_id, new_state)

            first_q = qs[0]
            assistant_text = format_clarify(0, len(qs), first_q)
            msg_id = insert_message(conn, req.session_id, "assistant", assistant_text, used_articles=[], confidence=top_score)
            conn.commit()

            return ChatResponse(
                answer=assistant_text,
                clarifying_questions=[first_q],
                safety_flags=safety_flags,
                confidence=top_score,
                used_articles=[],
                show_escalation=False,
                message_id=msg_id,
            )

        # Otherwise start coach mode immediately
        coach_state = {
            "stage": "coach",
            "article_id": top_article.get("id"),
            "article_title": top_article.get("title"),
            "original_issue": user_text,
            "steps": steps,
            "step_index": 0,
            "next_step": top_article.get("next_step"),
            "stop_and_escalate": stop_escalate,
        }
        set_triage_state(conn, req.session_id, coach_state)

        if steps:
            first = format_step(0, len(steps), steps[0])
            coach_state["step_index"] = 1
            set_triage_state(conn, req.session_id, coach_state)

            used_articles = [{"id": top_article["id"], "title": top_article["title"]}]
            msg_id = insert_message(conn, req.session_id, "assistant", first, used_articles=used_articles, confidence=top_score)
            conn.commit()

            return ChatResponse(
                answer=first,
                clarifying_questions=[],
                safety_flags=safety_flags,
                confidence=top_score,
                used_articles=[UsedArticle(**ua) for ua in used_articles],
                show_escalation=False,
                message_id=msg_id,
            )

        # No steps? return a short summary + one question
        answer = (
            f"{top_article.get('customer_summary')}\n\n"
            "Quick question: does this happen after rain/washing, or after using the water system?"
        )
        used_articles = [{"id": top_article["id"], "title": top_article["title"]}]
        msg_id = insert_message(conn, req.session_id, "assistant", answer, used_articles=used_articles, confidence=top_score)
        conn.commit()

        return ChatResponse(
            answer=answer,
            clarifying_questions=[],
            safety_flags=safety_flags,
            confidence=top_score,
            used_articles=[UsedArticle(**ua) for ua in used_articles],
            show_escalation=False,
            message_id=msg_id,
        )




@app.post("/v1/escalations", response_model=EscalationResponse)
def create_escalation(req: EscalationRequest):
    eid = str(uuid.uuid4())
    with db() as conn:
        sess = get_session(conn, req.session_id)

        rows = exec_all(
            conn,
            """
            SELECT role, content, created_at
            FROM messages
            WHERE session_id=%s
            ORDER BY created_at DESC
            LIMIT 12
            """,
            (req.session_id,),
        )

        transcript = "\n".join([f"{r['role']}: {r['content']}" for r in reversed(rows)])

        exec_no_return(
            conn,
            """
            INSERT INTO escalations
              (id, session_id, airstream_year, issue_summary, location, trigger, name, contact, preferred_contact, conversation_excerpt)
            VALUES
              (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                eid,
                req.session_id,
                req.airstream_year or sess.get("airstream_year"),
                req.issue_summary,
                req.location,
                req.trigger,
                req.name,
                req.contact,
                req.preferred_contact,
                transcript[:8000],
            ),
        )

        subject = f"Vinnie's Brain Help Request — Airstream (Year: {req.airstream_year or sess.get('airstream_year') or 'Unknown'})"
        body = (
            f"Issue: {req.issue_summary}\n"
            f"Location: {req.location or 'Unknown'}\n"
            f"Trigger: {req.trigger or 'Unknown'}\n"
            f"Name: {req.name or 'Unknown'}\n"
            f"Contact: {req.contact or 'Unknown'}\n"
            f"Preferred contact: {req.preferred_contact or 'Unknown'}\n\n"
            "Conversation:\n"
            f"{transcript}\n"
        )

        send_escalation_email(ESCALATION_EMAIL, subject, body)
        conn.commit()

    return EscalationResponse(ticket_id=eid)


@app.post("/v1/feedback")
def feedback(req: FeedbackRequest):
    fid = str(uuid.uuid4())
    with db() as conn:
        _ = get_session(conn, req.session_id)
        exec_no_return(
            conn,
            "INSERT INTO feedback (id, session_id, message_id, rating, note) VALUES (%s, %s, %s, %s, %s)",
            (fid, req.session_id, req.message_id, req.rating, req.note),
        )
        conn.commit()
    return {"ok": True, "feedback_id": fid}
