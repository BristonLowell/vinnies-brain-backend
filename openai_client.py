# openai_client.py

import os
import json
from typing import List, Tuple, Optional, Any, Dict

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5.2")


def embed_text(text: str) -> List[float]:
    if not (text or "").strip():
        text = " "
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


def _history_to_text(history: Any, max_items: int = 20) -> str:
    """
    history from DB is expected to be a list of dicts like:
      [{"role":"user","text":"..."}, {"role":"assistant","text":"..."}]
    We convert to a compact transcript.
    """
    if not history or not isinstance(history, list):
        return ""
    lines: List[str] = []
    for h in history[-max_items:]:
        if not isinstance(h, dict):
            continue
        role = (h.get("role") or "").strip()
        text = (h.get("text") or "").strip()
        if not role or not text:
            continue
        lines.append(f"{role.upper()}: {text}")
    return "\n".join(lines).strip()


def generate_answer(
    *,
    user_message: str,
    context: str,
    safety_flags: Optional[List[str]] = None,
    airstream_year: Optional[int] = None,
    category: Optional[str] = None,
    history: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, List[str], float]:
    """
    Matches what main.py expects:
      returns (answer, clarifying_questions, confidence)
    """
    safety_flags = safety_flags or []
    transcript = _history_to_text(history)

    # SYSTEM-LEVEL RULE: Airstream-only assistant + refusal behavior
    # Keep it strict, but still return the same JSON schema your app expects.
    system_instructions = f"""
You are Vinnie's Brain, an AI assistant that ONLY provides information related to Airstream travel trailers (especially 2010–2025) and their systems, maintenance, troubleshooting, repairs, parts, ownership, and model-specific guidance.

Hard rule:
- If the user's question is NOT directly related to Airstream trailers (or the question cannot reasonably be interpreted as Airstream-related), you MUST refuse.

When refusing:
- Be polite and brief
- State that you only answer Airstream-related questions
- Invite the user to ask an Airstream-specific question
- Do NOT provide general advice or an off-topic answer
- Still return valid JSON in the required schema

Core behavior:
- Be technical-but-clear and calm.
- Ask for missing info ONLY when necessary.
- If a safety risk is present, lead with safety steps.
- Prefer the knowledge base context when present.

KNOWN CONTEXT:
- Airstream year: {airstream_year if airstream_year is not None else "unknown"}
- Category: {category or "unknown"}
- Safety flags: {", ".join(safety_flags) if safety_flags else "none"}

CRITICAL OUTPUT RULES (to prevent info-dumps):
1) ONE-QUESTION-AT-A-TIME:
   - Ask at most ONE clarifying question per message (unless no question is needed).
   - clarifying_questions MUST contain 0 or 1 items (never 2–3).

2) NO "NEXT STEPS" BEFORE THE USER ANSWERS:
   - If a clarifying question is needed, DO NOT provide multi-step solutions, likely causes lists, or detailed fix instructions yet.
   - In that case, the "answer" must be short and ONLY:
     (a) a one-sentence acknowledgement/summary,
     (b) the reason you need the detail (one short sentence),
     (c) optionally ONE immediate safety check ONLY if there is a real safety risk (electric, propane, active leak, smoke, overheating).
   - Then ask the single clarifying question (in clarifying_questions).

3) SHORTNESS LIMITS:
   - Keep the "answer" under ~90 words when asking a clarifying question.
   - Otherwise, keep the "answer" under ~160 words and max 6 bullets unless the user asks for more detail.

4) TROUBLESHOOTING FLOW:
   - Start with diagnosis questions.
   - Only after the user answers, provide up to 3 next steps, then (if needed) ask the next single question.

Return STRICT JSON with this schema:
{{
  "answer": string,
  "clarifying_questions": string[],
  "confidence": number
}}

Rules:
- confidence is 0.0 to 1.0 (higher when KB context is strong and question is specific).
- clarifying_questions: MUST be either [] or [one short question].
- If you include clarifying_questions, your "answer" MUST NOT include step-by-step fixes beyond a single safety check if needed.
""".strip()

    # Combine KB + history + message
    kb_block = (context or "").strip()
    hist_block = transcript

    user_block_parts: List[str] = []
    if kb_block:
        user_block_parts.append("KNOWLEDGE BASE CONTEXT:\n" + kb_block)
    if hist_block:
        user_block_parts.append("RECENT CHAT HISTORY:\n" + hist_block)
    user_block_parts.append("USER MESSAGE:\n" + (user_message or "").strip())
    full_input = "\n\n---\n\n".join(user_block_parts)

    resp = client.responses.create(
        model=CHAT_MODEL,
        instructions=system_instructions,
        input=full_input,
        temperature=0.2,
    )

    # Try to read output_text first
    out_text = getattr(resp, "output_text", None) or ""
    out_text = out_text.strip()

    # Parse JSON; fall back gracefully if model returned plain text
    answer = ""
    clarifying: List[str] = []
    confidence = 0.55

    if out_text:
        try:
            data = json.loads(out_text)
            answer = str(data.get("answer", "")).strip()
            clarifying_raw = data.get("clarifying_questions", []) or []
            if isinstance(clarifying_raw, list):
                clarifying = [str(x).strip() for x in clarifying_raw if str(x).strip()]
            conf_raw = data.get("confidence", confidence)
            try:
                confidence = float(conf_raw)
            except Exception:
                confidence = 0.55
        except Exception:
            # Not JSON; treat as answer text
            answer = out_text

    # Extra fallback if output_text missing
    if not answer:
        chunks: List[str] = []
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                t = getattr(c, "text", None)
                if t:
                    chunks.append(t)
        answer = "\n".join(chunks).strip() or "Sorry — I couldn’t generate a response."

    # Clamp confidence
    if confidence < 0.0:
        confidence = 0.0
    if confidence > 1.0:
        confidence = 1.0

    return answer, clarifying, confidence
