# openai_client.py

import os
import json
import re
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


def _normalize_question(q: str) -> str:
    q = (q or "").strip().lower()
    q = re.sub(r"\s+", " ", q)
    q = q.strip(" .!?;:\n\r\t")
    return q


def _extract_questions_from_text(text: str) -> List[str]:
    """
    Best-effort extraction of questions from assistant text.
    We treat lines ending with '?' as questions, plus any '?'-terminated fragments.
    """
    if not text:
        return []
    candidates: List[str] = []

    for line in text.splitlines():
        line = line.strip()
        if line.endswith("?") and len(line) > 1:
            candidates.append(line)

    # Also grab fragments that end in '?'
    parts = re.split(r"(\?)", text)
    buf = ""
    for p in parts:
        buf += p
        if p == "?":
            frag = buf.strip()
            buf = ""
            if frag and len(frag) > 1:
                candidates.append(frag)

    out: List[str] = []
    seen = set()
    for c in candidates:
        n = _normalize_question(c)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _count_question_repeats_in_history(history: Optional[List[Dict[str, Any]]], question_norm: str) -> int:
    """
    Counts how many prior ASSISTANT messages contained this normalized question.
    """
    if not history or not question_norm:
        return 0
    count = 0
    for h in history:
        if not isinstance(h, dict):
            continue
        if (h.get("role") or "").strip().lower() != "assistant":
            continue
        text = (h.get("text") or "").strip()
        if not text:
            continue
        qs = _extract_questions_from_text(text)
        if question_norm in qs:
            count += 1
    return count


def _force_escalation_message() -> str:
    return (
        "I’m starting to repeat the same diagnostic question, which usually means we need a human to jump in.\n\n"
        "Most likely: this needs a hands-on check to confirm the exact source/path (a photo or quick inspection usually settles it).\n"
        "Also possible: the symptoms match more than one common failure point without a visual confirmation.\n\n"
        "Please tap **Request Help** so we can escalate this and get you to support with your session details."
    )


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
- If a safety risk is present, lead with safety steps.
- Prefer the knowledge base context when present.

PRIMARY RULE (reduce repetitive questions):
- Only ask a clarifying question if the user's answer would change the VERY NEXT STEP you would give.
- If the next step would be the same either way, do NOT ask — proceed with the best safe next step.

QUESTION STYLE:
- Do NOT default to yes/no questions.
- Prefer specific diagnostic questions that narrow the problem efficiently.
- When helpful, ask as a multiple-choice question with 2–4 options (e.g., "Which best matches: A / B / C?").
- Use yes/no only when the decision is truly binary and immediately branch-defining.

QUESTION LIMIT:
- Ask at most ONE clarifying question per message.
- clarifying_questions MUST contain 0 or 1 items.

ANTI-REPETITION:
- Before asking any question, check recent chat history + context.
- If the user already answered it (even implicitly), do not ask again. Move forward.

ASSUME AND ADVANCE:
- If key details are missing, make a reasonable best-guess assumption and proceed with the safest next step.
- If you make an assumption, label it briefly.

PROVIDE LIKELY CAUSES:
- Include 1–2 likely causes in most responses, phrased as "Most likely: ..." and "Also possible: ...".
- Keep it short.

KNOWN CONTEXT:
- Airstream year: {airstream_year if airstream_year is not None else "unknown"}
- Category: {category or "unknown"}
- Safety flags: {", ".join(safety_flags) if safety_flags else "none"}

OUTPUT RULES:
- If you ask ONE clarifying question, keep the answer concise, include 1–2 likely causes, and do not provide long fix instructions yet.

Return STRICT JSON:
{{
  "answer": string,
  "clarifying_questions": string[],
  "confidence": number
}}
""".strip()

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

    out_text = (getattr(resp, "output_text", None) or "").strip()

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
            answer = out_text

    if not answer:
        chunks: List[str] = []
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                t = getattr(c, "text", None)
                if t:
                    chunks.append(t)
        answer = "\n".join(chunks).strip() or "Sorry — I couldn’t generate a response."

    # --- AUTO ESCALATION-STYLE MESSAGE (repeat same question 2+ times) ---
    # We check BOTH:
    #   A) the explicit clarifying question (if provided)
    #   B) any question(s) embedded in the answer text (in case the model put it there)
    candidate_questions: List[str] = []

    if clarifying:
        candidate_questions.append(_normalize_question(clarifying[0]))

    # Extract from answer too
    candidate_questions.extend(_extract_questions_from_text(answer))

    # De-dupe candidates while preserving order
    seen = set()
    deduped: List[str] = []
    for q in candidate_questions:
        qn = _normalize_question(q)
        if qn and qn not in seen:
            seen.add(qn)
            deduped.append(qn)

    should_force = False
    for qn in deduped:
        repeats = _count_question_repeats_in_history(history, qn)
        if repeats >= 2:
            should_force = True
            break

    if should_force:
        # Force an escalation-style message (no clarifying questions)
        clarifying = []
        confidence = min(confidence, 0.25)
        answer = _force_escalation_message()

    # Clamp confidence
    if confidence < 0.0:
        confidence = 0.0
    if confidence > 1.0:
        confidence = 1.0

    return answer, clarifying, confidence
