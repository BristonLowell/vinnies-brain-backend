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


def _strip_question_lines_from_answer(answer: str) -> str:
    """
    Remove question lines from the main answer so the question only appears at the bottom UI.
    Also removes bolded question lines like **... ?**
    """
    if not answer:
        return answer

    cleaned_lines: List[str] = []
    for line in answer.splitlines():
        raw = line.strip()
        if not raw:
            cleaned_lines.append(line)
            continue

        # Strip markdown bold wrapper for evaluation
        unbold = raw
        if unbold.startswith("**") and unbold.endswith("**") and len(unbold) > 4:
            unbold = unbold[2:-2].strip()

        # If the line is a question, drop it
        if unbold.endswith("?"):
            continue

        cleaned_lines.append(line)

    # Also remove leading empty lines created by stripping
    out = "\n".join(cleaned_lines).strip()
    return out


def generate_answer(
    *,
    user_message: str,
    context: str,
    safety_flags: Optional[List[str]] = None,
    airstream_year: Optional[int] = None,
    category: Optional[str] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    pending_question: Optional[str] = None,
    authoritative_facts: Optional[List[str]] = None,
) -> Tuple[str, List[str], float]:
    """
    Matches what main.py expects:
      returns (answer, clarifying_questions, confidence)
    """
    safety_flags = safety_flags or []
    transcript = _history_to_text(history)

    system_instructions = f"""
You are Vinnie's Brain, an AI assistant that ONLY provides information related to Airstream travel trailers (especially 2010–2026) and their systems, maintenance, troubleshooting, repairs, parts, ownership, and model-specific guidance.

Hard rule:
- Prefer the knowledge base context when present.
- If the user's question is NOT directly related to Airstream trailers (or the question cannot reasonably be interpreted as Airstream-related), you MUST refuse.

When refusing:
- Be polite and brief
- State that you only answer Airstream-related questions
- Invite the user to ask an Airstream-specific question
- Do NOT provide general advice or an off-topic answer
- Still return valid JSON in the required schema

AUTHORITATIVE FACTS RULE:
- If AUTHORITATIVE FACTS are provided, they override general knowledge.
- Do NOT contradict them.
- If uncertain, defer to them.

CONTEXT ANCHOR (reduce confusion on short replies):
- If the input includes a PENDING_QUESTION, treat the user's message as answering it.
- Stay on that diagnostic thread; do not switch topics unless the user explicitly asks to reset/switch.

CRITICAL FORMAT RULE (UI requirement):
- NEVER put the clarifying question in the "answer" text (especially not bold at the top).
- If a question is required, put it ONLY in "clarifying_questions".
- The "answer" must contain explanation / likely causes / next steps — but NO question.

Core behavior:
- Speak like Vinnie: an experienced Airstream tech (practical, confident, no fluff).
- Never say “as an AI”, “I can’t browse”, or similar meta.
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
- The question itself must appear ONLY in clarifying_questions.

Return STRICT JSON:
{{
  "answer": string,
  "clarifying_questions": string[],
  "confidence": number
}}
""".strip()

    kb_block = (context or "").strip()
    hist_block = transcript

    facts_block = ""
    if authoritative_facts:
        cleaned_facts = [f.strip() for f in authoritative_facts if f and f.strip()]
        if cleaned_facts:
            facts_block = "AUTHORITATIVE FACTS:\n"
            for f in cleaned_facts[:8]:  # limit to avoid prompt bloat
                facts_block += f"- {f}\n"


    user_block_parts: List[str] = []

    # 🔥 Inject facts FIRST (highest authority)
    if facts_block:
        user_block_parts.append(facts_block)

    if kb_block:
        user_block_parts.append("KNOWLEDGE BASE CONTEXT:\n" + kb_block)

        if hist_block:
            user_block_parts.append("RECENT CHAT HISTORY:\n" + hist_block)

    # 🔒 STEP C — anchor the user's reply to the last clarifying question
    pending_q = (pending_question or "").strip()
    if pending_q:
        # defensively remove markdown wrappers if they slipped in
        if pending_q.startswith("**") and pending_q.endswith("**") and len(pending_q) > 4:
            pending_q = pending_q[2:-2].strip()

        user_block_parts.append(
            "PENDING CLARIFYING QUESTION (the user is answering this now):\n"
            + pending_q
        )

    # USER MESSAGE ALWAYS COMES LAST
    user_block_parts.append(
        "USER MESSAGE:\n" + (user_message or "").strip()
    )

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

    # --- UI FIX: remove any question lines from the top/body of the answer ---
    # This ensures the question only appears in clarifying_questions (bottom of your UI).
    answer = _strip_question_lines_from_answer(answer)

    # --- UI FIX: bold the clarifying question (so the bottom question is bold) ---
    if clarifying:
        q = clarifying[0].strip()
        # Avoid double-bolding
        if not (q.startswith("**") and q.endswith("**")):
            clarifying = [f"**{q}**"]

    # --- AUTO ESCALATION-STYLE MESSAGE (repeat same question 2+ times) ---
    candidate_questions: List[str] = []

    if clarifying:
        # remove ** for repeat-detection normalization
        q0 = clarifying[0].strip()
        if q0.startswith("**") and q0.endswith("**") and len(q0) > 4:
            q0 = q0[2:-2].strip()
        candidate_questions.append(_normalize_question(q0))

    candidate_questions.extend(_extract_questions_from_text(answer))

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
        clarifying = []
        confidence = min(confidence, 0.25)
        answer = _force_escalation_message()

    # Clamp confidence
    if confidence < 0.0:
        confidence = 0.0
    if confidence > 1.0:
        confidence = 1.0

    return answer, clarifying, confidence


def generate_checkpoint_summary(
    *,
    history: Optional[List[Dict[str, Any]]] = None,
    airstream_year: Optional[int] = None,
    category: Optional[str] = None,
) -> Dict[str, List[str]]:
    """Return a compact progress summary for UI/admin.

    Output:
      {
        "known": [...],
        "ruled_out": [...],
        "likely_causes": [...],
        "next_checks": [...]
      }
    """
    transcript = _history_to_text(history, max_items=24)

    instructions = f"""
You are summarizing an Airstream troubleshooting chat into a short checkpoint.

Rules:
- Keep each bullet short (max ~12 words).
- Do NOT invent facts. Only include what is stated or strongly implied.
- Provide at most:
  - known: up to 5 bullets
  - ruled_out: up to 4 bullets
  - likely_causes: 2–3 bullets
  - next_checks: 2–4 bullets
- If a section has nothing solid, return an empty list for it.

Context:
- Airstream year: {airstream_year if airstream_year is not None else "unknown"}
- Category: {category or "unknown"}

Return STRICT JSON:
{{
  "known": string[],
  "ruled_out": string[],
  "likely_causes": string[],
  "next_checks": string[]
}}
""".strip()

    input_text = ("CHAT HISTORY:\n" + transcript) if transcript else "CHAT HISTORY:\n(no history)"

    resp = client.responses.create(
        model=CHAT_MODEL,
        instructions=instructions,
        input=input_text,
        temperature=0.2,
    )

    out_text = (getattr(resp, "output_text", None) or "").strip()
    if not out_text:
        return {"known": [], "ruled_out": [], "likely_causes": [], "next_checks": []}

    try:
        data = json.loads(out_text)
        def _as_list(x):
            return [str(i).strip() for i in (x or []) if str(i).strip()]

        return {
            "known": _as_list(data.get("known")),
            "ruled_out": _as_list(data.get("ruled_out")),
            "likely_causes": _as_list(data.get("likely_causes")),
            "next_checks": _as_list(data.get("next_checks")),
        }
    except Exception:
        return {"known": [], "ruled_out": [], "likely_causes": [], "next_checks": []}
