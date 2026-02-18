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

# ✅ Default "always-on" authoritative facts for your domain.
# Keep these tight, accurate, and scoped to your supported years.
DEFAULT_AUTHORITATIVE_FACTS: List[str] = [
    # Modern Airstream scope (your app): torsion axles, not leaf springs.
    "Airstream travel trailers in the 2010–2026 model-year range use torsion axles (Dexter-style) in factory configuration, not leaf springs. "
    "If the user mentions leaf springs, treat it as either incorrect terminology or an unusual modification and advise a quick visual verification.",

    # ✅ Wheel studs default for your supported years (avoid absolute claim)
    "For Airstream travel trailers in the 2010–2026 range, wheel studs are typically 1/2\"-20 in factory configuration. "
    "If the user reports a different stud size (e.g., 9/16\") treat it as a possible axle/hub swap or unusual configuration and advise verifying the placard/manual or measuring the stud.",

    # ✅ Airstream-specific lug torque (prevents drift to generic RV numbers like 120)
    "For factory Airstream travel trailers (2010–2026) with OEM wheels and 1/2\"-20 studs, the manufacturer lug nut torque specification is 110 ft-lb unless the trailer's tire/loading placard or owner documentation states otherwise. "
    "If the trailer has aftermarket wheels, axle/hub swaps, or non-standard studs, instruct the user to verify the placard/manual and wheel manufacturer guidance before torquing."
]


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

SCOPE INTERPRETATION:
- Even if the user’s message is short or generic, interpret it in the context of the user owning an Airstream travel trailer.

SOURCE PRIORITY (HARD HIERARCHY):
1) AUTHORITATIVE FACTS (Airstream placard/manual/service docs, or app-provided facts) are highest authority.
2) KNOWLEDGE BASE CONTEXT (your curated KB) is next.
3) RECENT CHAT HISTORY is context only (not authoritative).
4) General knowledge is last resort.
- If AUTHORITATIVE FACTS or KB conflict with generic RV norms, ALWAYS follow AUTHORITATIVE FACTS / KB.
- Never “split the difference” between conflicting specs.

ACCURACY FIRST:
- Do NOT guess model-year-specific details you don’t have.
- Do NOT invent part numbers, wiring, specs, procedures, or policies.
- If the user asks for something that depends on exact configuration, ask ONE clarifying question or provide safe verification steps.
- When multiple answers are plausible, say "Most likely" vs "Also possible" and give a quick check to distinguish.
- Prefer being correct and cautious over being fast and confident.

AUTHORITATIVE FACTS RULE:
- If AUTHORITATIVE FACTS are provided, they override general knowledge.
- Do NOT contradict them.
- If uncertain, defer to them.

CONTEXT ANCHOR:
- If the input includes a PENDING_QUESTION, treat the user's message as answering it.
- Stay on that diagnostic thread; do not switch topics unless the user explicitly asks to reset/switch.

CRITICAL FORMAT RULE (UI requirement):
- NEVER put the clarifying question in the "answer" text.
- If a question is required, put it ONLY in "clarifying_questions".
- The "answer" must contain explanation / likely causes / next steps — but NO question.

Core behavior:
- Speak like Vinnie: an experienced Airstream tech (practical, confident, no fluff).
- Never say “as an AI”, “I can’t browse”, or similar meta.
- Be technical-but-clear and calm.
- If a safety risk is present, lead with safety steps.
- Prefer the knowledge base context when present.

QUESTION LIMIT:
- Ask at most ONE clarifying question per message.
- clarifying_questions MUST contain 0 or 1 items.

KNOWN CONTEXT:
- Airstream year: {airstream_year if airstream_year is not None else "unknown"}
- Category: {category or "unknown"}
- Safety flags: {", ".join(safety_flags) if safety_flags else "none"}

Return STRICT JSON:
{{
  "answer": string,
  "clarifying_questions": string[],
  "confidence": number
}}
""".strip()

    kb_block = (context or "").strip()
    hist_block = transcript

    # ✅ Merge default authoritative facts + any dynamic ones passed in
    merged_facts: List[str] = []
    merged_facts.extend(DEFAULT_AUTHORITATIVE_FACTS)
    if authoritative_facts:
        merged_facts.extend([f for f in authoritative_facts if f and f.strip()])

    facts_block = ""
    if merged_facts:
        cleaned_facts = [f.strip() for f in merged_facts if f and f.strip()]
        if cleaned_facts:
            facts_block = "AUTHORITATIVE FACTS:\n"
            for f in cleaned_facts[:12]:
                facts_block += f"- {f}\n"

    user_block_parts: List[str] = []

    # ✅ Inject facts FIRST (highest authority)
    if facts_block:
        user_block_parts.append(facts_block)

    if kb_block:
        user_block_parts.append("KNOWLEDGE BASE CONTEXT:\n" + kb_block)

    # ✅ Always include history if present
    if hist_block:
        user_block_parts.append("RECENT CHAT HISTORY:\n" + hist_block)

    pending_q = (pending_question or "").strip()
    if pending_q:
        if pending_q.startswith("**") and pending_q.endswith("**") and len(pending_q) > 4:
            pending_q = pending_q[2:-2].strip()
        user_block_parts.append(
            "PENDING CLARIFYING QUESTION (the user is answering this now):\n" + pending_q
        )

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

    answer = _strip_question_lines_from_answer(answer)

    if clarifying:
        q = clarifying[0].strip()
        if not (q.startswith("**") and q.endswith("**")):
            clarifying = [f"**{q}**"]

    candidate_questions: List[str] = []

    if clarifying:
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
    """Return a compact progress summary for UI/admin."""
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
