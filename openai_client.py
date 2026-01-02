# openai_client.py (paste this whole file as-is)

import os
from typing import List
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5.2")

def embed_text(text: str) -> List[float]:
    if not text.strip():
        text = " "
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding

def generate_answer(system_prompt: str, kb_context: str, user_message: str) -> str:
    resp = client.responses.create(
        model=CHAT_MODEL,
        instructions=system_prompt + "\n\n" + kb_context,
        input=user_message,
        temperature=0.2,
    )
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text.strip()

    chunks = []
    for item in getattr(resp, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            t = getattr(c, "text", None)
            if t:
                chunks.append(t)
    return "\n".join(chunks).strip() or "Sorry — I couldn’t generate a response."
