# web_search.py
import os
import requests
from typing import List, Dict, Any

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()

class WebSearchError(RuntimeError):
    pass

def web_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Returns a list of {title, url, content} results.
    Requires env var: TAVILY_API_KEY

    Notes:
    - Uses a shorter timeout to avoid tying up your API.
    - Raises WebSearchError with status/body to make debugging easy.
    """
    if not TAVILY_API_KEY:
        raise WebSearchError("TAVILY_API_KEY not set (Render env var missing)")

    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": False,
                "include_raw_content": False,
                "include_images": False,
            },
            timeout=12,
        )
    except requests.Timeout:
        raise WebSearchError("Tavily request timed out")
    except requests.RequestException as e:
        raise WebSearchError(f"Tavily network error: {e}")

    if resp.status_code >= 400:
        # include a trimmed body so you can see rate limit / auth errors
        body = (resp.text or "").strip()
        if len(body) > 600:
            body = body[:600] + "…"
        raise WebSearchError(f"Tavily HTTP {resp.status_code}: {body}")

    data = resp.json()
    results = data.get("results", []) or []

    cleaned: List[Dict[str, Any]] = []
    for r in results[:max_results]:
        cleaned.append(
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": (r.get("content") or "").strip(),
            }
        )
    return cleaned
