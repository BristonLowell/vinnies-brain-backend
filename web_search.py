# web_search.py
import os
import requests
from typing import List, Dict, Any

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

class WebSearchError(RuntimeError):
    pass

def web_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Returns a list of {title, url, content} results.
    Requires env var: TAVILY_API_KEY
    """
    if not TAVILY_API_KEY:
        raise WebSearchError("TAVILY_API_KEY not set")

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
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", []) or []

    cleaned = []
    for r in results[:max_results]:
        cleaned.append(
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": (r.get("content") or "").strip(),
            }
        )
    return cleaned
