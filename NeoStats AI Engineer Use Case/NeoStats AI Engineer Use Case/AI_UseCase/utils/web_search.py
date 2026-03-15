from typing import List, Dict

from tavily import TavilyClient

from config.config import get_settings


def should_trigger_web_search(query: str) -> bool:
    try:
        if not query:
            return False

        query_lower = query.lower()
        recency_keywords = [
            "latest",
            "today",
            "current",
            "recent",
            "news",
            "update",
            "2025",
            "2026",
            "this week",
            "this month",
            "stock",
            "price",
            "trend",
        ]
        return any(keyword in query_lower for keyword in recency_keywords)
    except Exception:
        return False


def search_web(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    try:
        settings = get_settings()
        if not settings.tavily_api_key:
            return []

        client = TavilyClient(api_key=settings.tavily_api_key)
        result = client.search(query=query, max_results=max_results, include_answer=True)

        items = []
        for row in result.get("results", []):
            items.append(
                {
                    "title": row.get("title", "Untitled"),
                    "url": row.get("url", ""),
                    "content": row.get("content", ""),
                }
            )

        return items
    except Exception:
        return []


def format_web_results(results: List[Dict[str, str]]) -> str:
    try:
        if not results:
            return ""

        formatted = []
        for index, item in enumerate(results, start=1):
            formatted.append(
                f"[{index}] {item.get('title', 'Untitled')}\n"
                f"URL: {item.get('url', '')}\n"
                f"Snippet: {item.get('content', '')}"
            )
        return "\n\n".join(formatted)
    except Exception:
        return ""
