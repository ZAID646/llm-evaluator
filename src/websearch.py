from tavily import TavilyClient
from src.config import TAVILY_API_KEY


_client: TavilyClient | None = None


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        if not TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY is not set")
        _client = TavilyClient(api_key=TAVILY_API_KEY)
    return _client


def search_context(query: str, max_results: int = 3) -> str:
    try:
        client = _get_client()
        response = client.search(query=query, max_results=max_results)
        results = response.get("results", [])
        if not results:
            return ""

        lines = []
        for r in results:
            title = r.get("title", "")
            content = r.get("content", "")
            if title and content:
                lines.append(f"{title}: {content}")

        return "\n\n".join(lines)
    except Exception:
        return ""
