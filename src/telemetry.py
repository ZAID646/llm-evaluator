import os
from langsmith import Client as LangSmithClient


_client: LangSmithClient | None = None


def _get_client() -> LangSmithClient | None:
    global _client
    if _client is None and os.getenv("LANGSMITH_API_KEY"):
        _client = LangSmithClient()
    return _client


def create_run(
    name: str,
    inputs: dict,
    outputs: dict | None = None,
    error: str | None = None,
    tags: list[str] | None = None,
    metadata: dict | None = None,
):
    client = _get_client()
    if client is None:
        return None

    run_type = "chain"

    client.create_run(
        name=name,
        inputs=inputs,
        outputs=outputs or {},
        error=error,
        run_type=run_type,
        tags=tags or [],
        extra={"metadata": metadata or {}},
    )
