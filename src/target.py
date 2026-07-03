import time
from openai import OpenAI
from src.models import TargetResponse
from src.config import OPENCODE_ZEN_API_KEY, LLM_BASE_URL, LLM_MODEL, MAX_TOKENS_PER_CALL


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENCODE_ZEN_API_KEY, base_url=LLM_BASE_URL)
    return _client


def call_target(prompt: str) -> TargetResponse:
    client = _get_client()
    start = time.perf_counter()

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer concisely and accurately."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_TOKENS_PER_CALL,
            temperature=0.1,
        )
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return TargetResponse(
            output="",
            error=str(e),
            success=False,
            latency_ms=round(elapsed, 1),
        )

    elapsed = (time.perf_counter() - start) * 1000
    output = response.choices[0].message.content or ""
    tokens_used = response.usage.total_tokens if response.usage else 0

    return TargetResponse(
        output=output,
        success=True,
        latency_ms=round(elapsed, 1),
        tokens_used=tokens_used,
    )
