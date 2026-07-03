import time
import httpx
from src.models import TargetResponse
from src.config import TARGET_URL


def call_target(prompt: str) -> TargetResponse:
    start = time.perf_counter()
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{TARGET_URL}/run", json={"prompt": prompt})
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as e:
        elapsed = (time.perf_counter() - start) * 1000
        return TargetResponse(
            output="",
            error=f"Target API error: {e}",
            success=False,
            latency_ms=round(elapsed, 1),
        )
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return TargetResponse(
            output="",
            error=f"Unexpected error: {e}",
            success=False,
            latency_ms=round(elapsed, 1),
        )

    elapsed = (time.perf_counter() - start) * 1000

    output = data.get("output") or data.get("error") or ""
    success = data.get("success", False)
    tokens_used = _estimate_tokens(output)

    return TargetResponse(
        output=output,
        success=success,
        latency_ms=round(elapsed, 1),
        tokens_used=tokens_used,
    )


def _estimate_tokens(text: str) -> int:
    return len(text) // 4
