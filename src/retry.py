import time
import functools


def with_retry(max_retries: int = 5, base_delay: float = 2.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    error_str = str(e)
                    if "429" in error_str or "Rate limit" in error_str or "FreeUsageLimitError" in error_str:
                        delay = base_delay * (2 ** attempt)
                        print(f"  Rate limited (attempt {attempt + 1}/{max_retries + 1}), retrying in {delay:.0f}s...")
                        time.sleep(delay)
                        continue
                    raise
            raise last_error
        return wrapper
    return decorator
