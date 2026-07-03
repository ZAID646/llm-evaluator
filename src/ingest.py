import json
from pathlib import Path
from src.models import EvalSample


def load_dataset(path: str | Path) -> list[EvalSample]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict) and "samples" in raw:
        items = raw["samples"]
    else:
        raise ValueError("Dataset must be a list or dict with 'samples' key")

    samples = []
    for i, item in enumerate(items):
        sample = EvalSample(
            id=str(item.get("id", i + 1)),
            prompt=item["prompt"],
            expected_answer=item["expected_answer"],
        )
        samples.append(sample)

    return samples
