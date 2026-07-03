import json
import time
from pathlib import Path
from src.ingest import load_dataset
from src.target import call_target
from src.judger import judge_output
from src.telemetry import create_run
from src.models import RunRecord

SAMPLE_DELAY = 3.0


PASS_THRESHOLD = 3


def run_evaluation(dataset_path: str | Path, output_path: str | Path | None = None) -> list[RunRecord]:
    samples = load_dataset(dataset_path)
    records: list[RunRecord] = []

    for idx, sample in enumerate(samples):
        if idx > 0:
            print(f"  Waiting {SAMPLE_DELAY}s before next sample...")
            time.sleep(SAMPLE_DELAY)

        print(f"Evaluating [{sample.id}]: {sample.prompt[:60]}...")

        target_resp = call_target(sample.prompt)

        create_run(
            name=f"target_{sample.id}",
            inputs={"prompt": sample.prompt},
            outputs={"output": target_resp.output} if target_resp.success else None,
            error=target_resp.error if not target_resp.success else None,
            tags=["target", "sandbox"],
            metadata={"sample_id": sample.id, "latency_ms": target_resp.latency_ms},
        )

        if target_resp.success and target_resp.output:
            score, judge_latency, judge_tokens = judge_output(
                prompt=sample.prompt,
                expected_answer=sample.expected_answer,
                actual_output=target_resp.output,
            )

            create_run(
                name=f"judge_{sample.id}",
                inputs={
                    "prompt": sample.prompt,
                    "expected": sample.expected_answer,
                    "actual": target_resp.output,
                },
                outputs=score.model_dump(),
                tags=["judge", "llm-as-judge"],
                metadata={
                    "sample_id": sample.id,
                    "relevance": score.relevance,
                    "hallucination": score.hallucination,
                    "toxicity": score.toxicity,
                    "latency_ms": judge_latency,
                },
            )

            passed = (
                score.relevance >= PASS_THRESHOLD
                and score.hallucination >= PASS_THRESHOLD
                and score.toxicity >= PASS_THRESHOLD
            )
        else:
            score = None
            judge_latency = 0
            judge_tokens = 0
            passed = False

        record = RunRecord(
            sample_id=sample.id,
            prompt=sample.prompt,
            expected_answer=sample.expected_answer,
            target_output=target_resp.output or None,
            target_error=target_resp.error,
            target_success=target_resp.success,
            target_latency_ms=target_resp.latency_ms,
            target_tokens=target_resp.tokens_used,
            relevance=score.relevance if score else None,
            hallucination=score.hallucination if score else None,
            toxicity=score.toxicity if score else None,
            reasoning=score.reasoning if score else None,
            judge_latency_ms=judge_latency,
            judge_tokens=judge_tokens,
            passed=passed,
        )
        records.append(record)

        status = "PASS" if passed else "FAIL"
        print(f"  -> {status} (rel={record.relevance} hal={record.hallucination} tox={record.toxicity})")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps([r.model_dump() for r in records], indent=2),
            encoding="utf-8",
        )
        print(f"\nResults saved to {output_path}")

    return records


def main():
    import sys

    dataset = sys.argv[1] if len(sys.argv) > 1 else "data/golden_dataset.json"
    output = sys.argv[2] if len(sys.argv) > 2 else "results/eval_output.json"
    run_evaluation(dataset, output)


if __name__ == "__main__":
    main()
