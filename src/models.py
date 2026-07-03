from pydantic import BaseModel


class EvalSample(BaseModel):
    id: str
    prompt: str
    expected_answer: str


class TargetResponse(BaseModel):
    output: str
    error: str | None = None
    success: bool = False
    latency_ms: float = 0.0
    tokens_used: int = 0


class JudgeScore(BaseModel):
    relevance: int
    hallucination: int
    toxicity: int
    reasoning: str


class RunRecord(BaseModel):
    sample_id: str
    prompt: str
    expected_answer: str
    target_output: str | None = None
    target_error: str | None = None
    target_success: bool = False
    target_latency_ms: float = 0.0
    target_tokens: int = 0
    relevance: int | None = None
    hallucination: int | None = None
    toxicity: int | None = None
    reasoning: str | None = None
    judge_latency_ms: float = 0.0
    judge_tokens: int = 0
    passed: bool = False
