from typing import Any

from pydantic import BaseModel, Field


class EvaluationSample(BaseModel):
    question: str
    ground_truth: str


class EvaluationInput(BaseModel):
    samples: list[EvaluationSample]
    top_k: int = 5


class EvaluationResult(BaseModel):
    scores: list[dict[str, Any]] = Field(default_factory=list)
    aggregate: dict[str, float] = Field(default_factory=dict)
    num_samples: int = 0
