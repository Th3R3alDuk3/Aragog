from typing import Any

from pydantic import BaseModel, Field


class EvaluationSample(BaseModel):
    question: str = Field(..., description="The test question")
    ground_truth: str = Field(..., description="Expected correct answer for RAGAS scoring")


class EvaluationRequest(BaseModel):
    samples: list[EvaluationSample] = Field(
        ...,
        min_length=1,
        description="List of question/ground_truth pairs to evaluate",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of retrieved documents per question",
    )


class EvaluationResponse(BaseModel):
    scores: list[dict[str, Any]] = Field(
        description="Per-question scores (question, faithfulness, answer_relevancy, context_precision)"
    )
    aggregate: dict[str, float] = Field(description="Mean scores across all samples")
    num_samples: int
