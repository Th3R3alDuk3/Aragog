from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Ingestion / Task tracking
# ---------------------------------------------------------------------------

class IndexResponse(BaseModel):
    indexed:   int        = Field(..., description="Number of document chunks written to Qdrant")
    source:    str        = Field(..., description="Original filename")
    minio_url: str | None = Field(None, description="URL of the original file in MinIO")


class TaskCreatedResponse(BaseModel):
    task_id: str = Field(..., description="Unique task ID — poll GET /tasks/{task_id} for progress")
    source:  str = Field(..., description="Uploaded filename")
    message: str = "Ingestion started. Poll /tasks/{task_id} for progress."


class TaskState(BaseModel):
    model_config = ConfigDict(frozen=False)  # fields are mutated by the ingestion service

    task_id:    str      = Field(..., description="Unique task ID")
    status:     Literal["pending", "running", "done", "failed"] = Field(..., description="Task lifecycle status")
    step:       str      = Field(..., description="Current pipeline step")
    source:     str      = Field(..., description="Uploaded filename")
    created_at: datetime = Field(..., description="Task creation time (UTC)")
    updated_at: datetime = Field(..., description="Last status update time (UTC)")
    result:     IndexResponse | None = Field(None, description="Populated when status=done")
    error:      str | None           = Field(None, description="Error message when status=failed")


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question (simple or compound)")
    top_k: int = Field(
        default=5, ge=1, le=50,
        description="Max source documents returned alongside the answer",
    )
    filters: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional Haystack metadata filters.  Example: "
            "{\"field\": \"source\", \"operator\": \"==\", \"value\": \"report.pdf\"}"
        ),
    )
    date_from: date | None = Field(
        default=None,
        description=(
            "Include only documents indexed on or after this date (ISO 8601, e.g. '2024-01-01'). "
            "Translated to a filter on the integer field ``indexed_at_ts``."
        ),
    )
    date_to: date | None = Field(
        default=None,
        description=(
            "Include only documents indexed on or before this date (ISO 8601, e.g. '2024-12-31'). "
            "The end of that day (23:59:59 UTC) is used so the whole day is included."
        ),
    )


class SourceDocument(BaseModel):
    content: str               # original_content (precise child chunk) for citation
    score: float | None = None
    meta: dict[str, Any] = {}


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    query: str
    sub_questions: list[str] = Field(
        default=[],
        description="Sub-questions detected by QueryAnalyzer (empty if query was simple)",
    )
    is_compound: bool = Field(
        default=False,
        description="True when the query was automatically decomposed into sub-questions",
    )
    low_confidence: bool = Field(
        default=False,
        description=(
            "True when CRAG determined the retrieved evidence is below the confidence "
            "threshold even after retries. Treat the answer with extra caution."
        ),
    )
    extracted_filters: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Metadata filters automatically extracted from the natural language query "
            "by QueryAnalyzer. Useful for debugging filter inference."
        ),
    )


# ---------------------------------------------------------------------------
# Evaluation (RAGAS)
# ---------------------------------------------------------------------------

class EvaluationSample(BaseModel):
    question: str = Field(..., description="The test question")
    ground_truth: str = Field(..., description="Expected correct answer for RAGAS scoring")


class EvaluationRequest(BaseModel):
    samples: list[EvaluationSample] = Field(
        ..., min_length=1,
        description="List of question/ground_truth pairs to evaluate",
    )
    top_k: int = Field(
        default=5, ge=1, le=20,
        description="Number of retrieved documents per question",
    )


class EvaluationResponse(BaseModel):
    scores: list[dict[str, Any]] = Field(
        description="Per-question scores (question, faithfulness, answer_relevancy, context_precision)"
    )
    aggregate: dict[str, float] = Field(
        description="Mean scores across all samples"
    )
    num_samples: int
