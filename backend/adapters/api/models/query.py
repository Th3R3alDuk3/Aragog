from datetime import date
from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question (simple or compound)")
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Max source documents returned alongside the answer",
    )
    filters: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional Haystack metadata filters. Example: "
            "{\"field\": \"source\", \"operator\": \"==\", \"value\": \"report.pdf\"}"
        ),
    )
    date_from: date | None = Field(
        default=None,
        description=(
            "Include only documents whose semantic document date or covered period starts on or after "
            "this date (ISO 8601, e.g. '2024-01-01')."
        ),
    )
    date_to: date | None = Field(
        default=None,
        description=(
            "Include only documents whose semantic document date or covered period ends on or before "
            "this date (ISO 8601, e.g. '2024-12-31'). The end of that day (23:59:59 UTC) is used "
            "so the whole day is included."
        ),
    )


class SourceDocument(BaseModel):
    content: str  # original_content (precise child chunk) for citation
    score: float | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    query: str
    sub_questions: list[str] = Field(
        default_factory=list,
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
