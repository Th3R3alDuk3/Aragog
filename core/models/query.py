from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field

from core.models.retrieval import RetrievedSource


class QueryInput(BaseModel):
    query: str
    top_k: int = 5
    filters: dict[str, Any] | None = None
    date_from: date | None = None
    date_to: date | None = None


class QueryResult(BaseModel):
    answer: str
    sources: list[RetrievedSource]
    query: str
    sub_questions: list[str] = Field(default_factory=list)
    is_compound: bool = False
    low_confidence: bool = False
    extracted_filters: dict[str, Any] | None = None
