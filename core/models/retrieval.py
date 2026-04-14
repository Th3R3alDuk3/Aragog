from datetime import date
from typing import Any

from pydantic import BaseModel, Field


class RetrievalInput(BaseModel):
    query: str
    top_k: int = 5
    filters: dict[str, Any] | None = None
    date_from: date | None = None
    date_to: date | None = None


class RetrievedSource(BaseModel):
    content: str
    score: float | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    query: str
    sources: list[RetrievedSource]
    sub_questions: list[str] = Field(default_factory=list)
    is_compound: bool = False
    low_confidence: bool = False
    extracted_filters: dict[str, Any] | None = None
