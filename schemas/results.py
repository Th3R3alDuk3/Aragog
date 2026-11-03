from typing import Any

from pydantic import BaseModel, Field

from schemas.enrichment import FilterMeta


class SearchHit(BaseModel):
    id: str = Field(
        description="Pass to read_chunks or read_neighbors.",
    )
    score: float = Field(
        description="Rerank score; higher is better.",
    )
    title: str = Field(
        description="Document title.",
    )
    snippet: str = Field(
        description="Preview for triage; not evidence.",
    )


class ChunkContent(FilterMeta):
    id: str = Field(
        description="Chunk id.",
    )
    title: str = Field(
        description="Document title.",
    )
    url: str = Field(
        description="Document link. Cite verbatim.",
    )
    document_meta: dict[str, Any] = Field(
        description="Metadata of the whole document; filtered_search values.",
    )
    content: str | None = Field(
        description="Full chunk text.",
    )


class SearchResult(BaseModel):
    hint: str = Field(
        description="Next-step guidance when empty; otherwise empty.",
    )
    hits: list[SearchHit] = Field(
        description="Reranked hits.",
    )


class ReadResult(BaseModel):
    hint: str = Field(
        description="Next-step guidance when empty; otherwise empty.",
    )
    chunks: list[ChunkContent] = Field(
        description="Full chunk contents.",
    )


class DocumentFields(BaseModel):
    hint: str = Field(
        description="Next-step guidance when empty; otherwise empty.",
    )
    fields: dict[str, dict[str, int]] = Field(
        description="Field → its most common exact values → number of chunks.",
    )
