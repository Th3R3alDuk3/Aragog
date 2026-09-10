from pydantic import BaseModel, Field


class SearchHit(BaseModel):
    id: str = Field(
        description="Use with read_chunks.",
    )
    score: float | None = Field(
        description="Rerank score; higher is better.",
    )
    source: str = Field(
        description="Source filename.",
    )
    page: int | None = Field(
        description="Starting page.",
    )
    headings: list[str] = Field(
        description="Section path.",
    )
    snippet: str = Field(
        description="Preview for triage; not evidence.",
    )


class ChunkContent(BaseModel):
    id: str = Field(
        description="Chunk id.",
    )
    source: str = Field(
        description="Source filename.",
    )
    url: str = Field(
        description=(
            "Temporary source link. Cite verbatim; it expires."
        ),
    )
    page: int | None = Field(
        description="Starting page.",
    )
    content: str | None = Field(
        description="Full chunk text with heading path.",
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
