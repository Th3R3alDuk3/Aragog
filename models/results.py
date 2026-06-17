from pydantic import BaseModel, Field


class SearchHit(BaseModel):
    id: str = Field(
        description=(
            "The chunk's unique id; pass it to `read_chunk` to read the full chunk."
        ),
    )
    score: float | None = Field(
        description="Cross-encoder rerank relevance score; higher means more relevant.",
    )
    source: str | None = Field(
        description="Filename of the source document this chunk comes from.",
    )
    url: str | None = Field(
        description=(
            "Temporary presigned URL to download the source document; "
            "expires after a short time."
        ),
    )
    page: int | None = Field(
        description="Page number in the source document where the chunk starts.",
    )
    headings: list[str] = Field(
        description=(
            "Heading path (section breadcrumb) of the chunk within the document."
        ),
    )
    snippet: str = Field(
        description=(
            "A short contextual summary of the chunk, for quick triage. Not the "
            "full text; open the chunk with `read_chunk` before answering."
        ),
    )


class ChunkContent(BaseModel):
    id: str = Field(
        description="The chunk's unique id.",
    )
    source: str | None = Field(
        description="Filename of the source document this chunk comes from.",
    )
    url: str | None = Field(
        description=(
            "Temporary presigned URL to download the source document; "
            "expires after a short time."
        ),
    )
    page: int | None = Field(
        description="Page number in the source document where the chunk starts.",
    )
    content: str | None = Field(
        description="Full text content of the chunk, with its heading path prepended.",
    )


class SearchResult(BaseModel):
    hint: str = Field(
        description=(
            "Suggested next step (read_chunk, find_related, read_neighbors or "
            "filtered_search) — guidance for the agent, not part of the data."
        ),
    )
    hits: list[SearchHit] = Field(
        description="The reranked search hits.",
    )


class ReadResult(BaseModel):
    hint: str = Field(
        description=(
            "Suggested next step — guidance for the agent, not part of the data."
        ),
    )
    chunks: list[ChunkContent] = Field(
        description="The full chunk contents.",
    )
