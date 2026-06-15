from pydantic import BaseModel, Field


class SearchHit(BaseModel):
    id: str = Field(
        description="The chunk's unique id; pass it to `read_chunk` to read the full chunk.",
    )
    score: float | None = Field(
        description="Cross-encoder rerank relevance score; higher means more relevant.",
    )
    source: str | None = Field(
        description="Filename of the source document this chunk comes from.",
    )
    url: str | None = Field(
        description="Temporary presigned URL to download the source document; expires after a short time.",
    )
    page: int | None = Field(
        description="Page number in the source document where the chunk starts.",
    )
    headings: list[str] = Field(
        description="Heading path (section breadcrumb) of the chunk within the document.",
    )
    snippet: str = Field(
        description="Concise summary of the chunk — its context within the document and its key point — for quick triage; read the full chunk with `read_chunk`.",
    )


class ChunkContent(BaseModel):
    id: str = Field(
        description="The chunk's unique id.",
    )
    source: str | None = Field(
        description="Filename of the source document this chunk comes from.",
    )
    url: str | None = Field(
        description="Temporary presigned URL to download the source document; expires after a short time.",
    )
    page: int | None = Field(
        description="Page number in the source document where the chunk starts.",
    )
    content: str | None = Field(
        description="Full text content of the chunk, with its heading path prepended.",
    )
