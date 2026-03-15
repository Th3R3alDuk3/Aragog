from pydantic import BaseModel, Field


class IndexCommand(BaseModel):
    file_name: str
    file_bytes: bytes


class IndexResponse(BaseModel):
    indexed: int = Field(..., description="Number of document chunks written to Qdrant")
    source: str = Field(..., description="Original filename")
