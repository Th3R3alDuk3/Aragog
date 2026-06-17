from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    # Server
    host: str
    port: int
    # Auth (OpenWebUI JWT)
    jwt_secret: str
    jwt_algorithm: str
    # S3 Storage
    minio_endpoint: str
    minio_user: str
    minio_password: str
    minio_bucket: str
    minio_url_expire: int
    # Document Store
    qdrant_url: str
    qdrant_token: str
    qdrant_embedding_dim: int
    qdrant_collection: str
    # Converter
    docling_url: str
    docling_timeout: int
    # Document Chunker
    chunker_tokenizer: str
    chunker_max_tokens: int
    # Enricher
    enricher_max_workers: int
    enricher_model: str
    enricher_url: str
    enricher_token: str
    enricher_timeout: int
    # Embedders (Dense + Sparse)
    dense_embedding_model: str
    dense_embedding_url: str
    dense_embedding_token: str
    dense_embedding_timeout: int
    sparse_embedding_model: str
    sparse_embedding_language: str
    sparse_embedding_device: str
    embedded_meta_fields: str
    # Reranker
    reranker_model: str
    reranker_url: str
    reranker_token: str
    reranker_timeout: int


@lru_cache
def get_settings() -> Settings:
    return Settings()
