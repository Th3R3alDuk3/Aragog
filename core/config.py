from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from core.models.vocabulary import DEFAULT_CHUNK_CLASSIFICATION_TAXONOMY

BACKEND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_children_collection: str = "children"
    qdrant_parents_collection: str = "parents"

    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "documents"
    minio_secure: bool = False

    docling_url: str = "http://localhost:5001"

    openai_url: str = "http://localhost:11434/v1"
    openai_api_key: str = ""
    llm_model: str = "qwen3:14b"
    llm_timeout: int = 120
    instruct_llm_model: str = ""

    @property
    def effective_instruct_model(self) -> str:
        return self.instruct_llm_model or self.llm_model

    embedding_model: str = "BAAI/bge-m3"
    embedding_dimension: int = 1024
    embedding_device: str = "cpu"

    sparse_embedding_model: str = "Qdrant/bm42-all-minilm-l6-v2-attentions"
    sparse_embedding_device: str = "cpu"

    analyzer_max_concurrency: int = 3
    analyzer_max_chars: int = 4000
    classification_taxonomy: str = DEFAULT_CHUNK_CLASSIFICATION_TAXONOMY

    parent_chunk_size: int = 600
    child_chunk_size: int = 200
    child_chunk_overlap: int = 20
    doc_beginning_chars: int = 1500
    # Anthropic Contextual Retrieval: full document passed as context per chunk.
    # 0 = no truncation (full document). Set e.g. 32000 to cap very large docs.
    contextual_doc_max_chars: int = 0
    # Use Anthropic SDK with cache_control on document content (requires
    # ANTHROPIC_API_KEY and the anthropic package; llm_model must be a claude-* model).
    anthropic_caching_enabled: bool = False
    anthropic_api_key: str = ""
    raptor_enabled: bool = False

    dense_retriever_top_k: int = 30
    sparse_retriever_top_k: int = 30
    hyde_enabled: bool = True
    auto_merge_threshold: float = 0.5
    colbert_enabled: bool = True
    colbert_model: str = "colbert-ir/colbertv2.0"
    colbert_top_k: int = 20
    colbert_device: str = "cpu"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_top_k: int = 5
    reranker_device: str = "cpu"
    crag_enabled: bool = True
    crag_score_threshold: float = 0.35
    crag_max_retries: int = 2

    final_top_k: int = 5
    ragas_enabled: bool = True

    task_store_size: int = 200
    indexing_max_concurrent: int = 3
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    cors_origins: str = "*"

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
