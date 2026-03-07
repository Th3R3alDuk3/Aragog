from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Infrastructure
    # -------------------------------------------------------------------------

    # --- Qdrant ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_children_collection: str = "children"
    qdrant_parents_collection: str = "parents"

    # --- MinIO ---
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "documents"
    minio_secure: bool = False

    # --- Docling ---
    docling_url: str = "http://localhost:5001/ui"

    # -------------------------------------------------------------------------
    # LLM  (OpenAI-compatible endpoint — shared base URL + API key)
    # -------------------------------------------------------------------------

    openai_url: str = "http://localhost:11434/v1"
    openai_api_key: str = ""

    # Final answer generation, HyDE, RAGAS evaluation — quality matters
    llm_model: str = "qwen3:14b"
    llm_timeout: int = 120  # seconds; increase for slow/thinking models

    # Structured extraction tasks (ContentAnalyzer, RAPTOR, QueryAnalyzer)
    # — small, fast, no-think; falls back to llm_model if not set
    instruct_llm_model: str = ""

    @property
    def effective_instruct_model(self) -> str:
        return self.instruct_llm_model or self.llm_model

    # -------------------------------------------------------------------------
    # Indexing
    # -------------------------------------------------------------------------

    # --- Dense Embeddings (HuggingFace / sentence-transformers, local) ---
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimension: int = 1024
    embedding_device: str = "cpu"

    # --- Sparse Embeddings (SPLADE via FastEmbed, local) ---
    sparse_embedding_model: str = "Qdrant/bm42-all-minilm-l6-v2-attentions"
    sparse_embedding_device: str = "cpu"

    # --- Content Analyzer (one LLM call per chunk during indexing) ---
    analyzer_max_concurrency: int = 8
    analyzer_max_chars: int = 4000

    # --- Classification Taxonomy ---
    classification_taxonomy: str = (
        "financial,legal,technical,scientific,hr,"
        "marketing,contract,report,manual,correspondence,general"
    )

    # --- Contextual Chunking ---
    parent_chunk_size: int = 600
    child_chunk_size: int = 200
    child_chunk_overlap: int = 20
    doc_beginning_chars: int = 1500

    # --- RAPTOR (multi-level summary chunks) ---
    raptor_enabled: bool = True

    # -------------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------------

    # --- Candidate pool ---
    dense_retriever_top_k: int = 30
    sparse_retriever_top_k: int = 30

    # --- HyDE (pre-retrieval: hypothetical document embedding) ---
    hyde_enabled: bool = True

    # --- AutoMergingRetriever (parent-context swap, threshold-based) ---
    auto_merge_threshold: float = 0.5

    # --- ColBERT (late-interaction first-pass before cross-encoder) ---
    colbert_enabled: bool = True
    colbert_model: str = "colbert-ir/colbertv2.0"
    colbert_top_k: int = 20
    colbert_device: str = "cpu"

    # --- Reranker (cross-encoder, local) ---
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_top_k: int = 5
    reranker_device: str = "cpu"

    # --- CRAG (corrective re-retrieval on low confidence) ---
    crag_enabled: bool = True
    crag_score_threshold: float = 0.35
    crag_max_retries: int = 2

    # -------------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------------

    final_top_k: int = 5

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    ragas_enabled: bool = True

    # -------------------------------------------------------------------------
    # Application
    # -------------------------------------------------------------------------

    task_store_size: int = 200
    indexing_max_concurrent: int = 3
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    # Comma-separated list of allowed CORS origins.
    # Use "*" for local development; set explicit origins in production,
    # e.g. "https://app.example.com,https://admin.example.com".
    cors_origins: str = "*"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
