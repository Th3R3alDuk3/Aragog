from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Qdrant ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"
    qdrant_api_key: str = ""

    # --- Docling ---
    docling_url: str = "http://localhost:5001/ui"

    # --- Dense Embeddings (HuggingFace / sentence-transformers, local) ---
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimension: int = 1024
    embedding_device: str = "cpu"

    # --- Sparse Embeddings (SPLADE via FastEmbed, local) ---
    sparse_embedding_model: str = "Qdrant/bm42-all-minilm-l6-v2-attentions"
    sparse_embedding_device: str = "cpu"

    # --- Reranker (HuggingFace cross-encoder, local) ---
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_top_k: int = 5
    reranker_device: str = "cpu"

    # --- LLM (OpenAI-compatible) ---
    openai_api_key: str = ""
    openai_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen3:14b"

    # --- Content Analyzer LLM (used during indexing) ---
    analyzer_llm_model: str = "qwen3:14b"
    analyzer_max_workers: int = 8
    analyzer_max_chars: int = 4000

    # --- Classification Taxonomy ---
    classification_taxonomy: str = (
        "financial,legal,technical,scientific,hr,"
        "marketing,contract,report,manual,correspondence,general"
    )

    # --- Contextual Chunking (Anthropic) ---
    child_chunk_size: int = 200
    child_chunk_overlap: int = 20
    parent_enabled: bool = True
    doc_beginning_chars: int = 1500

    # --- Retrieval ---
    dense_retriever_top_k: int = 20
    sparse_retriever_top_k: int = 20
    final_top_k: int = 5

    # --- HyDE (Hypothetical Document Embedding) ---
    hyde_enabled: bool = True

    # --- RAPTOR (multi-level summary chunks, added during indexing) ---
    raptor_enabled: bool = True

    # --- CRAG (Corrective RAG — re-retrieval on low relevance score) ---
    crag_enabled: bool = True
    crag_score_threshold: float = 0.3
    crag_max_retries: int = 2

    # --- ColBERT (late-interaction second-pass reranker, local) ---
    colbert_enabled: bool = True
    colbert_model: str = "colbert-ir/colbertv2.0"
    colbert_top_k: int = 5
    colbert_device: str = "cpu"

    # --- RAGAS evaluation ---
    ragas_enabled: bool = True

    # --- MinIO (object storage for original documents) ---
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "documents"
    minio_secure: bool = False

    # --- Task store ---
    task_store_size: int = 200
    indexing_max_concurrent: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()
