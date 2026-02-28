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
    qdrant_api_key: str = ""                 # empty = no auth

    # --- Docling ---
    docling_url: str = "http://localhost:5001/ui"

    # --- Dense Embeddings (HuggingFace / sentence-transformers, local) ---
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimension: int = 1024
    embedding_device: str = "cpu"        # "cpu" | "cuda" | "cuda:0" etc.

    # --- Sparse Embeddings (SPLADE via FastEmbed, local) ---
    sparse_embedding_model: str = "Qdrant/bm42-all-minilm-l6-v2-attentions"
    sparse_embedding_device: str = "cpu"    # "cpu" | "cuda" (ONNX provider)

    # --- Reranker (HuggingFace cross-encoder, local) ---
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_top_k: int = 5
    reranker_device: str = "cpu"            # "cpu" | "cuda" | "cuda:0" | "mps"

    # --- LLM (OpenAI-compatible) ---
    openai_api_key: str = ""
    openai_base_url: str = "http://localhost:11434/v1"  # empty = official OpenAI API
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
    child_chunk_size: int = 200             # words — small child for retrieval
    child_chunk_overlap: int = 20           # words
    parent_enabled: bool = True             # store parent section for LLM context
    doc_beginning_chars: int = 1500        # chars sent to context-prefix LLM

    # --- Retrieval ---
    dense_retriever_top_k: int = 20
    sparse_retriever_top_k: int = 20
    final_top_k: int = 5

    # --- HyDE (Hypothetical Document Embedding) ---
    hyde_enabled: bool = False
    hyde_model: str = ""             # empty → use llm_model

    # --- RAPTOR (multi-level summary chunks, added during indexing) ---
    raptor_enabled: bool = False

    # --- CRAG (Corrective RAG — re-retrieval on low relevance score) ---
    crag_enabled: bool = False
    crag_score_threshold: float = 0.3   # reranker score below this triggers re-retrieval
    crag_max_retries: int = 2

    # --- ColBERT (late-interaction second-pass reranker, local) ---
    colbert_enabled: bool = False
    colbert_model: str = "colbert-ir/colbertv2.0"
    colbert_top_k: int = 5
    colbert_device: str = "cpu"             # "cpu" | "cuda" | "cuda:0"

    # --- RAGAS evaluation ---
    ragas_enabled: bool = False

    # --- MinIO (object storage for original documents) ---
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "documents"
    minio_secure: bool = False

    # --- App ---
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_log_level: str = "info"


@lru_cache
def get_settings() -> Settings:
    return Settings()
