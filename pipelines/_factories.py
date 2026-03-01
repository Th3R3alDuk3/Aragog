"""
Embedder + reranker + generator factories.

Embedding strategy
──────────────────
Dense  : SentenceTransformers (HuggingFace, local inference).
         Default: BAAI/bge-m3  (multilingual, 1024 dim, state of the art).

Sparse : SPLADE / BM42 via FastEmbed (local ONNX inference, no API needed).
         Default: Qdrant/bm42-all-minilm-l6-v2-attentions (multilingual).

Both embedders run entirely on the local machine.

Reranker
────────
HuggingFace cross-encoder — SentenceTransformersSimilarityRanker (Haystack ≥ 2.9).
Default: BAAI/bge-reranker-v2-m3 (multilingual, state of the art).
Note: TransformersSimilarityRanker is the legacy class and is deprecated.

LLM
────
OpenAI-compatible generator (OpenAIGenerator from haystack-ai).
Works with any OpenAI Chat Completions API-compatible endpoint:
  OpenAI, Ollama, vLLM, LM Studio, Groq, Together AI, …
"""


from config import Settings
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret, ComponentDevice
from haystack.components.rankers import SentenceTransformersSimilarityRanker
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack_integrations.components.embedders.fastembed import (
    FastembedSparseDocumentEmbedder,
    FastembedSparseTextEmbedder
)


def _onnx_providers(device: str) -> list[str]:
    """
    Map an embedding_device string to ONNX Runtime execution providers.

    FastEmbed (sparse embedders) uses ONNX Runtime instead of PyTorch.
    ONNX providers control whether inference runs on CPU or GPU.

      "cpu"             → ["CPUExecutionProvider"]
      "cuda" / "cuda:N" → ["CUDAExecutionProvider", "CPUExecutionProvider"]
      other             → ["CPUExecutionProvider"] (safe fallback)
    """
    if device.startswith("cuda"):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


# ---------------------------------------------------------------------------
# Dense embedders  (HuggingFace / sentence-transformers, local)
# ---------------------------------------------------------------------------


def build_document_embedder(settings: Settings) -> SentenceTransformersDocumentEmbedder:
    """
    Dense document embedder for the indexing pipeline.

    ``meta_fields_to_embed`` appends the section title and contextual prefix
    to the embedded text so that even short chunks are richly represented.
    """
    return SentenceTransformersDocumentEmbedder(
        model=settings.embedding_model,
        meta_fields_to_embed=["section_title", "context_prefix"],
        normalize_embeddings=True,
        device=ComponentDevice.from_str(settings.embedding_device),
    )


def build_text_embedder(settings: Settings) -> SentenceTransformersTextEmbedder:
    """Dense text embedder for the query pipeline."""
    return SentenceTransformersTextEmbedder(
        model=settings.embedding_model,
        normalize_embeddings=True,
        device=ComponentDevice.from_str(settings.embedding_device),
    )


# ---------------------------------------------------------------------------
# Sparse embedders  (SPLADE / BM42 via FastEmbed, local ONNX)
# ---------------------------------------------------------------------------


def build_sparse_document_embedder(settings: Settings):
    """
    Sparse document embedder for the indexing pipeline.

    Produces SparseEmbedding objects that Qdrant stores alongside dense
    vectors in the same collection.  True hybrid retrieval without a
    separate search engine.
    """
    return FastembedSparseDocumentEmbedder(
        model=settings.sparse_embedding_model,
        model_kwargs={"providers": _onnx_providers(settings.sparse_embedding_device)},
    )


def build_sparse_text_embedder(settings: Settings):
    """Sparse text embedder for the query pipeline."""
    return FastembedSparseTextEmbedder(
        model=settings.sparse_embedding_model,
        model_kwargs={"providers": _onnx_providers(settings.sparse_embedding_device)},
    )


# ---------------------------------------------------------------------------
# Reranker  (HuggingFace cross-encoder, local)
# ---------------------------------------------------------------------------


def build_reranker(settings: Settings) -> SentenceTransformersSimilarityRanker:
    """
    Cross-encoder reranker — scores every candidate document against the query.

    Far more accurate than bi-encoder similarity, at the cost of O(n) forward
    passes.  Applied AFTER RRF fusion on the top-K candidates.

    Uses SentenceTransformersSimilarityRanker (Haystack ≥ 2.9, replaces the
    deprecated TransformersSimilarityRanker).  Adds ONNX / OpenVINO backend
    support and query/document prefix injection for instruction-tuned models.
    """
    return SentenceTransformersSimilarityRanker(
        model=settings.reranker_model,
        top_k=settings.reranker_top_k,
        device=ComponentDevice.from_str(settings.reranker_device),
    )


# ---------------------------------------------------------------------------
# LLM Generator  (OpenAI-compatible)
# ---------------------------------------------------------------------------

def build_generator(settings: Settings) -> OpenAIGenerator:
    """
    OpenAI-compatible text generator.

    Set ``OPENAI_BASE_URL`` in .env to use a different backend:
      Ollama  → http://localhost:11434/v1
      vLLM    → http://localhost:8000/v1
      Groq    → https://api.groq.com/openai/v1
    """
    kwargs: dict = {
        "api_key": Secret.from_token(settings.openai_api_key),
        "model":   settings.llm_model,
    }
    if settings.openai_base_url:
        kwargs["api_base_url"] = settings.openai_base_url
    return OpenAIGenerator(**kwargs)
