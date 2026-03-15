from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.generators import OpenAIGenerator
from haystack.components.rankers import SentenceTransformersSimilarityRanker
from haystack.utils import ComponentDevice, Secret
from haystack_integrations.components.embedders.fastembed import (
    FastembedSparseDocumentEmbedder,
    FastembedSparseTextEmbedder,
)

from core.config import Settings


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


_EMBED_META_FIELDS = [
    # structural context
    "section_title",
    "title",
    "document_type",
    # AI-generated semantic fields
    "summary",
    "keywords",
    # named entities (flat list[str])
    "ent_persons",
    "ent_organizations",
    "ent_products",
    "ent_laws",
]


def build_document_embedder(settings: Settings) -> SentenceTransformersDocumentEmbedder:
    """
    Dense document embedder for the indexing pipeline.

    ``meta_fields_to_embed`` appends structural and semantic metadata to the
    embedded text.  ``context_prefix`` is already prepended to document.content
    by ChunkAnalyzer and therefore not listed here.
    """
    return SentenceTransformersDocumentEmbedder(
        model=settings.embedding_model,
        meta_fields_to_embed=_EMBED_META_FIELDS,
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


def build_sparse_document_embedder(
    settings: Settings,
) -> FastembedSparseDocumentEmbedder:
    """
    Sparse document embedder for the indexing pipeline.

    Produces SparseEmbedding objects that Qdrant stores alongside dense
    vectors in the same collection.  True hybrid retrieval without a
    separate search engine.
    """
    return FastembedSparseDocumentEmbedder(
        model=settings.sparse_embedding_model,
        model_kwargs={"providers": _onnx_providers(settings.sparse_embedding_device)},
        meta_fields_to_embed=_EMBED_META_FIELDS,
    )


def build_sparse_text_embedder(settings: Settings) -> FastembedSparseTextEmbedder:
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
    """OpenAI-compatible text generator."""
    return OpenAIGenerator(
        api_base_url=settings.openai_url,
        api_key=Secret.from_token(settings.openai_api_key),
        model=settings.llm_model,
        timeout=settings.llm_timeout,
    )
