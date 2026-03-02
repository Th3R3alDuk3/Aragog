"""
Retrieval pipeline — Advanced Hybrid RAG (Haystack 2.x + Qdrant)

Pipeline flow per sub-question
───────────────────────────────

  Query text
      │
      ├─ SentenceTransformersTextEmbedder ──→ dense vector
      │                                              │
      │                                   QdrantEmbeddingRetriever
      │                                              │
      └─ FastembedSparseTextEmbedder ──→ sparse vector
                                                     │
                                          QdrantSparseEmbeddingRetriever
                                                     │
                                   DocumentJoiner (Reciprocal Rank Fusion)
                                                     │
                              SentenceTransformersSimilarityRanker (cross-encoder)

Multi-question handling
───────────────────────
Multi-question decomposition happens at the router layer (routers/query.py),
not inside this pipeline.  The router calls QueryAnalyzer, runs this
pipeline once per sub-question, merges document sets, and then passes the
combined context to the generation pipeline.

Parent-child retrieval
──────────────────────
Child chunks (small, precise) are stored in Qdrant and retrieved.
swap_to_parent_content() replaces each child's content with
meta["parent_content"] (full markdown section) before the LLM
generates its answer — broader context without sacrificing retrieval precision.
"""

from haystack import Document
from haystack.core.pipeline.async_pipeline import AsyncPipeline
from haystack.components.joiners import DocumentJoiner
from haystack_integrations.components.retrievers.qdrant import (
    QdrantEmbeddingRetriever,
    QdrantSparseEmbeddingRetriever,
)
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from config import Settings
from pipelines._factories import (
    build_reranker,
    build_sparse_text_embedder,
    build_text_embedder,
)


def build_retrieval_pipeline(
    settings: Settings,
    document_store: QdrantDocumentStore,
) -> AsyncPipeline:
    """Build the hybrid retrieval pipeline.

    Combines dense and sparse retrievers via Reciprocal Rank Fusion, followed
    by a cross-encoder reranker.  Called once at startup; the returned pipeline
    is reused for every query.

    Args:
        settings:       Application settings for retriever top-k values and
                        embedder/reranker model configuration.
        document_store: Shared QdrantDocumentStore instance (created by the
                        ingestion pipeline builder).

    Returns:
        A Haystack ``AsyncPipeline`` with components ``dense_embedder``,
        ``sparse_embedder``, ``dense_retriever``, ``sparse_retriever``,
        ``joiner``, and ``reranker`` wired in sequence.
    """
    dense_embedder   = build_text_embedder(settings)
    sparse_embedder  = build_sparse_text_embedder(settings)
    dense_retriever  = QdrantEmbeddingRetriever(
        document_store=document_store,
        top_k=settings.dense_retriever_top_k,
    )
    sparse_retriever = QdrantSparseEmbeddingRetriever(
        document_store=document_store,
        top_k=settings.sparse_retriever_top_k,
    )
    joiner = DocumentJoiner(
        join_mode="reciprocal_rank_fusion",
        top_k=max(settings.dense_retriever_top_k, settings.sparse_retriever_top_k),
    )
    reranker = build_reranker(settings)

    retrieval = AsyncPipeline()
    retrieval.add_component("dense_embedder",   dense_embedder)
    retrieval.add_component("sparse_embedder",  sparse_embedder)
    retrieval.add_component("dense_retriever",  dense_retriever)
    retrieval.add_component("sparse_retriever", sparse_retriever)
    retrieval.add_component("joiner",           joiner)
    retrieval.add_component("reranker",         reranker)
    retrieval.connect("dense_embedder.embedding",         "dense_retriever.query_embedding")
    retrieval.connect("dense_retriever.documents",        "joiner.documents")
    retrieval.connect("sparse_embedder.sparse_embedding", "sparse_retriever.query_sparse_embedding")
    retrieval.connect("sparse_retriever.documents",       "joiner.documents")
    retrieval.connect("joiner.documents",                 "reranker.documents")

    return retrieval


# ---------------------------------------------------------------------------
# Parent-content swap  (applied at service layer, not inside the pipeline)
# ---------------------------------------------------------------------------

def swap_to_parent_content(documents: list) -> list:
    """Replace each retrieved child chunk's content with its parent section text.

    Called by the router after retrieval, before passing documents to the
    generation pipeline.  Gives the LLM full section context while keeping
    retrieval precise (small child chunks for embedding/retrieval).

    If a document has no ``parent_content`` the original content is kept.
    Deduplicates by parent content to avoid sending the same section twice.

    Args:
        documents: Retrieved child chunk documents from the reranker.

    Returns:
        Deduplicated list of documents with content replaced by
        ``meta["parent_content"]`` where available.
    """
    seen: set[str] = set()
    result: list[Document] = []

    for doc in documents:
        parent = doc.meta.get("parent_content") or doc.content
        key    = parent[:200]               # dedup key: first 200 chars
        if key in seen:
            continue
        seen.add(key)
        result.append(
            Document(content=parent, meta=doc.meta, id=doc.id, score=doc.score)
        )

    return result
