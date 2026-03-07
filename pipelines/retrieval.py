from haystack.components.joiners import DocumentJoiner
from haystack.components.retrievers.auto_merging_retriever import AutoMergingRetriever
from haystack.core.pipeline.async_pipeline import AsyncPipeline
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
    children_store: QdrantDocumentStore,
    parents_store: QdrantDocumentStore,
) -> AsyncPipeline:
    """Build the hybrid retrieval pipeline.

    Combines dense and sparse retrievers via Reciprocal Rank Fusion, followed
    by AutoMergingRetriever (parent-context swap) and a cascade reranker.
    If ``COLBERT_ENABLED=true``, ColBERT runs as a pre-filter before the
    cross-encoder. Called once at startup; reused for every query.

    Args:
        settings:        Application settings for retriever top-k values and
                         embedder/reranker model configuration.
        children_store:  QdrantDocumentStore for the children collection
                         (dense + sparse, retrieval target).
        parents_store:   QdrantDocumentStore for the parents collection
                         (fetched by ID via AutoMergingRetriever).

    Returns:
        A Haystack ``AsyncPipeline`` with components ``dense_embedder``,
        ``sparse_embedder``, ``dense_retriever``, ``sparse_retriever``,
        ``joiner``, ``auto_merging_retriever``, ``reranker``, and optionally
        ``colbert_reranker`` and HyDE components.
    """
    
    # --- Stage 1: dense embedding — original query (always active) ---
    dense_embedder = build_text_embedder(settings)

    # --- Stage 2: sparse query embedding (SPLADE / BM42) ---
    sparse_embedder = build_sparse_text_embedder(settings)

    # --- Stage 3: dense retrieval — original query branch ---
    dense_retriever = QdrantEmbeddingRetriever(
        document_store=children_store,
        top_k=settings.dense_retriever_top_k,
    )

    # --- Stage 4: sparse retrieval ---
    sparse_retriever = QdrantSparseEmbeddingRetriever(
        document_store=children_store,
        top_k=settings.sparse_retriever_top_k,
    )

    # --- Stage 5: RRF fusion ---
    joiner = DocumentJoiner(
        join_mode="reciprocal_rank_fusion"
    )

    # --- Stage 6: AutoMergingRetriever — swap children → parent when threshold met ---
    auto_merger = AutoMergingRetriever(
        document_store=parents_store,
        threshold=settings.auto_merge_threshold,
    )

    # --- Stage 7: cross-encoder reranker (final pass, sees AutoMerger output) ---
    reranker = build_reranker(settings)

    # --- Stage 8 (optional): HyDE second dense branch ---
    # Only wired into the pipeline when HYDE_ENABLED=true at startup.
    # hyde_generator receives the raw query and emits a hypothetical passage;
    # its text output feeds dense_embedder_hyde so the LLM call is fully
    # encapsulated inside the pipeline.
    if settings.hyde_enabled:
        from components.hyde_generator import HyDEGenerator
        hyde_generator       = HyDEGenerator(
            openai_url=settings.openai_url,
            openai_api_key=settings.openai_api_key,
            llm_model=settings.effective_instruct_model,
        )
        dense_embedder_hyde  = build_text_embedder(settings)
        dense_retriever_hyde = QdrantEmbeddingRetriever(
            document_store=children_store,
            top_k=settings.dense_retriever_top_k,
        )

    # --- Stage 9 (optional): ColBERT late-interaction pre-filter (before cross-encoder) ---
    colbert_reranker = None
    if settings.colbert_enabled:
        from components.colbert_reranker import ColBERTReranker
        colbert_reranker = ColBERTReranker(
            model_name=settings.colbert_model,
            top_k=settings.colbert_top_k,
            device=settings.colbert_device,
        )

    retrieval = AsyncPipeline()
    retrieval.add_component("dense_embedder",        dense_embedder)
    retrieval.add_component("sparse_embedder",       sparse_embedder)
    retrieval.add_component("dense_retriever",       dense_retriever)
    retrieval.add_component("sparse_retriever",      sparse_retriever)
    retrieval.add_component("joiner",                joiner)
    retrieval.add_component("auto_merging_retriever", auto_merger)
    retrieval.add_component("reranker",              reranker)
    if settings.hyde_enabled:
        retrieval.add_component("hyde_generator",       hyde_generator)
        retrieval.add_component("dense_embedder_hyde",  dense_embedder_hyde)
        retrieval.add_component("dense_retriever_hyde", dense_retriever_hyde)
    if colbert_reranker is not None:
        retrieval.add_component("colbert_reranker", colbert_reranker)

    retrieval.connect("dense_embedder.embedding",         "dense_retriever.query_embedding")
    retrieval.connect("dense_retriever.documents",        "joiner.documents")
    retrieval.connect("sparse_embedder.sparse_embedding", "sparse_retriever.query_sparse_embedding")
    retrieval.connect("sparse_retriever.documents",       "joiner.documents")
    if settings.hyde_enabled:
        retrieval.connect("hyde_generator.text",            "dense_embedder_hyde.text")
        retrieval.connect("dense_embedder_hyde.embedding",  "dense_retriever_hyde.query_embedding")
        retrieval.connect("dense_retriever_hyde.documents", "joiner.documents")
    retrieval.connect("joiner.documents", "auto_merging_retriever.documents")
    if colbert_reranker is not None:
        retrieval.connect("auto_merging_retriever.documents", "colbert_reranker.documents")
        retrieval.connect("colbert_reranker.documents",       "reranker.documents")
    else:
        retrieval.connect("auto_merging_retriever.documents", "reranker.documents")

    return retrieval
