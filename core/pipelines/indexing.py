from haystack.components.preprocessors import DocumentCleaner
from haystack.components.writers import DocumentWriter
from haystack.core.pipeline.async_pipeline import AsyncPipeline
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from core.components.chunk_analyzer import ChunkAnalyzer
from core.components.chunk_annotator import ChunkAnnotator
from core.components.context_injector import ContextInjector
from core.components.docling import Docling
from core.components.document_analyzer import DocumentAnalyzer
from core.components.parent_child_splitter import ParentChildSplitter
from core.config import Settings
from core.pipelines._factories import (
    build_dense_document_embedder,
    build_sparse_document_embedder,
)


def build_children_store(settings: Settings) -> QdrantDocumentStore:
    """Children collection — dense + sparse vectors, hybrid retrieval target."""
    return QdrantDocumentStore(
        url=settings.qdrant_url,
        api_key=Secret.from_token(settings.qdrant_api_key),
        index=settings.qdrant_children_collection,
        embedding_dim=settings.embedding_dimension,
        use_sparse_embeddings=True,
        similarity="cosine",
        recreate_index=False,
    )


def build_parents_store(settings: Settings) -> QdrantDocumentStore:
    """Parents collection — dense vectors only; fetched by ID at query time."""
    return QdrantDocumentStore(
        url=settings.qdrant_url,
        api_key=Secret.from_token(settings.qdrant_api_key),
        index=settings.qdrant_parents_collection,
        embedding_dim=settings.embedding_dimension,
        use_sparse_embeddings=False,
        similarity="cosine",
        recreate_index=False,
    )


def build_indexing_pipeline(
        settings: Settings,
) -> tuple[AsyncPipeline, QdrantDocumentStore, QdrantDocumentStore]:
    """Build the indexing pipeline and both Qdrant document stores.

    Returns:
        (pipeline, children_store, parents_store)
    """

    children_store = build_children_store(settings)
    parents_store  = build_parents_store(settings)

    # --- Stage 1: conversion ---
    converter = Docling(docling_url=settings.docling_url)

    # --- Stage 2: document-level metadata ---
    document_analyzer = DocumentAnalyzer(
        openai_url=settings.openai_url,
        openai_api_key=settings.openai_api_key,
        llm_model=settings.effective_instruct_model,
        embedding_provider="huggingface",
        embedding_model=settings.embedding_model,
        embedding_dimension=settings.embedding_dimension,
        doc_beginning_chars=settings.doc_beginning_chars,
        max_concurrency=settings.analyzer_max_concurrency,
    )

    # --- Stage 3: clean ---
    cleaner = DocumentCleaner(
        remove_empty_lines=True,
        remove_extra_whitespaces=True,
        remove_repeated_substrings=False,
    )

    # --- Stage 4: parent-child linking via HierarchicalDocumentSplitter ---
    # Outputs: children (→ enrich/analyze/embed branch)
    #          parents  (→ embed + write to parents collection)
    parent_child_splitter = ParentChildSplitter(
        parent_chunk_size=settings.parent_chunk_size,
        child_chunk_size=settings.child_chunk_size,
        child_chunk_overlap=settings.child_chunk_overlap,
    )

    # --- Children branch ---

    # Stage 5: chunk-level structural metadata
    chunk_annotator = ChunkAnnotator()

    # Stage 6: contextual prefix + semantic metadata (one LLM call/chunk)
    # Full document passed as context — Anthropic Contextual Retrieval approach.
    chunk_analyzer = ChunkAnalyzer(
        openai_api_key=settings.openai_api_key,
        llm_model=settings.effective_instruct_model,
        openai_url=settings.openai_url,
        taxonomy=settings.classification_taxonomy,
        max_concurrency=settings.analyzer_max_concurrency,
        max_chars=settings.analyzer_max_chars,
        max_doc_chars=settings.contextual_doc_max_chars,
        anthropic_caching_enabled=settings.anthropic_caching_enabled,
        anthropic_api_key=settings.anthropic_api_key,
    )

    # Stage 8 (optional): RAPTOR multi-level summaries
    raptor = None
    if settings.raptor_enabled:
        from core.components.raptor import RAPTOR
        raptor = RAPTOR(
            openai_api_key=settings.openai_api_key,
            llm_model=settings.effective_instruct_model,
            openai_url=settings.openai_url,
            max_workers=settings.analyzer_max_concurrency,
        )

    # Stage 9: inject context prefix — applied BEFORE both embedders so that
    # both sparse and dense vectors encode the contextually enriched text
    # (Anthropic Contextual Retrieval paper recommendation).
    context_injector = ContextInjector()

    # Stage 10: sparse embedding (context-prefixed content)
    sparse_embedder = build_sparse_document_embedder(settings)

    # Stage 11: dense embedding (same context-prefixed content)
    dense_embedder = build_dense_document_embedder(settings)

    # Stage 12: write children
    children_writer = DocumentWriter(
        document_store=children_store,
        policy=DuplicatePolicy.OVERWRITE,
    )

    # --- Parents branch ---

    # Stage P1: write parents (no embedding needed — AutoMergingRetriever fetches by ID)
    parents_writer = DocumentWriter(
        document_store=parents_store,
        policy=DuplicatePolicy.OVERWRITE,
    )

    pipeline = AsyncPipeline()
    pipeline.add_component("converter",             converter)
    pipeline.add_component("document_analyzer",     document_analyzer)
    pipeline.add_component("cleaner",               cleaner)
    pipeline.add_component("parent_child_splitter", parent_child_splitter)
    # children branch
    pipeline.add_component("chunk_annotator",        chunk_annotator)
    pipeline.add_component("chunk_analyzer",         chunk_analyzer)
    if raptor:
        pipeline.add_component("raptor",             raptor)
    pipeline.add_component("context_injector",       context_injector)
    pipeline.add_component("sparse_embedder",        sparse_embedder)
    pipeline.add_component("dense_embedder",         dense_embedder)
    pipeline.add_component("children_writer",        children_writer)
    # parents branch
    pipeline.add_component("parents_writer",         parents_writer)

    pipeline.connect("converter.documents",                    "document_analyzer.documents")
    pipeline.connect("document_analyzer.documents",            "cleaner.documents")
    pipeline.connect("cleaner.documents",                      "parent_child_splitter.documents")
    # children branch
    pipeline.connect("parent_child_splitter.children",         "chunk_annotator.documents")
    pipeline.connect("chunk_annotator.documents",              "chunk_analyzer.documents")
    if raptor:
        pipeline.connect("chunk_analyzer.documents",           "raptor.documents")
        pipeline.connect("raptor.documents",                   "context_injector.documents")
    else:
        pipeline.connect("chunk_analyzer.documents",           "context_injector.documents")
    # Context is now injected before BOTH embedders (Anthropic paper: apply to dense + sparse)
    pipeline.connect("context_injector.documents",             "sparse_embedder.documents")
    pipeline.connect("sparse_embedder.documents",              "dense_embedder.documents")
    pipeline.connect("dense_embedder.documents",               "children_writer.documents")
    # parents branch (no embedding — stored by ID only)
    pipeline.connect("parent_child_splitter.parents",          "parents_writer.documents")

    return pipeline, children_store, parents_store
