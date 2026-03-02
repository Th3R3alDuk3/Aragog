from haystack.core.pipeline.async_pipeline import AsyncPipeline
from haystack.utils import Secret
from haystack.components.preprocessors import DocumentCleaner, MarkdownHeaderSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from components.chunk_enricher import ChunkEnricher
from components.content_analyzer import ContentAnalyzer
from components.docling_converter import DoclingConverter
from components.metadata_enricher import MetadataEnricher
from components.parent_child_splitter import ParentChildSplitter
from config import Settings
from pipelines._factories import (
    build_document_embedder,
    build_sparse_document_embedder,
)


def build_document_store(
        settings: Settings,
) -> QdrantDocumentStore:
    """
    Create and return the QdrantDocumentStore.

    ``use_sparse_embeddings=True`` tells Qdrant to create a named-vector
    collection that holds both dense and sparse vectors per document —
    enabling true hybrid retrieval in a single query.
    """
    return QdrantDocumentStore(
        url=settings.qdrant_url,
        api_key=Secret.from_token(settings.qdrant_api_key),
        index=settings.qdrant_collection,
        embedding_dim=settings.embedding_dimension,
        use_sparse_embeddings=True,
        similarity="cosine",
        recreate_index=False,
    )


def build_indexing_pipeline(
        settings: Settings,
) -> tuple[AsyncPipeline, QdrantDocumentStore]:

    document_store = build_document_store(settings)

    # --- Stage 1: conversion ---
    converter = DoclingConverter(docling_url=settings.docling_url)

    # --- Stage 2: document-level metadata ---
    meta_enricher = MetadataEnricher(
        embedding_provider="huggingface",
        embedding_model=settings.embedding_model,
        embedding_dimension=settings.embedding_dimension,
        doc_beginning_chars=settings.doc_beginning_chars,
    )

    # --- Stage 3: clean ---
    cleaner = DocumentCleaner(
        remove_empty_lines=True,
        remove_extra_whitespaces=True,
        remove_repeated_substrings=False,
    )

    # --- Stage 4: semantic heading-aware split ---
    header_splitter = MarkdownHeaderSplitter(
        keep_headers=True,
        skip_empty_documents=True,
    )

    # --- Stage 5: parent-child linking ---
    parent_child_splitter = ParentChildSplitter(
        child_chunk_size=settings.child_chunk_size,
        child_chunk_overlap=settings.child_chunk_overlap,
    )

    # --- Stage 6: chunk-level structural metadata ---
    chunk_enricher = ChunkEnricher()

    # --- Stage 7: contextual prefix + semantic metadata (one LLM call/chunk) ---
    analyzer = ContentAnalyzer(
        openai_api_key=settings.openai_api_key,
        llm_model=settings.llm_model,
        openai_url=settings.openai_url,
        taxonomy=settings.classification_taxonomy,
        max_concurrency=settings.analyzer_max_concurrency,
        max_chars=settings.analyzer_max_chars,
        doc_beginning_chars=settings.doc_beginning_chars,
    )

    # --- Stage 8 (optional): RAPTOR multi-level summaries ---
    raptor = None
    if settings.raptor_enabled:
        from components.raptor_summarizer import RaptorSummarizer
        raptor = RaptorSummarizer(
            openai_api_key=settings.openai_api_key,
            llm_model=settings.llm_model,
            openai_url=settings.openai_url,
            max_workers=settings.analyzer_max_concurrency,
        )

    # --- Stage 9: dense embedding ---
    dense_embedder = build_document_embedder(settings)

    # --- Stage 10: sparse embedding (SPLADE / BM42) ---
    sparse_embedder = build_sparse_document_embedder(settings)

    # --- Stage 11: write ---
    writer = DocumentWriter(
        document_store=document_store,
        policy=DuplicatePolicy.OVERWRITE,
    )

    pipeline = AsyncPipeline()
    pipeline.add_component("converter",             converter)
    pipeline.add_component("meta_enricher",         meta_enricher)
    pipeline.add_component("cleaner",               cleaner)
    pipeline.add_component("header_splitter",       header_splitter)
    pipeline.add_component("parent_child_splitter", parent_child_splitter)
    pipeline.add_component("chunk_enricher",        chunk_enricher)
    pipeline.add_component("analyzer",              analyzer)
    if raptor:
        pipeline.add_component("raptor", raptor)
    pipeline.add_component("dense_embedder",        dense_embedder)
    pipeline.add_component("sparse_embedder",       sparse_embedder)
    pipeline.add_component("writer",                writer)

    pipeline.connect("converter.documents",             "meta_enricher.documents")
    pipeline.connect("meta_enricher.documents",         "cleaner.documents")
    pipeline.connect("cleaner.documents",               "header_splitter.documents")
    pipeline.connect("header_splitter.documents",       "parent_child_splitter.documents")
    pipeline.connect("parent_child_splitter.documents", "chunk_enricher.documents")
    pipeline.connect("chunk_enricher.documents",        "analyzer.documents")
    if raptor:
        pipeline.connect("analyzer.documents",          "raptor.documents")
        pipeline.connect("raptor.documents",            "dense_embedder.documents")
    else:
        pipeline.connect("analyzer.documents",          "dense_embedder.documents")
    pipeline.connect("dense_embedder.documents",        "sparse_embedder.documents")
    pipeline.connect("sparse_embedder.documents",       "writer.documents")

    return pipeline, document_store
