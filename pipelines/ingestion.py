"""
Indexing pipeline (Haystack 2.x + Qdrant)

Full pipeline flow — Anthropic Contextual Retrieval Option A
────────────────────────────────────────────────────────────

  DoclingConverter          Convert PDF/DOCX/… → markdown via docling-serve (Gradio API)
        │
  MetadataEnricher          doc_id, title, word_count, indexed_at, indexed_at_ts,
        │                   language (via langdetect on the full doc, most accurate),
        │                   doc_beginning (first N chars for context prefix LLM),
        │                   embedding provenance.
        │
  DocumentCleaner           Normalise whitespace, remove empty lines.
        │
  MarkdownHeaderSplitter    Split at H1-H6 heading boundaries (built-in Haystack).
        │                   Sets meta["header"] and meta["parent_headers"] per chunk.
        │
  ParentChildSplitter       For sections > child_chunk_size words: split into smaller
        │                   child chunks using RecursiveDocumentSplitter (built-in).
        │                   Stores parent section text in meta["parent_content"].
        │
  ChunkContextEnricher      chunk_index, chunk_total, section_title (← meta["header"]),
        │                   section_path (← parent_headers › header), chunk_type.
        │
  ContentAnalyzer           ONE LLM call per chunk (parallelised, OpenAI-compat):
        │                   • context_prefix  → prepended to chunk before embedding
        │                   • summary, keywords, classification, entities
        │                   Stores original chunk in meta["original_content"].
        │                   Language is already in meta — not re-detected here.
        │
  [RaptorSummarizer]        OPTIONAL (RAPTOR_ENABLED=true): adds section-level
        │                   and document-level summary chunks alongside normal chunks.
        │                   chunk_type="raptor_section" | "raptor_doc"
        │                   Enables high-level retrieval for overview/summary questions.
        │
  DocumentEmbedder          Dense vector via SentenceTransformers (HuggingFace, local).
        │                   Embeds: context_prefix + section_title + chunk_content.
        │
  SparseDocumentEmbedder    SPLADE / BM42 sparse vector via FastEmbed (local ONNX).
        │
  DocumentWriter            Write to QdrantDocumentStore (OVERWRITE policy).

See models/README.md for the full metadata specification.
"""

from haystack import Pipeline
from haystack.components.preprocessors import DocumentCleaner, MarkdownHeaderSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from components.chunk_context_enricher import ChunkContextEnricher
from components.content_analyzer import ContentAnalyzer
from components.docling_converter import DoclingConverter
from components.metadata_enricher import MetadataEnricher
from components.parent_child_splitter import ParentChildSplitter
from config import Settings
from pipelines._embedders import (
    build_document_embedder,
    build_sparse_document_embedder,
)


def build_document_store(settings: Settings) -> QdrantDocumentStore:
    """
    Create and return the QdrantDocumentStore.

    ``use_sparse_embeddings=True`` tells Qdrant to create a named-vector
    collection that holds both dense and sparse vectors per document —
    enabling true hybrid retrieval in a single query.
    """
    return QdrantDocumentStore(
        url=settings.qdrant_url,
        api_key=Secret.from_token(settings.qdrant_api_key) if settings.qdrant_api_key else None,
        index=settings.qdrant_collection,
        embedding_dim=settings.embedding_dimension,
        use_sparse_embeddings=True,
        similarity="cosine",
        recreate_index=False,
    )


def build_indexing_pipeline(settings: Settings) -> tuple[Pipeline, QdrantDocumentStore]:
    document_store = build_document_store(settings)

    # --- Stage 1: conversion ---
    converter = DoclingConverter(docling_url=settings.docling_url)

    # --- Stage 2: document-level metadata ---
    meta_enricher = MetadataEnricher(
        embedding_model=settings.embedding_model,
        embedding_provider="huggingface",
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
    # MarkdownHeaderSplitter (built-in) splits at every H1-H6 boundary and
    # annotates each section with meta["header"] and meta["parent_headers"].
    header_splitter = MarkdownHeaderSplitter(
        keep_headers=True,       # headers stay in content (needed for context)
        skip_empty_documents=True,
    )

    # --- Stage 5: parent-child linking ---
    parent_child_splitter = ParentChildSplitter(
        child_chunk_size=settings.child_chunk_size,
        child_chunk_overlap=settings.child_chunk_overlap,
    )

    # --- Stage 6: chunk-level structural metadata ---
    chunk_enricher = ChunkContextEnricher()

    # --- Stage 7: contextual prefix + semantic metadata (one LLM call/chunk) ---
    analyzer = ContentAnalyzer(
        openai_api_key=settings.openai_api_key,
        llm_model=settings.analyzer_llm_model,
        openai_base_url=settings.openai_base_url,
        taxonomy=settings.classification_taxonomy,
        max_workers=settings.analyzer_max_workers,
        max_chars=settings.analyzer_max_chars,
        doc_beginning_chars=settings.doc_beginning_chars,
    )

    # --- Stage 8 (optional): RAPTOR multi-level summaries ---
    raptor = None
    if settings.raptor_enabled:
        from components.raptor_summarizer import RaptorSummarizer
        raptor = RaptorSummarizer(
            openai_api_key=settings.openai_api_key,
            llm_model=settings.analyzer_llm_model,
            openai_base_url=settings.openai_base_url,
            max_workers=settings.analyzer_max_workers,
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

    pipeline = Pipeline()
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

    pipeline.connect("converter.documents",                "meta_enricher.documents")
    pipeline.connect("meta_enricher.documents",            "cleaner.documents")
    pipeline.connect("cleaner.documents",                  "header_splitter.documents")
    pipeline.connect("header_splitter.documents",          "parent_child_splitter.documents")
    pipeline.connect("parent_child_splitter.documents",    "chunk_enricher.documents")
    pipeline.connect("chunk_enricher.documents",           "analyzer.documents")

    # Conditional RAPTOR stage: inserted between analyzer and dense_embedder
    if raptor:
        pipeline.connect("analyzer.documents",             "raptor.documents")
        pipeline.connect("raptor.documents",               "dense_embedder.documents")
    else:
        pipeline.connect("analyzer.documents",             "dense_embedder.documents")

    pipeline.connect("dense_embedder.documents",           "sparse_embedder.documents")
    pipeline.connect("sparse_embedder.documents",          "writer.documents")

    return pipeline, document_store
