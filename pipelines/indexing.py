from haystack import AsyncPipeline
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy

from pipelines._factories import (
    build_document_store,
    build_converter,
    build_chunker,
    build_chunk_enricher,
    build_dense_document_embedder,
    build_sparse_document_embedder,
)


def build_indexing_pipeline() -> AsyncPipeline:

    document_store = build_document_store()

    pipeline = AsyncPipeline()
    pipeline.add_component("converter",
        build_converter())
    pipeline.add_component("chunker",
        build_chunker())
    pipeline.add_component("chunk_enricher",
        build_chunk_enricher())
    pipeline.add_component("dense_embedder",
        build_dense_document_embedder())
    pipeline.add_component("sparse_embedder",
        build_sparse_document_embedder())
    pipeline.add_component("writer",
        DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP))

    pipeline.connect("converter.documents", "chunker.documents")
    pipeline.connect("chunker.documents", "chunk_enricher.documents")
    pipeline.connect("chunk_enricher.documents", "dense_embedder.documents")
    pipeline.connect("dense_embedder.documents", "sparse_embedder.documents")
    pipeline.connect("sparse_embedder.documents", "writer.documents")

    return pipeline
