from haystack import Pipeline
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from pipelines._factories import (
    build_chunk_enricher,
    build_chunker,
    build_converter,
    build_dense_document_embedder,
    build_sparse_document_embedder,
)


def build_indexing_pipeline(
    document_store: QdrantDocumentStore,
) -> Pipeline:

    pipeline = Pipeline()
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
        DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE))

    pipeline.connect("converter.documents", "chunker.documents")
    pipeline.connect("chunker.documents", "chunk_enricher.documents")
    pipeline.connect("chunk_enricher.documents", "dense_embedder.documents")
    pipeline.connect("dense_embedder.documents", "sparse_embedder.documents")
    pipeline.connect("sparse_embedder.documents", "writer.documents")

    return pipeline
