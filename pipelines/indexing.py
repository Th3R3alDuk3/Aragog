from haystack import Pipeline
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from pipelines._factories import (
    build_chunker,
    build_converter,
    build_dense_document_embedder,
    build_enricher,
    build_sparse_document_embedder,
)


def build_indexing_pipeline(
    document_store: QdrantDocumentStore,
) -> Pipeline:
    return Pipeline().add_components({
        "converter": build_converter(),
        "chunker": build_chunker(),
        "enricher": build_enricher(),
        "dense_embedder": build_dense_document_embedder(),
        "sparse_embedder": build_sparse_document_embedder(),
        "writer": DocumentWriter(
            document_store=document_store,
            policy=DuplicatePolicy.OVERWRITE,
        ),
    }).connect_many([
        ("converter.documents", "chunker.documents"),
        ("chunker.documents", "enricher.documents"),
        ("enricher.documents", "dense_embedder.documents"),
        ("dense_embedder.documents", "sparse_embedder.documents"),
        ("sparse_embedder.documents", "writer.documents"),
    ])
