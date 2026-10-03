from haystack import Pipeline
from haystack.components.joiners import DocumentJoiner
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from pipelines._factories import (
    build_dense_embedding_retriever,
    build_dense_text_embedder,
    build_reranker,
    build_sparse_embedding_retriever,
    build_sparse_text_embedder,
)


def build_dense_retrieval_pipeline(
    document_store: QdrantDocumentStore,
) -> Pipeline:
    return Pipeline().add_components({
        "embedder": build_dense_text_embedder(),
        "retriever": build_dense_embedding_retriever(document_store),
        "reranker": build_reranker(),
    }).connect_many([
        ("embedder.embedding", "retriever.query_embedding"),
        ("retriever.documents", "reranker.documents"),
    ])


def build_sparse_retrieval_pipeline(
    document_store: QdrantDocumentStore,
) -> Pipeline:
    return Pipeline().add_components({
        "embedder": build_sparse_text_embedder(),
        "retriever": build_sparse_embedding_retriever(document_store),
        "reranker": build_reranker(),
    }).connect_many([
        ("embedder.sparse_embedding", "retriever.query_sparse_embedding"),
        ("retriever.documents", "reranker.documents"),
    ])


def build_hybrid_retrieval_pipeline(
    document_store: QdrantDocumentStore,
) -> Pipeline:
    return Pipeline().add_components({
        "dense_embedder": build_dense_text_embedder(),
        "sparse_embedder": build_sparse_text_embedder(),
        "dense_retriever": build_dense_embedding_retriever(document_store),
        "sparse_retriever": build_sparse_embedding_retriever(document_store),
        "joiner": DocumentJoiner(join_mode="concatenate"),
        "reranker": build_reranker(),
    }).connect_many([
        ("dense_embedder.embedding", "dense_retriever.query_embedding"),
        ("sparse_embedder.sparse_embedding", "sparse_retriever.query_sparse_embedding"),
        ("dense_retriever.documents", "joiner.documents"),
        ("sparse_retriever.documents", "joiner.documents"),
        ("joiner.documents", "reranker.documents"),
    ])
