from haystack import AsyncPipeline
from haystack.components.joiners import DocumentJoiner
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from pipelines._factories import (
    build_dense_text_embedder,
    build_sparse_text_embedder,
    build_dense_embedding_retriever,
    build_sparse_embedding_retriever,
    build_reranker,
)


def build_dense_retrieval_pipeline(document_store: QdrantDocumentStore) -> AsyncPipeline:

    pipeline = AsyncPipeline()
    pipeline.add_component("embedder",
        build_dense_text_embedder())
    pipeline.add_component("retriever",
        build_dense_embedding_retriever(document_store))
    pipeline.add_component("reranker",
        build_reranker())

    pipeline.connect("embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "reranker.documents")

    return pipeline


def build_sparse_retrieval_pipeline(document_store: QdrantDocumentStore) -> AsyncPipeline:

    pipeline = AsyncPipeline()
    pipeline.add_component("embedder",
        build_sparse_text_embedder())
    pipeline.add_component("retriever",
        build_sparse_embedding_retriever(document_store))
    pipeline.add_component("reranker",
        build_reranker())

    pipeline.connect("embedder.sparse_embedding", "retriever.query_sparse_embedding")
    pipeline.connect("retriever.documents", "reranker.documents")

    return pipeline


def build_hybrid_retrieval_pipeline(document_store: QdrantDocumentStore) -> AsyncPipeline:

    pipeline = AsyncPipeline()
    pipeline.add_component("dense_embedder",
        build_dense_text_embedder())
    pipeline.add_component("sparse_embedder",
        build_sparse_text_embedder())
    pipeline.add_component("dense_retriever",
        build_dense_embedding_retriever(document_store))
    pipeline.add_component("sparse_retriever",
        build_sparse_embedding_retriever(document_store))
    pipeline.add_component("joiner",
        DocumentJoiner(join_mode="concatenate"))
    pipeline.add_component("reranker",
        build_reranker())

    pipeline.connect("dense_embedder.embedding", "dense_retriever.query_embedding")
    pipeline.connect("sparse_embedder.sparse_embedding", "sparse_retriever.query_sparse_embedding")
    pipeline.connect("dense_retriever.documents", "joiner.documents")
    pipeline.connect("sparse_retriever.documents", "joiner.documents")
    pipeline.connect("joiner.documents", "reranker.documents")

    return pipeline
