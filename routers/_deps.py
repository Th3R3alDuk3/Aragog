"""
FastAPI dependency providers for the indexing and query pipelines.

Both pipelines are built once at application startup (see main.py) and stored
on the FastAPI app state.  The dependency functions simply fetch them from state
so FastAPI can inject them into route handlers.
"""

from fastapi import Request
from haystack import Pipeline
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore


def get_document_store(request: Request) -> QdrantDocumentStore:
    return request.app.state.document_store


def get_ingestion_pipeline(request: Request) -> Pipeline:
    return request.app.state.ingestion_pipeline


def get_retrieval_pipeline(request: Request) -> Pipeline:
    return request.app.state.retrieval_pipeline


def get_generation_pipeline(request: Request) -> Pipeline:
    return request.app.state.generation_pipeline


def get_query_analyzer(request: Request):
    """Returns the QueryAnalyzer instance (also aliased as query_decomposer)."""
    return request.app.state.query_analyzer


def get_hyde_generator(request: Request):
    """Returns the HyDEGenerator instance, or None if HYDE_ENABLED=false."""
    return request.app.state.hyde_generator


def get_colbert_reranker(request: Request):
    """Returns the ColBERTReranker instance, or None if COLBERT_ENABLED=false."""
    return request.app.state.colbert_reranker


def get_minio_store(request: Request):
    """Returns the MinioStore instance, or None if MinIO is not configured."""
    return request.app.state.minio_store
