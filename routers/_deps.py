"""
FastAPI dependency providers — fetch pipeline instances from app.state
(populated at startup in main.py's lifespan context manager) and inject
them into route handlers via FastAPI's Depends() mechanism.
"""

import asyncio

from fastapi import Request
from haystack import Pipeline
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from components.query_analyzer import QueryAnalyzer
from config import Settings


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_document_store(request: Request) -> QdrantDocumentStore:
    return request.app.state.document_store


def get_indexing_pipeline(request: Request) -> Pipeline:
    return request.app.state.indexing_pipeline


def get_retrieval_pipeline(request: Request) -> Pipeline:
    return request.app.state.retrieval_pipeline


def get_generation_pipeline(request: Request) -> Pipeline:
    return request.app.state.generation_pipeline


def get_query_analyzer(request: Request) -> QueryAnalyzer:
    return request.app.state.query_analyzer


def get_hyde_generator(request: Request):
    """Returns the HyDEGenerator instance, or None if HYDE_ENABLED=false."""
    return request.app.state.hyde_generator


def get_colbert_reranker(request: Request):
    """Returns the ColBERTReranker instance, or None if COLBERT_ENABLED=false."""
    return request.app.state.colbert_reranker


def get_minio_store(request: Request):
    """Returns the MinioStore instance."""
    return request.app.state.minio_store


def get_task_store(request: Request) -> dict:
    """Returns the in-memory task store (dict[task_id, TaskState])."""
    return request.app.state.tasks


def get_indexing_semaphore(request: Request) -> asyncio.Semaphore:
    """Returns the semaphore that caps concurrent indexing jobs."""
    return request.app.state.indexing_semaphore
