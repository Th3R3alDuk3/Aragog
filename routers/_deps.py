"""
FastAPI dependency providers — fetch pipeline instances from app.state
(populated at startup in main.py's lifespan context manager) and inject
them into route handlers via FastAPI's Depends() mechanism.
"""

from asyncio import Semaphore

from fastapi import Request
from haystack.core.pipeline.async_pipeline import AsyncPipeline
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from components.query_analyzer import QueryAnalyzer
from config import Settings


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


# --- Infrastructure ---


def get_document_store(request: Request) -> QdrantDocumentStore:
    return request.app.state.document_store


def get_minio_store(request: Request):
    return request.app.state.minio_store


# --- Indexing ---


def get_indexing_pipeline(request: Request) -> AsyncPipeline:
    return request.app.state.indexing_pipeline


def get_indexing_semaphore(request: Request) -> Semaphore:
    return request.app.state.indexing_semaphore


# --- Retrieval ---


def get_query_analyzer(request: Request) -> QueryAnalyzer:
    return request.app.state.query_analyzer


def get_retrieval_pipeline(request: Request) -> AsyncPipeline:
    return request.app.state.retrieval_pipeline


# --- Generation ---


def get_generation_pipeline(request: Request) -> AsyncPipeline:
    return request.app.state.generation_pipeline


# --- Application ---


def get_task_store(request: Request) -> dict:
    return request.app.state.tasks
