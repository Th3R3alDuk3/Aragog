from fastapi import Request

from core.config import Settings
from core.runtime import RagRuntime
from core.storage.task_store import TaskStore


def get_runtime(request: Request) -> RagRuntime:
    return request.app.state.runtime


def get_settings(request: Request) -> Settings:
    return get_runtime(request).settings


def get_minio_store(request: Request):
    return get_runtime(request).minio_store


def get_query_engine(request: Request):
    return get_runtime(request).query_engine


def get_indexing_service(request: Request):
    return get_runtime(request).indexing_service


def get_task_store(request: Request) -> TaskStore:
    return get_runtime(request).task_store
