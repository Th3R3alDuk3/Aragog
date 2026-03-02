from asyncio import Semaphore, create_task
from datetime import datetime, timezone
from logging import getLogger
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from haystack.core.pipeline.async_pipeline import AsyncPipeline
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from models.api import TaskCreatedResponse, TaskState
from routers._deps import (
    get_document_store,
    get_indexing_pipeline,
    get_indexing_semaphore,
    get_minio_store,
    get_task_store,
)
from services import indexing as indexing_service
from services.minio_store import MinioStore

logger = getLogger(__name__)


router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/index",
    response_model=TaskCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload and index a document (async)",
    description=(
        "Upload a single file (PDF, DOCX, PPTX, …). Returns a `task_id` immediately. "
        "Poll **GET /tasks/{task_id}** to track progress through the indexing pipeline. "
        "Existing chunks for the same filename are deleted before re-indexing."
    ),
)
async def index_document(
    file: UploadFile = File(...),
    document_store: QdrantDocumentStore = Depends(get_document_store),
    minio_store: MinioStore = Depends(get_minio_store),
    indexing_pipeline: AsyncPipeline = Depends(get_indexing_pipeline),
    indexing_semaphore: Semaphore = Depends(get_indexing_semaphore),
    task_store: dict = Depends(get_task_store),
) -> TaskCreatedResponse:

    file_bytes = await file.read()
    file_name = Path(file.filename).name

    now = datetime.now(timezone.utc)
    task = TaskState(
        task_id=str(uuid4()),
        status="pending",
        step="pending",
        source=file_name,
        created_at=now,
        updated_at=now,
    )

    try:
        task_store[task.task_id] = task
    except OverflowError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Task store is full, cannot accept new indexing jobs: {error}",
        )

    async def _task():
        async with indexing_semaphore:
            await indexing_service.run_indexing(
                task=task,
                document_store=document_store,
                minio_store=minio_store,
                pipeline=indexing_pipeline,
                file_name=file_name,
                file_bytes=file_bytes,
            )

    create_task(_task())

    return TaskCreatedResponse(
        task_id=task.task_id,
        source=file_name,
    )
