from asyncio import Semaphore, create_task
from datetime import datetime, timezone
from logging import getLogger
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import RedirectResponse
from haystack.core.pipeline.async_pipeline import AsyncPipeline
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from models.api import TaskCreatedResponse, TaskState
from routers._deps import (
    get_children_store,
    get_indexing_pipeline,
    get_indexing_semaphore,
    get_minio_store,
    get_parents_store,
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
    children_store: QdrantDocumentStore = Depends(get_children_store),
    parents_store: QdrantDocumentStore = Depends(get_parents_store),
    minio_store: MinioStore = Depends(get_minio_store),
    indexing_pipeline: AsyncPipeline = Depends(get_indexing_pipeline),
    indexing_semaphore: Semaphore = Depends(get_indexing_semaphore),
    task_store: dict = Depends(get_task_store),
) -> TaskCreatedResponse:

    _MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
    _READ_CHUNK = 64 * 1024              # 64 KB per read call
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_READ_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File too large. Maximum allowed is 50 MB.",
            )
        chunks.append(chunk)
    file_bytes = b"".join(chunks)
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file must have a filename.",
        )
    file_name = Path(file.filename).name

    now = datetime.now(timezone.utc)
    steps = indexing_service.build_indexing_steps(indexing_pipeline)
    task = TaskState(
        task_id=str(uuid4()),
        status="pending",
        step="pending",
        current_step_index=-1,
        steps=steps,
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
                children_store=children_store,
                parents_store=parents_store,
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


@router.get(
    "/download/{minio_key:path}",
    summary="Redirect to the original document in MinIO",
    description=(
        "Creates a fresh presigned MinIO download URL and redirects to it. "
        "Use this stable backend link instead of storing expiring object URLs."
    ),
)
async def download_document(
    minio_key: str,
    minio_store: MinioStore = Depends(get_minio_store),
) -> RedirectResponse:
    try:
        download_url = minio_store.download_url(minio_key)
    except Exception as error:
        logger.warning("Could not create download URL for '%s': %s", minio_key, error)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Original document could not be opened.",
        ) from error

    return RedirectResponse(download_url, status_code=status.HTTP_307_TEMPORARY_REDIRECT)
