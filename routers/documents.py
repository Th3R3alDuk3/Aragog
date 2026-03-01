import asyncio
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from models.schemas import TaskCreatedResponse, TaskState
from routers._deps import get_document_store, get_indexing_pipeline, get_indexing_semaphore, get_minio_store, get_task_store
from services import indexing as indexing_service

logger = logging.getLogger(__name__)

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
    pipeline=Depends(get_indexing_pipeline),
    document_store=Depends(get_document_store),
    minio_store=Depends(get_minio_store),
    task_store: dict = Depends(get_task_store),
    semaphore: asyncio.Semaphore = Depends(get_indexing_semaphore),
) -> TaskCreatedResponse:
    file_bytes    = await file.read()
    original_name = Path(file.filename).name if file.filename else "upload"

    now  = datetime.now(timezone.utc)
    task = TaskState(
        task_id    = str(uuid.uuid4()),
        status     = "pending",
        step       = "pending",
        source     = original_name,
        created_at = now,
        updated_at = now,
    )
    try:
        task_store[task.task_id] = task
    except OverflowError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))

    async def _guarded():
        async with semaphore:
            await indexing_service.run_indexing(
                task, pipeline, document_store, minio_store, file_bytes, original_name,
            )

    asyncio.create_task(_guarded())

    return TaskCreatedResponse(task_id=task.task_id, source=original_name)
