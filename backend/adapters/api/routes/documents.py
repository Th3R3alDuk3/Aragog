from asyncio import create_task
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import RedirectResponse

from adapters.api.deps import get_indexing_service, get_minio_store
from adapters.api.models.tasks import TaskCreatedResponse
from core.models.indexing import IndexCommand

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/index",
    response_model=TaskCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload and index a document (async)",
)
async def index_document(
    file: UploadFile = File(...),
    indexing_service=Depends(get_indexing_service),
) -> TaskCreatedResponse:
    max_upload_bytes = 50 * 1024 * 1024
    read_chunk = 64 * 1024
    chunks: list[bytes] = []
    total = 0

    while True:
        chunk = await file.read(read_chunk)
        if not chunk:
            break
        total += len(chunk)
        if total > max_upload_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="File too large. Maximum allowed is 50 MB.",
            )
        chunks.append(chunk)

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file must have a filename.",
        )

    command = IndexCommand(
        file_name=Path(file.filename).name,
        file_bytes=b"".join(chunks),
    )

    try:
        task = indexing_service.enqueue(command)
    except OverflowError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Task store is full: {error}",
        ) from error

    create_task(indexing_service.run(task.task_id))

    return TaskCreatedResponse(task_id=task.task_id, source=task.source)


@router.get("/download/{blob_key:path}", summary="Redirect to the original document")
async def download_document(
    blob_key: str,
    minio_store=Depends(get_minio_store),
) -> RedirectResponse:
    try:
        download_url = minio_store.download_url(blob_key)
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Original document could not be opened.",
        ) from error

    return RedirectResponse(download_url, status_code=status.HTTP_307_TEMPORARY_REDIRECT)
