from fastapi import APIRouter, Depends, HTTPException, status

from models.schemas import TaskState
from routers._deps import get_task_store

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get(
    "/{task_id}",
    response_model=TaskState,
    summary="Get indexing task status",
    description=(
        "Returns the current state of an indexing task. "
        "Poll this endpoint after calling **POST /documents/index**. "
        "`step` reflects the active pipeline stage; `status` is one of "
        "`pending | running | done | failed`."
    ),
)
async def get_task(
    task_id: str,
    task_store: dict = Depends(get_task_store),
) -> TaskState:
    task = task_store.get(task_id)
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found.",
        )
    return task
