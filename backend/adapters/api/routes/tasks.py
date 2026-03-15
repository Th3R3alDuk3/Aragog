from fastapi import APIRouter, Depends, HTTPException, status

from adapters.api.deps import get_task_store
from core.models.tasks import TaskState

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("/{task_id}", response_model=TaskState, summary="Get indexing task status")
async def get_task(
    task_id: str,
    task_store: dict[str, TaskState] = Depends(get_task_store),
) -> TaskState:
    task = task_store.get(task_id)
    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found.",
        )
    return task
