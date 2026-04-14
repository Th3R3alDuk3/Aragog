from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from core.models.indexing import IndexResponse


class TaskStep(BaseModel):
    key: str = Field(..., description="Stable internal step key")
    label: str = Field(..., description="Human-readable step label")
    index: int = Field(..., description="0-based step index in the indexing sequence")
    status: Literal["pending", "running", "done", "failed"] = Field(
        ...,
        description="Progress state of this individual indexing step",
    )


class TaskState(BaseModel):
    model_config = ConfigDict(frozen=False)

    task_id: str = Field(..., description="Unique task ID")
    status: Literal["pending", "running", "done", "failed"] = Field(..., description="Task lifecycle status")
    step: str = Field(..., description="Current pipeline step")
    current_step_index: int = Field(
        default=-1,
        description="0-based index of the active or most recently completed pipeline step",
    )
    steps: list[TaskStep] = Field(default_factory=list)
    source: str = Field(..., description="Uploaded filename")
    created_at: datetime = Field(..., description="Task creation time (UTC)")
    updated_at: datetime = Field(..., description="Last status update time (UTC)")
    result: IndexResponse | None = Field(None, description="Populated when status=done")
    error: str | None = Field(None, description="Error message when status=failed")


class TaskInfo(BaseModel):
    task_id: str
    source: str
