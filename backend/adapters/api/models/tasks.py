from pydantic import BaseModel, Field


class TaskCreatedResponse(BaseModel):
    task_id: str = Field(..., description="Unique task ID — poll GET /tasks/{task_id} for progress")
    source: str = Field(..., description="Uploaded filename")
    message: str = "Ingestion started. Poll /tasks/{task_id} for progress."
