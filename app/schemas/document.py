from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    filename: str
    mime_type: str | None = None
    size: int | None = None
    chunk_count: int
    status: str
    created_at: datetime
