from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ChatSessionCreate(BaseModel):
    title: str = Field("新对话", max_length=200)


class ChatSessionUpdate(BaseModel):
    title: str | None = Field(None, max_length=200)


class ChatSessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    title: str
    created_at: datetime
    updated_at: datetime


class ChatMessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: int
    role: str
    content: str
    created_at: datetime


class ChatRequest(BaseModel):
    input: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    answer: str
