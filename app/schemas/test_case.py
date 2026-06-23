from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class TestCaseOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    title: str
    preconditions: str | None
    steps: list
    expected: str | None
    case_type: str
    priority: str
    source: str | None
    score: int | None
    created_at: datetime


class AgentGenerateRequest(BaseModel):
    input: str = Field(..., min_length=1, description="需求主题或具体指令")


class JudgeGraphDebugRequest(BaseModel):
    point: str = Field(..., min_length=1, description="功能点名称或描述")
    case_type: str = Field("functional", description="functional/boundary/exception/negative")
    max_retry: int = Field(2, ge=0, le=5)
    threshold: int = Field(18, ge=5, le=25)


class TestCaseUpdate(BaseModel):
    title: str | None = Field(None, max_length=300)
    preconditions: str | None = None
    steps: list | None = None
    expected: str | None = None
    case_type: str | None = Field(None, max_length=30)
    priority: str | None = Field(None, max_length=10)


class ProjectStats(BaseModel):
    project_id: int
    doc_count: int
    case_count: int
    avg_score: float | None
    type_dist: dict[str, int]
    priority_dist: dict[str, int]


class BulkDeleteRequest(BaseModel):
    ids: list[int] = Field(..., min_length=1, max_length=500)


class BulkUpdatePatch(BaseModel):
    case_type: str | None = Field(None, max_length=30)
    priority: str | None = Field(None, max_length=10)


class BulkUpdateRequest(BaseModel):
    ids: list[int] = Field(..., min_length=1, max_length=500)
    patch: BulkUpdatePatch


class BulkResult(BaseModel):
    affected: int
