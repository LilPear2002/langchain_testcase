"""全局看板汇总接口。"""
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db import models
from app.db.session import get_db

router = APIRouter(tags=["dashboard"])


class ProjectSummary(BaseModel):
    id: int
    name: str
    doc_count: int
    case_count: int
    avg_score: float | None


class DashboardTotals(BaseModel):
    project_count: int
    doc_count: int
    case_count: int
    avg_score: float | None


class DashboardOut(BaseModel):
    totals: DashboardTotals
    projects: list[ProjectSummary]
    type_dist: dict[str, int]
    priority_dist: dict[str, int]


@router.get("/dashboard", response_model=DashboardOut)
def get_dashboard(db: Session = Depends(get_db)):
    projects = (
        db.query(models.Project).order_by(models.Project.id.asc()).all()
    )

    doc_rows = (
        db.query(models.Document.project_id, func.count(models.Document.id))
        .group_by(models.Document.project_id)
        .all()
    )
    doc_map = {pid: n for pid, n in doc_rows}

    case_rows = (
        db.query(
            models.TestCase.project_id,
            func.count(models.TestCase.id),
            func.avg(models.TestCase.score),
        )
        .group_by(models.TestCase.project_id)
        .all()
    )
    case_map = {
        pid: (int(n or 0), round(float(avg), 2) if avg is not None else None)
        for pid, n, avg in case_rows
    }

    type_rows = (
        db.query(models.TestCase.case_type, func.count(models.TestCase.id))
        .group_by(models.TestCase.case_type)
        .all()
    )
    priority_rows = (
        db.query(models.TestCase.priority, func.count(models.TestCase.id))
        .group_by(models.TestCase.priority)
        .all()
    )

    summaries: list[ProjectSummary] = []
    for p in projects:
        case_count, avg_score = case_map.get(p.id, (0, None))
        summaries.append(
            ProjectSummary(
                id=p.id,
                name=p.name,
                doc_count=int(doc_map.get(p.id, 0)),
                case_count=case_count,
                avg_score=avg_score,
            )
        )

    overall_avg = (
        db.query(func.avg(models.TestCase.score))
        .filter(models.TestCase.score.isnot(None))
        .scalar()
    )
    totals = DashboardTotals(
        project_count=len(projects),
        doc_count=sum(doc_map.values()),
        case_count=sum(n for n, _ in case_map.values()),
        avg_score=round(float(overall_avg), 2) if overall_avg is not None else None,
    )

    return DashboardOut(
        totals=totals,
        projects=summaries,
        type_dist={k or "unknown": v for k, v in type_rows},
        priority_dist={k or "unknown": v for k, v in priority_rows},
    )
