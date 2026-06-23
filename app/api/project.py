from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import models
from app.db.session import get_db
from app.schemas.common import Msg
from app.schemas.project import ProjectCreate, ProjectOut, ProjectUpdate

router = APIRouter(prefix="/projects", tags=["projects"])


@router.post("", response_model=ProjectOut)
def create_project(body: ProjectCreate, db: Session = Depends(get_db)):
    obj = models.Project(name=body.name, description=body.description)
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


@router.get("", response_model=list[ProjectOut])
def list_projects(db: Session = Depends(get_db)):
    return (
        db.query(models.Project)
        .order_by(models.Project.id.desc())
        .all()
    )


@router.get("/{project_id}", response_model=ProjectOut)
def get_project(project_id: int, db: Session = Depends(get_db)):
    obj = db.get(models.Project, project_id)
    if not obj:
        raise HTTPException(404, "Project not found")
    return obj


@router.patch("/{project_id}", response_model=ProjectOut)
def update_project(
    project_id: int, body: ProjectUpdate, db: Session = Depends(get_db)
):
    obj = db.get(models.Project, project_id)
    if not obj:
        raise HTTPException(404, "Project not found")
    if body.name is not None:
        obj.name = body.name
    if body.description is not None:
        obj.description = body.description
    db.commit()
    db.refresh(obj)
    return obj


@router.delete("/{project_id}", response_model=Msg)
def delete_project(project_id: int, db: Session = Depends(get_db)):
    obj = db.get(models.Project, project_id)
    if not obj:
        raise HTTPException(404, "Project not found")
    db.delete(obj)
    db.commit()
    return Msg(message="ok")
