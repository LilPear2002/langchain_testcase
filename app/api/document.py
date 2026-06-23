from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session

from app.db import models
from app.db.session import get_db
from app.schemas.common import Msg
from app.schemas.document import DocumentOut
from app.services.document_service import delete_document, save_and_index_upload

router = APIRouter(
    prefix="/projects/{project_id}/documents", tags=["documents"]
)


@router.post("", response_model=DocumentOut)
async def upload_document(
    project_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    return await save_and_index_upload(db, project_id, file)


@router.get("", response_model=list[DocumentOut])
def list_documents(project_id: int, db: Session = Depends(get_db)):
    return (
        db.query(models.Document)
        .filter(models.Document.project_id == project_id)
        .order_by(models.Document.id.desc())
        .all()
    )


@router.delete("/{doc_id}", response_model=Msg)
def remove_document(
    project_id: int, doc_id: int, db: Session = Depends(get_db)
):
    delete_document(db, doc_id)
    return Msg(message="ok")
