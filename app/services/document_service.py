import hashlib
import logging
from pathlib import Path

from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import settings
from app.db import models
from app.rag.loader import SUPPORTED_EXTS, load_file_as_documents
from app.rag.splitter import get_splitter
from app.services import vector_store

logger = logging.getLogger(__name__)


async def save_and_index_upload(
    db: Session, project_id: int, upload: UploadFile
) -> models.Document:
    project = db.get(models.Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")

    ext = Path(upload.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTS:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    content = await upload.read()
    md5 = hashlib.md5(content).hexdigest()

    exists = (
        db.query(models.Document)
        .filter(
            models.Document.project_id == project_id,
            models.Document.md5 == md5,
        )
        .first()
    )
    if exists:
        return exists

    project_dir = Path(settings.upload_dir) / f"project_{project_id}"
    project_dir.mkdir(parents=True, exist_ok=True)
    save_path = project_dir / f"{md5}{ext}"
    save_path.write_bytes(content)

    doc = models.Document(
        project_id=project_id,
        filename=upload.filename or save_path.name,
        file_path=str(save_path),
        mime_type=upload.content_type,
        size=len(content),
        md5=md5,
        status="pending",
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    try:
        raw_docs = load_file_as_documents(str(save_path))
        chunks = get_splitter().split_documents(raw_docs)
        for c in chunks:
            c.metadata["doc_name"] = doc.filename
        vector_store.add_documents(
            chunks, project_id=project_id, doc_id=doc.id
        )
        doc.chunk_count = len(chunks)
        doc.status = "indexed"
        db.commit()
    except Exception as e:
        logger.exception("Failed to index document")
        doc.status = "failed"
        db.commit()
        raise HTTPException(500, f"Index failed: {e}")

    return doc


def delete_document(db: Session, doc_id: int) -> None:
    doc = db.get(models.Document, doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    try:
        vector_store.delete_by_doc(doc_id)
    except Exception:
        logger.warning("vector delete failed", exc_info=True)
    try:
        Path(doc.file_path).unlink(missing_ok=True)
    except Exception:
        pass
    db.delete(doc)
    db.commit()
