import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db import models
from app.db.session import get_db
from app.rag.rag_chain import build_rag_chain
from app.schemas.chat import (
    ChatMessageOut,
    ChatRequest,
    ChatResponse,
    ChatSessionCreate,
    ChatSessionOut,
    ChatSessionUpdate,
)
from app.schemas.common import Msg

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


# ---------- 会话 CRUD ----------

@router.post(
    "/projects/{project_id}/chat/sessions", response_model=ChatSessionOut
)
def create_session(
    project_id: int,
    body: ChatSessionCreate,
    db: Session = Depends(get_db),
):
    project = db.get(models.Project, project_id)
    if not project:
        raise HTTPException(404, "Project not found")
    obj = models.ChatSession(project_id=project_id, title=body.title)
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


@router.get(
    "/projects/{project_id}/chat/sessions",
    response_model=list[ChatSessionOut],
)
def list_sessions(project_id: int, db: Session = Depends(get_db)):
    return (
        db.query(models.ChatSession)
        .filter(models.ChatSession.project_id == project_id)
        .order_by(models.ChatSession.id.desc())
        .all()
    )


@router.patch(
    "/chat/sessions/{session_id}", response_model=ChatSessionOut
)
def update_session(
    session_id: int,
    body: ChatSessionUpdate,
    db: Session = Depends(get_db),
):
    obj = db.get(models.ChatSession, session_id)
    if not obj:
        raise HTTPException(404, "Session not found")
    if body.title is not None:
        obj.title = body.title
    db.commit()
    db.refresh(obj)
    return obj


@router.delete("/chat/sessions/{session_id}", response_model=Msg)
def delete_session(session_id: int, db: Session = Depends(get_db)):
    obj = db.get(models.ChatSession, session_id)
    if not obj:
        raise HTTPException(404, "Session not found")
    db.delete(obj)
    db.commit()
    return Msg(message="ok")


@router.get(
    "/chat/sessions/{session_id}/messages",
    response_model=list[ChatMessageOut],
)
def list_messages(session_id: int, db: Session = Depends(get_db)):
    session = db.get(models.ChatSession, session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return (
        db.query(models.ChatMessage)
        .filter(models.ChatMessage.session_id == session_id)
        .order_by(models.ChatMessage.id.asc())
        .all()
    )


# ---------- 对话接口 ----------

def _get_session_or_404(db: Session, session_id: int) -> models.ChatSession:
    obj = db.get(models.ChatSession, session_id)
    if not obj:
        raise HTTPException(404, "Session not found")
    return obj


@router.post(
    "/chat/sessions/{session_id}/messages", response_model=ChatResponse
)
def chat(
    session_id: int,
    body: ChatRequest,
    db: Session = Depends(get_db),
):
    session = _get_session_or_404(db, session_id)
    chain = build_rag_chain(project_id=session.project_id)
    answer = chain.invoke(
        {"input": body.input},
        config={"configurable": {"session_id": str(session_id)}},
    )
    return ChatResponse(answer=answer)


@router.post("/chat/sessions/{session_id}/stream")
def chat_stream(
    session_id: int,
    body: ChatRequest,
    db: Session = Depends(get_db),
):
    session = _get_session_or_404(db, session_id)
    chain = build_rag_chain(project_id=session.project_id)

    def event_gen():
        try:
            for chunk in chain.stream(
                {"input": body.input},
                config={"configurable": {"session_id": str(session_id)}},
            ):
                if chunk:
                    payload = json.dumps(
                        {"type": "chunk", "content": chunk},
                        ensure_ascii=False,
                    )
                    yield f"data: {payload}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            logger.exception("chat stream error")
            err = json.dumps(
                {"type": "error", "message": str(e)}, ensure_ascii=False
            )
            yield f"data: {err}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
