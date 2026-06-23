from collections.abc import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.db import models
from app.db.session import SessionLocal


def _msg_to_role(m: BaseMessage) -> str:
    if isinstance(m, HumanMessage):
        return "human"
    if isinstance(m, AIMessage):
        return "ai"
    if isinstance(m, SystemMessage):
        return "system"
    return m.type


def _row_to_msg(role: str, content: str) -> BaseMessage:
    if role == "human":
        return HumanMessage(content=content)
    if role == "ai":
        return AIMessage(content=content)
    if role == "system":
        return SystemMessage(content=content)
    return HumanMessage(content=content)


class PostgresChatMessageHistory(BaseChatMessageHistory):
    """将对话历史持久化到 PostgreSQL 的 chat_messages 表。"""

    def __init__(self, session_id: int):
        self.session_id = session_id

    @property
    def messages(self) -> list[BaseMessage]:
        with SessionLocal() as db:
            rows = (
                db.query(models.ChatMessage)
                .filter(models.ChatMessage.session_id == self.session_id)
                .order_by(models.ChatMessage.id.asc())
                .all()
            )
            return [_row_to_msg(r.role, r.content) for r in rows]

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        if not messages:
            return
        with SessionLocal() as db:
            for m in messages:
                content = m.content if isinstance(m.content, str) else str(m.content)
                db.add(
                    models.ChatMessage(
                        session_id=self.session_id,
                        role=_msg_to_role(m),
                        content=content,
                    )
                )
            db.commit()

    def clear(self) -> None:
        with SessionLocal() as db:
            db.query(models.ChatMessage).filter(
                models.ChatMessage.session_id == self.session_id
            ).delete()
            db.commit()
