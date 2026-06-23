from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

from app.core.llm import get_generation_llm
from app.rag.chat_history import PostgresChatMessageHistory
from app.services.vector_store import similarity_search

SYSTEM_PROMPT = (
    "你是一个专业的需求分析助手。请基于下方『参考资料』解答用户的问题，"
    "如果资料不足以回答，请明确告知用户而不是编造内容。\n"
    "回答风格：专业、简洁、分点清晰。\n\n"
    "参考资料：\n{context}"
)


def _format_docs(docs: list[Document]) -> str:
    if not docs:
        return "（无相关参考资料）"
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("doc_name") or d.metadata.get("source") or "未知"
        parts.append(f"[片段 {i} | 来源: {src}]\n{d.page_content}")
    return "\n\n".join(parts)


def build_rag_chain(project_id: int, k: int = 4) -> Runnable:
    """构建带历史记忆的 RAG 对话链。

    invoke 入参: {"input": "用户问题"}
    config 需传: {"configurable": {"session_id": 会话id}}
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    def retrieve(value: dict) -> str:
        docs = similarity_search(value["input"], project_id=project_id, k=k)
        return _format_docs(docs)

    chain = (
        RunnablePassthrough.assign(context=RunnableLambda(retrieve))
        | prompt
        | get_generation_llm()
        | StrOutputParser()
    )

    return RunnableWithMessageHistory(
        chain,
        lambda session_id: PostgresChatMessageHistory(int(session_id)),
        input_messages_key="input",
        history_messages_key="history",
    )
