import os
from functools import lru_cache

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings

from app.config import settings

if settings.dashscope_api_key:
    os.environ.setdefault("DASHSCOPE_API_KEY", settings.dashscope_api_key)


@lru_cache(maxsize=1)
def get_generation_llm() -> ChatTongyi:
    return ChatTongyi(model=settings.generation_model)


@lru_cache(maxsize=1)
def get_judge_llm() -> ChatTongyi:
    return ChatTongyi(model=settings.judge_model)


@lru_cache(maxsize=1)
def get_embeddings() -> DashScopeEmbeddings:
    return DashScopeEmbeddings(model=settings.embedding_model)
