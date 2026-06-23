from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    database_url: str = (
        "postgresql+psycopg2://postgres:123456@localhost:5432/testcase_agent"
    )
    dashscope_api_key: str = ""

    generation_model: str = "qwen3-max"
    judge_model: str = "qwen-plus"
    embedding_model: str = "text-embedding-v4"

    upload_dir: str = "./data/uploads"
    chroma_dir: str = "./data/chroma"
    chroma_collection: str = "requirements"

    chunk_size: int = 800
    chunk_overlap: int = 100

    # MinerU 文档解析（pdf/word/excel）
    mineru_mode: str = "flash"
    mineru_token: str = ""
    mineru_language: str = "ch"
    mineru_timeout: int = 600
    # 这些域名直连、绕过本地代理（国内 CDN 经代理常 SSL 握手失败）
    mineru_no_proxy: str = (
        "mineru.net,openxlab.org.cn,opendatalab.com,aliyuncs.com"
    )

    api_prefix: str = "/api"

    # LangSmith 可观测性
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "testcase-agent"
    langsmith_endpoint: str = "https://api.smith.langchain.com"


settings = Settings()

Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
Path(settings.chroma_dir).mkdir(parents=True, exist_ok=True)


def _configure_langsmith() -> None:
    """把 LangSmith 相关配置写入 os.environ，供 langchain/langgraph 自动读取。"""
    import os

    if settings.langsmith_tracing and settings.langsmith_api_key:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint


_configure_langsmith()
