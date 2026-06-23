import os
from contextlib import contextmanager
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

from app.config import settings

TEXT_EXTS = {".txt", ".md"}
MINERU_EXTS = {".pdf", ".docx", ".xlsx", ".xls"}
SUPPORTED_EXTS = TEXT_EXTS | MINERU_EXTS


def load_file_as_documents(file_path: str) -> list[Document]:
    ext = Path(file_path).suffix.lower()
    if ext in TEXT_EXTS:
        return TextLoader(file_path, encoding="utf-8").load()
    if ext in MINERU_EXTS:
        return _load_with_mineru(file_path)
    raise ValueError(f"Unsupported file type: {ext}")


@contextmanager
def _bypass_proxy(domains: str):
    """临时把指定域名追加进 NO_PROXY，让 MinerU 直连国内 CDN，退出后恢复。"""
    keys = ("NO_PROXY", "no_proxy")
    saved = {k: os.environ.get(k) for k in keys}
    extra = [d.strip() for d in domains.split(",") if d.strip()]
    for k in keys:
        cur = [d for d in (saved[k] or "").split(",") if d]
        os.environ[k] = ",".join(dict.fromkeys(cur + extra))
    try:
        yield
    finally:
        for k in keys:
            if saved[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = saved[k]


def _load_with_mineru(file_path: str) -> list[Document]:
    from langchain_mineru import MinerULoader

    kwargs = {
        "source": file_path,
        "mode": settings.mineru_mode,
        "language": settings.mineru_language,
        "timeout": settings.mineru_timeout,
    }
    if settings.mineru_mode == "precision" and settings.mineru_token:
        kwargs["token"] = settings.mineru_token
    with _bypass_proxy(settings.mineru_no_proxy):
        return MinerULoader(**kwargs).load()
