from functools import lru_cache

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


@lru_cache(maxsize=1)
def get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
        length_function=len,
    )
