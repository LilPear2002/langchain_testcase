from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import settings
from app.core.llm import get_embeddings


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_dir,
    )


def add_documents(
    docs: list[Document], project_id: int, doc_id: int
) -> list[str]:
    for d in docs:
        d.metadata.setdefault("source", "")
        d.metadata["project_id"] = project_id
        d.metadata["doc_id"] = doc_id
    ids = [f"p{project_id}_d{doc_id}_c{i}" for i in range(len(docs))]
    vs = get_vector_store()
    vs.add_documents(documents=docs, ids=ids)
    return ids


def delete_by_doc(doc_id: int) -> None:
    vs = get_vector_store()
    vs._collection.delete(where={"doc_id": doc_id})


def similarity_search(
    query: str, project_id: int, k: int = 4
) -> list[Document]:
    vs = get_vector_store()
    return vs.similarity_search(
        query, k=k, filter={"project_id": project_id}
    )
