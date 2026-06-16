from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.core.config import settings
from app.services.embeddings import create_embeddings


class VectorStoreService:
    def __init__(self) -> None:
        self.embeddings = create_embeddings()
        self.vector_store = Chroma(
            collection_name=settings.chroma_collection,
            embedding_function=self.embeddings,
            persist_directory=str(settings.chroma_dir),
        )

    def add_documents(self, documents: list[Document], ids: list[str]) -> None:
        if len(documents) != len(ids):
            raise ValueError("documents and ids must have the same length")
        if not documents:
            raise ValueError("No document chunks to index")
        self.vector_store.add_documents(documents=documents, ids=ids)

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        return self.vector_store.similarity_search_with_score(query=query, k=k)

    def delete(self, ids: list[str]) -> None:
        if ids:
            self.vector_store.delete(ids=ids)


vector_store_service = VectorStoreService()

