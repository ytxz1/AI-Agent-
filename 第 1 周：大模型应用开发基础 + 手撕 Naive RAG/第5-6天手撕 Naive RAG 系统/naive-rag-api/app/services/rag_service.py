from app.core.config import settings
from app.schemas.chat import ChatResponse, RetrievedChunk, SourceChunk
from app.services.chat_model import create_chat_model
from app.services.metadata_store import metadata_store
from app.services.vector_store import vector_store_service


class RagService:
    def __init__(self) -> None:
        self.chat_model = None

    def answer(self, question: str, top_k: int | None = None) -> ChatResponse:
        if not metadata_store.list_documents():
            return ChatResponse(
                answer="当前知识库为空，请先上传文档。",
                sources=[],
                retrieved_chunks=[],
            )

        k = top_k or settings.default_top_k
        docs_with_scores = vector_store_service.search(query=question, k=k)
        documents = [document for document, _score in docs_with_scores]

        if not documents:
            return ChatResponse(
                answer="根据已上传文档，我不知道。",
                sources=[],
                retrieved_chunks=[],
            )

        answer = self._chat_model().generate(question=question, documents=documents)
        return ChatResponse(
            answer=answer,
            sources=self._build_sources(docs_with_scores),
            retrieved_chunks=self._build_retrieved_chunks(docs_with_scores),
        )

    def _chat_model(self):
        if self.chat_model is None:
            self.chat_model = create_chat_model()
        return self.chat_model

    def _build_sources(self, docs_with_scores) -> list[SourceChunk]:
        sources = []
        for document, score in docs_with_scores:
            metadata = document.metadata
            content_preview = " ".join(document.page_content.split())[:240]
            sources.append(
                SourceChunk(
                    document_id=str(metadata.get("document_id", "")),
                    filename=str(metadata.get("filename", "")),
                    chunk_id=str(metadata.get("chunk_id", "")),
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    content_preview=content_preview,
                    score=float(score) if score is not None else None,
                )
            )
        return sources

    def _build_retrieved_chunks(self, docs_with_scores) -> list[RetrievedChunk]:
        chunks = []
        for document, score in docs_with_scores:
            chunks.append(
                RetrievedChunk(
                    content=document.page_content,
                    metadata=dict(document.metadata),
                    score=float(score) if score is not None else None,
                )
            )
        return chunks


rag_service = RagService()
