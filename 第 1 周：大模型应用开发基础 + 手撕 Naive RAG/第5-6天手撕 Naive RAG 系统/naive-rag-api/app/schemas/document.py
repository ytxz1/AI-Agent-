from pydantic import BaseModel


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    saved_filename: str
    content_type: str | None = None
    file_size: int
    chunk_count: int
    chunk_ids: list[str]
    created_at: str


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    status: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]


class DocumentDetailResponse(BaseModel):
    document: DocumentInfo


class DeleteDocumentResponse(BaseModel):
    document_id: str
    deleted_chunks: int
    status: str

