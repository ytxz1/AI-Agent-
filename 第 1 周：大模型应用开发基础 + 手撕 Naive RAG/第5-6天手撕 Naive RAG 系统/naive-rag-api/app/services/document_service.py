import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.core.config import settings
from app.core.time import utc_now_iso
from app.schemas.document import (
    DeleteDocumentResponse,
    DocumentDetailResponse,
    DocumentInfo,
    DocumentUploadResponse,
)
from app.services.loader import SUPPORTED_SUFFIXES, load_file
from app.services.metadata_store import metadata_store
from app.services.splitter import split_documents
from app.services.vector_store import vector_store_service


class DocumentService:
    def list_documents(self) -> list[DocumentInfo]:
        return [DocumentInfo(**record) for record in metadata_store.list_documents()]

    def get_document(self, document_id: str) -> DocumentDetailResponse:
        record = metadata_store.get_document(document_id)
        if record is None:
            raise ValueError(f"Document not found: {document_id}")
        return DocumentDetailResponse(document=DocumentInfo(**record))

    def upload_document(self, file: UploadFile) -> DocumentUploadResponse:
        original_filename = file.filename or "uploaded_file"
        suffix = Path(original_filename).suffix.lower()

        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file type: {suffix}")

        document_id = str(uuid4())
        saved_filename = f"{document_id}{suffix}"
        target_path = settings.upload_dir / saved_filename
        file_size = self._save_upload_file(file=file, target_path=target_path)

        raw_documents = load_file(target_path)
        for document in raw_documents:
            document.metadata.update(
                {
                    "document_id": document_id,
                    "filename": original_filename,
                    "saved_filename": saved_filename,
                    "source": str(target_path),
                }
            )

        chunks = split_documents(raw_documents)
        chunk_ids = []
        for index, chunk in enumerate(chunks):
            chunk_id = f"{document_id}:chunk:{index}"
            chunk.metadata.update(
                {
                    "chunk_id": chunk_id,
                    "chunk_index": index,
                }
            )
            chunk_ids.append(chunk_id)

        vector_store_service.add_documents(documents=chunks, ids=chunk_ids)

        metadata_store.add_document(
            {
                "document_id": document_id,
                "filename": original_filename,
                "saved_filename": saved_filename,
                "content_type": file.content_type,
                "file_size": file_size,
                "chunk_count": len(chunks),
                "chunk_ids": chunk_ids,
                "created_at": utc_now_iso(),
            }
        )

        return DocumentUploadResponse(
            document_id=document_id,
            filename=original_filename,
            chunk_count=len(chunks),
            status="indexed",
        )

    def delete_document(self, document_id: str) -> DeleteDocumentResponse:
        record = metadata_store.remove_document(document_id)
        if record is None:
            raise ValueError(f"Document not found: {document_id}")

        chunk_ids = record.get("chunk_ids", [])
        vector_store_service.delete(ids=chunk_ids)

        saved_filename = record.get("saved_filename")
        if saved_filename:
            path = settings.upload_dir / saved_filename
            if path.exists():
                path.unlink()

        return DeleteDocumentResponse(
            document_id=document_id,
            deleted_chunks=len(chunk_ids),
            status="deleted",
        )

    def _save_upload_file(self, file: UploadFile, target_path: Path) -> int:
        total = 0
        with target_path.open("wb") as buffer:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > settings.max_file_size_bytes:
                    buffer.close()
                    target_path.unlink(missing_ok=True)
                    raise ValueError(
                        f"File is too large. Max size is {settings.max_file_size_mb} MB"
                    )
                buffer.write(chunk)

        if total == 0:
            target_path.unlink(missing_ok=True)
            raise ValueError("Uploaded file is empty")

        file.file.seek(0)
        return total


document_service = DocumentService()

