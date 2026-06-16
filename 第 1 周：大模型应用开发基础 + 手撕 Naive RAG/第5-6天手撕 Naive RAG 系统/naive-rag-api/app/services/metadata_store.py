import json

from app.core.config import settings


class MetadataStore:
    def __init__(self) -> None:
        settings.ensure_dirs()
        if not settings.registry_path.exists():
            self._write({"documents": []})

    def list_documents(self) -> list[dict]:
        return self._read()["documents"]

    def get_document(self, document_id: str) -> dict | None:
        for document in self.list_documents():
            if document["document_id"] == document_id:
                return document
        return None

    def add_document(self, record: dict) -> None:
        registry = self._read()
        registry["documents"].append(record)
        self._write(registry)

    def remove_document(self, document_id: str) -> dict | None:
        registry = self._read()
        kept = []
        removed = None

        for document in registry["documents"]:
            if document["document_id"] == document_id:
                removed = document
            else:
                kept.append(document)

        if removed is None:
            return None

        registry["documents"] = kept
        self._write(registry)
        return removed

    def _read(self) -> dict:
        return json.loads(settings.registry_path.read_text(encoding="utf-8"))

    def _write(self, registry: dict) -> None:
        settings.registry_path.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


metadata_store = MetadataStore()

