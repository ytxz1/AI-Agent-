from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {suffix}")

    if suffix == ".pdf":
        return PyPDFLoader(str(path)).load()

    text = path.read_text(encoding="utf-8")
    return [Document(page_content=text, metadata={"source": str(path)})]

