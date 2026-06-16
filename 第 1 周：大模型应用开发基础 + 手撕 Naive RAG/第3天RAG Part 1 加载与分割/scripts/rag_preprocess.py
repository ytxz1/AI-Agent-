"""
RAG document loading and chunking demo.

This script is intentionally small and dependency-light so you can read it
end-to-end while learning RAG Part 1: loading and splitting.

Supported inputs:
- .md / .markdown
- .txt
- .json files exported by tools such as MinerU or Docling, if they contain
  common text fields like markdown, text, content, page_content, or md_content
- .pdf, when optional dependency pypdf is installed

Example:
    python scripts/rag_preprocess.py --input examples/sample.md --output outputs/chunks.jsonl

    python scripts/rag_preprocess.py --input examples --recursive --output outputs/chunks.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".json", ".pdf"}


@dataclass
class Document:
    """A unified document object before or after splitting."""

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkReport:
    """Quality summary for generated chunks."""

    total_chunks: int
    empty_chunks: int
    too_short_chunks: int
    too_long_chunks: int
    missing_source: int
    missing_chunk_id: int
    min_length: int
    max_length: int
    avg_length: float


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_text(text: str) -> str:
    """Clean common whitespace noise without destroying useful structure."""

    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def stable_file_id(path: Path) -> str:
    """Create a readable ID stem from a file name."""

    stem = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", path.stem)
    return stem.strip("_") or "document"


def stable_text_hash(text: str, length: int = 10) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def base_metadata(path: Path, parser: str) -> dict[str, Any]:
    return {
        "source": str(path),
        "file_name": path.name,
        "file_type": path.suffix.lower().lstrip("."),
        "parser": parser,
        "created_at": utc_now_iso(),
    }


def iter_input_files(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    pattern = "**/*" if recursive else "*"
    files = [
        path
        for path in input_path.glob(pattern)
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return sorted(files)


def load_txt(path: Path) -> list[Document]:
    text = normalize_text(path.read_text(encoding="utf-8"))
    return [Document(text, base_metadata(path, "raw_text"))]


def load_markdown_raw(path: Path) -> list[Document]:
    text = normalize_text(path.read_text(encoding="utf-8"))
    return [Document(text, base_metadata(path, "raw_markdown"))]


def parse_markdown_sections(path: Path) -> list[Document]:
    """Split Markdown into heading-aware section documents.

    Headings inside fenced code blocks are ignored. Each returned section keeps
    a heading_path metadata field, which is very useful for RAG retrieval.
    """

    text = path.read_text(encoding="utf-8")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    docs: list[Document] = []
    heading_stack: list[str] = []
    current_lines: list[str] = []
    current_heading_path: list[str] = []
    in_fence = False

    def flush() -> None:
        content = normalize_text("\n".join(current_lines))
        if not content:
            return
        metadata = base_metadata(path, "markdown_heading")
        metadata["heading_path"] = " > ".join(current_heading_path)
        metadata["section"] = current_heading_path[-1] if current_heading_path else ""
        docs.append(Document(content, metadata))

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            current_lines.append(line)
            continue

        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match and not in_fence:
            flush()
            current_lines = [line]

            level = len(match.group(1))
            title = match.group(2).strip()
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(title)
            current_heading_path = heading_stack.copy()
            continue

        current_lines.append(line)

    flush()

    if docs:
        return docs

    return load_markdown_raw(path)


def extract_text_from_json_value(value: Any, out: list[str]) -> None:
    """Extract likely text fields from nested JSON exported by parsers.

    MinerU and Docling can export rich JSON structures. Their exact schema can
    vary by version and command, so this teaching script uses a conservative
    recursive extractor for common text-bearing keys.
    """

    text_keys = {
        "text",
        "content",
        "page_content",
        "markdown",
        "md",
        "md_content",
        "body",
        "caption",
        "html",
    }

    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in text_keys and isinstance(item, str):
                cleaned = normalize_text(item)
                if cleaned:
                    out.append(cleaned)
            else:
                extract_text_from_json_value(item, out)
        return

    if isinstance(value, list):
        for item in value:
            extract_text_from_json_value(item, out)


def load_parser_json(path: Path) -> list[Document]:
    data = json.loads(path.read_text(encoding="utf-8"))
    texts: list[str] = []
    extract_text_from_json_value(data, texts)

    if not texts:
        raise ValueError(f"No text-like fields found in JSON: {path}")

    docs: list[Document] = []
    for index, text in enumerate(texts):
        metadata = base_metadata(path, "generic_parser_json")
        metadata["json_text_index"] = index
        docs.append(Document(text, metadata))
    return docs


def load_pdf_with_pypdf(path: Path) -> list[Document]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF loading requires optional dependency pypdf. "
            "Install it with: pip install -r requirements.txt. "
            "For complex PDFs, prefer converting with MinerU or Docling first."
        ) from exc

    reader = PdfReader(str(path))
    docs: list[Document] = []

    for page_index, page in enumerate(reader.pages):
        text = normalize_text(page.extract_text() or "")
        if not text:
            continue
        metadata = base_metadata(path, "pypdf")
        metadata["page"] = page_index + 1
        metadata["total_pages"] = len(reader.pages)
        docs.append(Document(text, metadata))

    return docs


def load_documents(path: Path, markdown_mode: str) -> list[Document]:
    suffix = path.suffix.lower()

    if suffix in {".md", ".markdown"}:
        if markdown_mode == "heading":
            return parse_markdown_sections(path)
        return load_markdown_raw(path)

    if suffix == ".txt":
        return load_txt(path)

    if suffix == ".json":
        return load_parser_json(path)

    if suffix == ".pdf":
        return load_pdf_with_pypdf(path)

    raise ValueError(f"Unsupported file type: {path}")


def split_text_recursive(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str] | None = None,
) -> list[str]:
    """A small recursive character splitter for learning purposes."""

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    separators = separators or ["\n\n", "\n", "。", "！", "？", ".", "!", "?", "，", ",", " ", ""]
    text = normalize_text(text)

    def hard_split(part: str) -> list[str]:
        step = chunk_size - chunk_overlap
        return [part[i : i + chunk_size] for i in range(0, len(part), step)]

    def merge_pieces(pieces: list[str], separator: str) -> list[str]:
        chunks: list[str] = []
        current = ""

        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue

            candidate = piece if not current else current + separator + piece
            if len(candidate) <= chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(current)

            if len(piece) > chunk_size:
                chunks.extend(split_with_separators(piece, separators[1:]))
                current = ""
            else:
                current = piece

        if current:
            chunks.append(current)

        return chunks

    def split_with_separators(part: str, seps: list[str]) -> list[str]:
        part = part.strip()
        if not part:
            return []
        if len(part) <= chunk_size:
            return [part]
        if not seps:
            return hard_split(part)

        separator = seps[0]
        if separator == "":
            return hard_split(part)

        if separator not in part:
            return split_with_separators(part, seps[1:])

        pieces = part.split(separator)
        return merge_pieces(pieces, separator)

    chunks = split_with_separators(text, separators)

    if chunk_overlap <= 0 or len(chunks) <= 1:
        return chunks

    with_overlap: list[str] = []
    for index, chunk in enumerate(chunks):
        if index == 0:
            with_overlap.append(chunk)
            continue
        prefix = chunks[index - 1][-chunk_overlap:].strip()
        combined = normalize_text(prefix + "\n" + chunk)
        with_overlap.append(combined[: chunk_size + chunk_overlap])
    return with_overlap


def render_chunk_text(doc: Document, include_metadata_header: bool) -> str:
    text = doc.page_content.strip()
    if not include_metadata_header:
        return text

    header_lines: list[str] = []
    heading_path = doc.metadata.get("heading_path")
    page = doc.metadata.get("page")

    if heading_path:
        header_lines.append(f"标题路径：{heading_path}")
    if page:
        header_lines.append(f"页码：{page}")

    if not header_lines:
        return text

    return "\n".join(header_lines) + "\n\n" + text


def split_document(
    doc: Document,
    chunk_size: int,
    chunk_overlap: int,
    include_metadata_header: bool,
) -> list[Document]:
    source_text = render_chunk_text(doc, include_metadata_header)
    pieces = split_text_recursive(source_text, chunk_size, chunk_overlap)
    chunks: list[Document] = []

    source = Path(str(doc.metadata.get("source", "document")))
    file_id = stable_file_id(source)

    for index, piece in enumerate(pieces):
        metadata = dict(doc.metadata)
        metadata["chunk_index"] = index
        metadata["chunk_id"] = f"{file_id}_{index:04d}_{stable_text_hash(piece)}"
        metadata["chunk_length"] = len(piece)
        chunks.append(Document(piece, metadata))

    return chunks


def split_documents(
    docs: list[Document],
    chunk_size: int,
    chunk_overlap: int,
    include_metadata_header: bool,
) -> list[Document]:
    chunks: list[Document] = []
    global_index = 0

    for doc in docs:
        doc_chunks = split_document(doc, chunk_size, chunk_overlap, include_metadata_header)
        for chunk in doc_chunks:
            chunk.metadata["global_chunk_index"] = global_index
            global_index += 1
            chunks.append(chunk)

    return chunks


def quality_report(chunks: list[Document], min_chars: int, max_chars: int) -> ChunkReport:
    lengths = [len(chunk.page_content) for chunk in chunks]
    return ChunkReport(
        total_chunks=len(chunks),
        empty_chunks=sum(1 for chunk in chunks if not chunk.page_content.strip()),
        too_short_chunks=sum(1 for length in lengths if length < min_chars),
        too_long_chunks=sum(1 for length in lengths if length > max_chars),
        missing_source=sum(1 for chunk in chunks if not chunk.metadata.get("source")),
        missing_chunk_id=sum(1 for chunk in chunks if not chunk.metadata.get("chunk_id")),
        min_length=min(lengths) if lengths else 0,
        max_length=max(lengths) if lengths else 0,
        avg_length=round(sum(lengths) / len(lengths), 2) if lengths else 0,
    )


def write_jsonl(chunks: Iterable[Document], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def print_preview(chunks: list[Document], limit: int) -> None:
    for chunk in chunks[:limit]:
        print("=" * 88)
        print(json.dumps(chunk.metadata, ensure_ascii=False, indent=2))
        print("-" * 88)
        print(chunk.page_content[:500])
        if len(chunk.page_content) > 500:
            print("...")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load and split documents for a Naive RAG pipeline.")
    parser.add_argument("--input", help="Input file or directory. Defaults to examples/sample.md when omitted.")
    parser.add_argument("--output", default="outputs/chunks.jsonl", help="Output JSONL file.")
    parser.add_argument("--recursive", action="store_true", help="Read supported files recursively when input is a directory.")
    parser.add_argument("--chunk-size", type=int, default=800, help="Target max chunk size in characters.")
    parser.add_argument("--chunk-overlap", type=int, default=120, help="Overlap size in characters.")
    parser.add_argument(
        "--markdown-mode",
        choices=["heading", "raw"],
        default="heading",
        help="Use heading-aware Markdown loading or raw Markdown loading.",
    )
    parser.add_argument(
        "--include-metadata-header",
        action="store_true",
        help="Render heading path and page number into chunk text before splitting.",
    )
    parser.add_argument("--preview", type=int, default=3, help="Number of chunks to print.")
    parser.add_argument("--min-chars", type=int, default=50, help="Warning threshold for very short chunks.")
    parser.add_argument("--max-chars", type=int, default=1400, help="Warning threshold for very long chunks.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent

    if args.input is None:
        args.input = str(project_dir / "examples" / "sample.md")
        args.output = str(project_dir / "outputs" / "chunks.jsonl")
        args.include_metadata_header = True
        print("No --input provided. Running the built-in sample:")
        print(f"  input : {args.input}")
        print(f"  output: {args.output}")
        print("Tip: pass --input your_file.md to process your own document.")
        print()

    input_path = Path(args.input)
    output_path = Path(args.output)

    files = iter_input_files(input_path, args.recursive)
    if not files:
        raise FileNotFoundError(f"No supported files found: {input_path}")

    all_docs: list[Document] = []
    for file_path in files:
        docs = load_documents(file_path, args.markdown_mode)
        all_docs.extend(docs)

    chunks = split_documents(
        all_docs,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        include_metadata_header=args.include_metadata_header,
    )

    report = quality_report(chunks, min_chars=args.min_chars, max_chars=args.max_chars)
    write_jsonl(chunks, output_path)

    print("Loaded files:", len(files))
    print("Loaded documents:", len(all_docs))
    print("Wrote chunks:", output_path)
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    print_preview(chunks, args.preview)


if __name__ == "__main__":
    main()
