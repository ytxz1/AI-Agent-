from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


PDF_SUFFIXES = {".pdf"}


@dataclass
class NormalizedBlock:
    document_id: str
    source_file: str
    parser: str
    block_id: str
    block_type: str
    text: str = ""
    html: str | None = None
    image_path: str | None = None
    page_idx: int | None = None
    bbox: Any | None = None
    parent_id: str | None = None
    section_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RagChunk:
    chunk_id: str
    document_id: str
    source_file: str
    parser: str
    chunk_type: str
    text: str
    source_blocks: list[str]
    page_range: list[int | None]
    section_path: list[str] = field(default_factory=list)
    html: str | None = None
    image_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dataclass_to_dict(item: Any) -> dict[str, Any]:
    return asdict(item)


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def stable_id(*parts: str, length: int = 12) -> str:
    raw = "|".join(parts).encode("utf-8", errors="ignore")
    return hashlib.sha1(raw).hexdigest()[:length]


def document_id_from_path(path: Path) -> str:
    return path.stem


def iter_input_pdfs(input_path: Path) -> list[Path]:
    if input_path.is_file() and input_path.suffix.lower() in PDF_SUFFIXES:
        return [input_path]
    if input_path.is_dir():
        return sorted(p for p in input_path.rglob("*") if p.suffix.lower() in PDF_SUFFIXES)
    raise FileNotFoundError(f"No PDF file or directory found: {input_path}")


def guess_page_range(blocks: list[dict[str, Any]]) -> list[int | None]:
    pages = [b.get("page_idx") for b in blocks if b.get("page_idx") is not None]
    if not pages:
        return [None, None]
    return [min(pages), max(pages)]


def build_base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", required=True, help="Input PDF file or directory.")
    parser.add_argument("--output", required=True, help="Output directory.")
    return parser


def count_by(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "unknown")
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items(), key=lambda item: item[0]))


def load_normalized_blocks(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    data = read_json(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "blocks" in data:
        return data["blocks"]
    raise ValueError(f"Unsupported normalized block file: {path}")

