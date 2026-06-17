from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from common import NormalizedBlock, dataclass_to_dict, ensure_dir, normalize_text, read_json, write_jsonl


def first_existing_text(item: dict[str, Any], keys: list[str]) -> str:
    parts = []
    for key in keys:
        value = item.get(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(str(v) for v in value if v)
    return normalize_text("\n".join(parts))


def find_content_list_files(input_dir: Path) -> list[Path]:
    patterns = ["*content_list.json", "*_content_list.json"]
    results: set[Path] = set()
    for pattern in patterns:
        results.update(input_dir.rglob(pattern))
    return sorted(results)


def infer_document_id(path: Path) -> str:
    name = path.name
    for suffix in ["_content_list.json", "content_list.json"]:
        if name.endswith(suffix):
            return name[: -len(suffix)].rstrip("_-") or path.parent.name
    return path.parent.name


def normalize_one_content_list(content_list_path: Path, output_dir: Path) -> list[dict]:
    items = read_json(content_list_path)
    document_id = infer_document_id(content_list_path)
    blocks = []
    section_path: list[str] = []

    for idx, item in enumerate(items):
        raw_type = item.get("type") or "text"
        block_type = {
            "text": "text",
            "title": "title",
            "table": "table",
            "image": "image",
            "equation": "equation",
            "interline_equation": "equation",
            "isolated_equation": "equation",
            "list": "list",
        }.get(raw_type, raw_type)

        text = first_existing_text(
            item,
            [
                "text",
                "table_caption",
                "table_body",
                "image_caption",
                "img_caption",
                "latex",
            ],
        )

        text_level = item.get("text_level")
        if block_type == "text" and isinstance(text_level, int):
            block_type = "title"
        if block_type == "title" and text:
            while len(section_path) >= (text_level or 1):
                section_path.pop()
            section_path.append(text)

        image_path = item.get("img_path") or item.get("image_path")
        if image_path and not Path(image_path).is_absolute():
            image_path = str((content_list_path.parent / image_path).resolve())

        block = NormalizedBlock(
            document_id=document_id,
            source_file=item.get("source_file") or document_id,
            parser="mineru",
            block_id=f"{document_id}-mineru-{idx}",
            block_type=block_type,
            text=text,
            html=item.get("table_body") if block_type == "table" else None,
            image_path=image_path,
            page_idx=item.get("page_idx"),
            bbox=item.get("bbox"),
            section_path=list(section_path),
            metadata={
                "raw_index": idx,
                "raw_type": raw_type,
                "content_list_path": str(content_list_path),
                **item,
            },
        )
        blocks.append(dataclass_to_dict(block))

    write_jsonl(output_dir / f"{document_id}.mineru.blocks.jsonl", blocks)
    return blocks


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize MinerU content_list.json files.")
    parser.add_argument("--input", default="outputs/mineru", help="MinerU output directory.")
    parser.add_argument("--output", default="outputs/normalized", help="Output directory.")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = ensure_dir(Path(args.output))
    paths = find_content_list_files(input_dir)
    if not paths:
        print(f"No content_list.json found under {input_dir}")
        return 1

    for path in paths:
        print(f"[normalize:mineru] {path}")
        normalize_one_content_list(path, output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

