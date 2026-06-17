from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import NormalizedBlock, dataclass_to_dict, ensure_dir, normalize_text, read_json, write_jsonl


TYPE_MAP = {
    "Title": "title",
    "NarrativeText": "text",
    "Text": "text",
    "ListItem": "list",
    "Table": "table",
    "Image": "image",
    "PageBreak": "page_break",
    "Header": "header",
    "Footer": "footer",
}


def normalize_one_document(elements_path: Path, output_dir: Path) -> list[dict]:
    rows = read_json(elements_path)
    document_id = elements_path.parent.name
    blocks = []
    section_path: list[str] = []

    for row in rows:
        category = row.get("category") or "Text"
        metadata = row.get("metadata") or {}
        block_type = TYPE_MAP.get(category, category.lower())
        text = normalize_text(row.get("text"))

        if block_type == "title" and text:
            section_path = [text]

        page_number = metadata.get("page_number")
        page_idx = page_number - 1 if isinstance(page_number, int) else page_number

        block = NormalizedBlock(
            document_id=document_id,
            source_file=row.get("source_file") or "",
            parser="unstructured",
            block_id=f"{document_id}-unstructured-{row.get('index')}",
            block_type=block_type,
            text=text,
            html=metadata.get("text_as_html"),
            image_path=metadata.get("image_path"),
            page_idx=page_idx,
            bbox=metadata.get("coordinates"),
            section_path=list(section_path),
            metadata={
                "raw_category": category,
                "raw_index": row.get("index"),
                **metadata,
            },
        )
        blocks.append(dataclass_to_dict(block))

    write_jsonl(output_dir / f"{document_id}.unstructured.blocks.jsonl", blocks)
    return blocks


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize Unstructured elements.json files.")
    parser.add_argument("--input", default="outputs/unstructured", help="Directory containing */elements.json.")
    parser.add_argument("--output", default="outputs/normalized", help="Output directory.")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = ensure_dir(Path(args.output))
    paths = sorted(input_dir.rglob("elements.json"))
    if not paths:
        print(f"No elements.json found under {input_dir}")
        return 1

    for path in paths:
        print(f"[normalize:unstructured] {path}")
        normalize_one_document(path, output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

