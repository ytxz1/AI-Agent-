from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean

from common import count_by, load_normalized_blocks, normalize_text, write_json


def image_path_exists(block: dict) -> bool:
    image_path = block.get("image_path")
    if not image_path:
        return False
    return Path(image_path).exists()


def evaluate(blocks: list[dict]) -> dict:
    total = len(blocks)
    if total == 0:
        return {"total_blocks": 0}

    text_lengths = [len(normalize_text(b.get("text"))) for b in blocks]
    image_blocks = [b for b in blocks if b.get("block_type") in {"image", "chart", "table"} and b.get("image_path")]
    table_blocks = [b for b in blocks if b.get("block_type") == "table"]

    by_doc_parser: dict[str, int] = {}
    for block in blocks:
        key = f"{block.get('document_id')}::{block.get('parser')}"
        by_doc_parser[key] = by_doc_parser.get(key, 0) + 1

    return {
        "total_blocks": total,
        "by_parser": count_by(blocks, "parser"),
        "by_type": count_by(blocks, "block_type"),
        "by_document_parser": dict(sorted(by_doc_parser.items())),
        "text": {
            "with_text_ratio": round(sum(length > 0 for length in text_lengths) / total, 4),
            "avg_text_chars": round(mean(text_lengths), 2),
            "empty_text_blocks": sum(length == 0 for length in text_lengths),
        },
        "metadata": {
            "with_page_ratio": round(sum(b.get("page_idx") is not None for b in blocks) / total, 4),
            "with_bbox_ratio": round(sum(b.get("bbox") is not None for b in blocks) / total, 4),
            "with_section_path_ratio": round(sum(bool(b.get("section_path")) for b in blocks) / total, 4),
        },
        "tables": {
            "table_blocks": len(table_blocks),
            "with_html_ratio": round(
                sum(bool(b.get("html")) for b in table_blocks) / len(table_blocks), 4
            )
            if table_blocks
            else 0,
        },
        "assets": {
            "blocks_with_image_path": len(image_blocks),
            "existing_image_path_ratio": round(
                sum(image_path_exists(b) for b in image_blocks) / len(image_blocks), 4
            )
            if image_blocks
            else 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate normalized parse quality.")
    parser.add_argument("--input", default="outputs/normalized/all_blocks.jsonl")
    parser.add_argument("--output", default="outputs/reports/parse_quality_report.json")
    args = parser.parse_args()

    blocks = load_normalized_blocks(Path(args.input))
    report = evaluate(blocks)
    write_json(Path(args.output), report)
    print(f"Wrote report: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

