from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from common import NormalizedBlock, dataclass_to_dict, ensure_dir, normalize_text, write_jsonl


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


def markdown_to_blocks(markdown_path: Path, output_dir: Path) -> list[dict]:
    document_id = markdown_path.parent.name
    text = markdown_path.read_text(encoding="utf-8")
    blocks = []
    section_path: list[str] = []
    buffer: list[str] = []
    block_idx = 0

    def flush_text() -> None:
        nonlocal block_idx
        body = normalize_text("\n".join(buffer))
        buffer.clear()
        if not body:
            return
        block = NormalizedBlock(
            document_id=document_id,
            source_file=document_id,
            parser="docling",
            block_id=f"{document_id}-docling-{block_idx}",
            block_type="text",
            text=body,
            section_path=list(section_path),
            metadata={"source_markdown": str(markdown_path)},
        )
        blocks.append(dataclass_to_dict(block))
        block_idx += 1

    for line in text.splitlines():
        heading = HEADING_RE.match(line)
        if heading:
            flush_text()
            level = len(heading.group(1))
            title = normalize_text(heading.group(2))
            while len(section_path) >= level:
                section_path.pop()
            section_path.append(title)
            block = NormalizedBlock(
                document_id=document_id,
                source_file=document_id,
                parser="docling",
                block_id=f"{document_id}-docling-{block_idx}",
                block_type="title",
                text=title,
                section_path=list(section_path),
                metadata={"heading_level": level, "source_markdown": str(markdown_path)},
            )
            blocks.append(dataclass_to_dict(block))
            block_idx += 1
        else:
            buffer.append(line)

    flush_text()
    write_jsonl(output_dir / f"{document_id}.docling.blocks.jsonl", blocks)
    return blocks


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize Docling Markdown output.")
    parser.add_argument("--input", default="outputs/docling", help="Directory containing */docling.md.")
    parser.add_argument("--output", default="outputs/normalized", help="Output directory.")
    args = parser.parse_args()

    output_dir = ensure_dir(Path(args.output))
    paths = sorted(Path(args.input).rglob("docling.md"))
    if not paths:
        print(f"No docling.md found under {args.input}")
        return 1

    for path in paths:
        print(f"[normalize:docling] {path}")
        markdown_to_blocks(path, output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

