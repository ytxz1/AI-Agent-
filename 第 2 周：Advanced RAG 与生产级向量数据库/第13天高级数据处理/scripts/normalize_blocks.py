from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common import ensure_dir, load_normalized_blocks, write_json, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge normalized *.blocks.jsonl into one file.")
    parser.add_argument("--input", default="outputs/normalized", help="Directory containing blocks json/jsonl.")
    parser.add_argument("--output", default="outputs/normalized/all_blocks.jsonl")
    parser.add_argument("--manifest", default="outputs/normalized/manifest.json")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)
    ensure_dir(output_path.parent)

    files = sorted(
        p for p in input_dir.rglob("*.blocks.jsonl") if p.resolve() != output_path.resolve()
    )
    all_blocks = []
    manifest = []

    for path in files:
        blocks = load_normalized_blocks(path)
        all_blocks.extend(blocks)
        manifest.append(
            {
                "file": str(path),
                "parser": blocks[0].get("parser") if blocks else None,
                "document_id": blocks[0].get("document_id") if blocks else None,
                "blocks": len(blocks),
            }
        )

    write_jsonl(output_path, all_blocks)
    write_json(Path(args.manifest), {"files": manifest, "total_blocks": len(all_blocks)})
    print(json.dumps({"files": len(files), "total_blocks": len(all_blocks)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

