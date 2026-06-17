from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import ensure_dir, iter_input_pdfs


def parse_one_pdf(pdf_path: Path, output_dir: Path) -> None:
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise RuntimeError("docling is not installed. Run: pip install -r requirements.txt") from exc

    document_dir = ensure_dir(output_dir / pdf_path.stem)
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = result.document

    (document_dir / "docling.md").write_text(doc.export_to_markdown(), encoding="utf-8")
    (document_dir / "docling.json").write_text(doc.export_to_json(), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse PDFs with Docling as a baseline.")
    parser.add_argument("--input", default="data/raw_pdfs", help="Input PDF file or directory.")
    parser.add_argument("--output", default="outputs/docling", help="Output directory.")
    args = parser.parse_args()

    output_dir = ensure_dir(Path(args.output))
    for pdf_path in iter_input_pdfs(Path(args.input)):
        print(f"[docling] parsing {pdf_path}")
        parse_one_pdf(pdf_path, output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

