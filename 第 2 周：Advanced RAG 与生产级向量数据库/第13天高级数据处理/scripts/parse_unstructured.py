from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import ensure_dir, iter_input_pdfs, write_json


def element_to_dict(element, idx: int, source_pdf: Path) -> dict:
    metadata = element.metadata.to_dict() if getattr(element, "metadata", None) else {}
    return {
        "index": idx,
        "source_file": str(source_pdf),
        "category": getattr(element, "category", element.__class__.__name__),
        "text": str(element),
        "metadata": metadata,
    }


def parse_one_pdf(
    pdf_path: Path,
    output_dir: Path,
    strategy: str,
    languages: list[str],
    infer_table_structure: bool,
    extract_image_blocks: bool,
) -> None:
    try:
        from unstructured.partition.pdf import partition_pdf
    except ImportError as exc:
        raise RuntimeError(
            "unstructured is not installed. Run: pip install -r requirements.txt"
        ) from exc

    document_dir = ensure_dir(output_dir / pdf_path.stem)
    image_output_dir = ensure_dir(document_dir / "assets")

    kwargs = {
        "filename": str(pdf_path),
        "strategy": strategy,
        "languages": languages,
        "include_page_breaks": True,
        "infer_table_structure": infer_table_structure,
    }

    if extract_image_blocks:
        kwargs.update(
            {
                "extract_image_block_types": ["Image", "Table"],
                "extract_image_block_to_payload": False,
                "extract_image_block_output_dir": str(image_output_dir),
            }
        )

    elements = partition_pdf(**kwargs)
    rows = [element_to_dict(element, idx, pdf_path) for idx, element in enumerate(elements)]

    write_json(document_dir / "elements.json", rows)

    preview_lines = []
    for row in rows:
        page = row["metadata"].get("page_number")
        preview_lines.append(
            f"\n\n<!-- idx={row['index']} type={row['category']} page={page} -->\n{row['text']}"
        )
    (document_dir / "preview.md").write_text("".join(preview_lines).strip(), encoding="utf-8")

    tables_dir = ensure_dir(document_dir / "tables")
    for row in rows:
        if row["category"] != "Table":
            continue
        html = row["metadata"].get("text_as_html")
        if html:
            page = row["metadata"].get("page_number", "unknown")
            (tables_dir / f"page_{page}_table_{row['index']}.html").write_text(
                html, encoding="utf-8"
            )

    stats = {}
    for row in rows:
        stats[row["category"]] = stats.get(row["category"], 0) + 1
    write_json(document_dir / "stats.json", stats)


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse complex PDFs with Unstructured.")
    parser.add_argument("--input", default="data/raw_pdfs", help="Input PDF file or directory.")
    parser.add_argument("--output", default="outputs/unstructured", help="Output directory.")
    parser.add_argument("--strategy", default="hi_res", choices=["auto", "fast", "hi_res", "ocr_only"])
    parser.add_argument("--languages", default="eng,chi_sim", help="OCR languages, comma separated.")
    parser.add_argument("--no-table-structure", action="store_true")
    parser.add_argument("--extract-image-blocks", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = ensure_dir(Path(args.output))
    languages = [item.strip() for item in args.languages.split(",") if item.strip()]

    pdfs = iter_input_pdfs(input_path)
    for pdf_path in pdfs:
        print(f"[unstructured] parsing {pdf_path}")
        parse_one_pdf(
            pdf_path=pdf_path,
            output_dir=output_dir,
            strategy=args.strategy,
            languages=languages,
            infer_table_structure=not args.no_table_structure,
            extract_image_blocks=args.extract_image_blocks,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

