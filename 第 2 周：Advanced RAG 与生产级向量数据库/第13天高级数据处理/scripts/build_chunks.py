from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import RagChunk, dataclass_to_dict, ensure_dir, guess_page_range, load_normalized_blocks, normalize_text, stable_id, write_jsonl


TEXT_TYPES = {"title", "text", "list", "header", "footer"}
SPECIAL_TYPES = {"table", "image", "chart", "equation"}


def approx_tokens(text: str) -> int:
    # Good enough for chunk sizing without adding a tokenizer dependency.
    cjk_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other_words = len([w for w in text.split() if w])
    return cjk_chars + other_words


def text_for_block(block: dict) -> str:
    prefix = ""
    section_path = block.get("section_path") or []
    if section_path:
        prefix = " > ".join(section_path) + "\n"
    return normalize_text(prefix + (block.get("text") or ""))


def make_chunk(document_id: str, parser: str, blocks: list[dict], chunk_index: int) -> dict:
    source_file = blocks[0].get("source_file") or document_id
    text = normalize_text("\n\n".join(text_for_block(b) for b in blocks if b.get("text")))
    chunk_type = blocks[0].get("block_type") if len(blocks) == 1 else "text"
    chunk_id = f"{document_id}-{parser}-chunk-{chunk_index:04d}-{stable_id(text)}"
    section_path = blocks[-1].get("section_path") or []

    html = None
    image_path = None
    if len(blocks) == 1:
        html = blocks[0].get("html")
        image_path = blocks[0].get("image_path")

    return dataclass_to_dict(
        RagChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            source_file=source_file,
            parser=parser,
            chunk_type=chunk_type,
            text=text,
            source_blocks=[b.get("block_id") for b in blocks],
            page_range=guess_page_range(blocks),
            section_path=section_path,
            html=html,
            image_path=image_path,
            metadata={
                "block_types": [b.get("block_type") for b in blocks],
                "approx_tokens": approx_tokens(text),
            },
        )
    )


def build_chunks(blocks: list[dict], max_tokens: int, overlap_blocks: int) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for block in blocks:
        key = (block.get("document_id") or "unknown", block.get("parser") or "unknown")
        grouped.setdefault(key, []).append(block)

    chunks = []
    for (document_id, parser), doc_blocks in grouped.items():
        buffer: list[dict] = []
        chunk_index = 0

        def flush_buffer(keep_overlap: bool = True) -> None:
            nonlocal buffer, chunk_index
            if not buffer:
                return
            chunks.append(make_chunk(document_id, parser, buffer, chunk_index))
            chunk_index += 1
            if keep_overlap and overlap_blocks > 0:
                buffer = buffer[-overlap_blocks:]
            else:
                buffer = []

        for block in doc_blocks:
            block_type = block.get("block_type")
            text = normalize_text(block.get("text"))
            if not text and not block.get("html") and not block.get("image_path"):
                continue

            if block_type in SPECIAL_TYPES:
                flush_buffer(keep_overlap=False)
                special_text = text
                if block_type == "table" and block.get("html") and not special_text:
                    special_text = "Table block. See html payload."
                if block_type == "image" and not special_text:
                    special_text = "Image block. See image_path payload."
                special_block = {**block, "text": special_text}
                chunks.append(make_chunk(document_id, parser, [special_block], chunk_index))
                chunk_index += 1
                continue

            if block_type == "title":
                flush_buffer(keep_overlap=False)

            candidate = buffer + [block]
            candidate_text = "\n\n".join(text_for_block(b) for b in candidate)
            if buffer and approx_tokens(candidate_text) > max_tokens:
                flush_buffer()
            buffer.append(block)

        flush_buffer()

    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(description="Build RAG chunks from normalized blocks.")
    parser.add_argument("--input", default="outputs/normalized/all_blocks.jsonl")
    parser.add_argument("--output", default="outputs/normalized/rag_chunks.jsonl")
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--overlap-blocks", type=int, default=1)
    args = parser.parse_args()

    blocks = load_normalized_blocks(Path(args.input))
    chunks = build_chunks(blocks, max_tokens=args.max_tokens, overlap_blocks=args.overlap_blocks)
    write_jsonl(Path(args.output), chunks)
    print(f"Built {len(chunks)} chunks from {len(blocks)} blocks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
