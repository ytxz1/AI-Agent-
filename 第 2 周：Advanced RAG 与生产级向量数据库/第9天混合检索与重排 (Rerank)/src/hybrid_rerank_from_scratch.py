"""
Hybrid Retrieval + Rerank from scratch.

This script is intentionally dependency-free. It demonstrates the core ideas
behind a production RAG retrieval pipeline:

1. Load documents.
2. Split documents into chunks.
3. Retrieve candidates with BM25.
4. Retrieve candidates with a TF-IDF cosine retriever as an embedding-like baseline.
5. Fuse candidates with Reciprocal Rank Fusion.
6. Rerank fused candidates with a lightweight cross-feature reranker.
7. Print and save a debug trace.

Run:
    python src/hybrid_rerank_from_scratch.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"
TRACE_PATH = OUTPUT_DIR / "from_scratch_retrieval_debug.jsonl"


@dataclass(frozen=True)
class Document:
    doc_id: str
    source: str
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    rank: int = 0
    method: str = ""
    debug: dict[str, float | int | str] = field(default_factory=dict)


def tokenize(text: str) -> list[str]:
    """Tokenize English terms, numbers, identifiers, and Chinese characters.

    This is not a production Chinese tokenizer. It is a small teaching tokenizer
    that makes the demo work without external dependencies.
    """

    text = text.lower()
    english_terms = re.findall(r"[a-z][a-z0-9_\-\.]*|\d+(?:\.\d+)?", text)
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    return english_terms + chinese_chars


def make_char_ngrams(text: str, n: int = 2) -> list[str]:
    """Create Chinese character bigrams for a stronger semantic-ish signal."""

    chars = re.findall(r"[\u4e00-\u9fff]", text)
    return ["".join(chars[i : i + n]) for i in range(max(0, len(chars) - n + 1))]


def analysis_terms(text: str) -> list[str]:
    """Terms used by the TF-IDF retriever.

    It combines tokenization and Chinese bigrams. In a real embedding retriever,
    this function would be replaced by an embedding model call.
    """

    return tokenize(text) + make_char_ngrams(text)


def load_documents(data_dir: Path) -> list[Document]:
    files = sorted(
        path for path in data_dir.rglob("*") if path.suffix.lower() in {".md", ".txt"}
    )
    documents = []

    for path in files:
        if path.name.lower() == "readme.md":
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        documents.append(
            Document(
                doc_id=path.stem,
                source=path.relative_to(ROOT).as_posix(),
                text=text,
            )
        )

    if not documents:
        raise RuntimeError(
            f"No .md or .txt documents found in {data_dir}. "
            "Add sample documents or keep the provided examples."
        )

    return documents


def split_into_chunks(
    documents: Iterable[Document],
    chunk_size: int = 700,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    chunks: list[Chunk] = []

    for document in documents:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", document.text) if p.strip()]
        current = ""
        local_index = 0

        for paragraph in paragraphs:
            candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph

            if len(candidate) <= chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.doc_id}::chunk_{local_index:03d}",
                        doc_id=document.doc_id,
                        source=document.source,
                        text=current,
                        metadata={"source": document.source},
                    )
                )
                local_index += 1

            overlap_text = current[-chunk_overlap:] if chunk_overlap > 0 else ""
            current = f"{overlap_text}\n\n{paragraph}".strip()

        if current:
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}::chunk_{local_index:03d}",
                    doc_id=document.doc_id,
                    source=document.source,
                    text=current,
                    metadata={"source": document.source},
                )
            )

    return chunks


class BM25Retriever:
    """A compact BM25 implementation for teaching."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.term_freqs = [Counter(tokenize(chunk.text)) for chunk in chunks]
        self.doc_lengths = [sum(freqs.values()) for freqs in self.term_freqs]
        self.avg_doc_length = sum(self.doc_lengths) / max(1, len(self.doc_lengths))
        self.doc_freqs = self._build_doc_freqs()
        self.total_docs = len(chunks)

    def _build_doc_freqs(self) -> Counter:
        doc_freqs = Counter()
        for freqs in self.term_freqs:
            for term in freqs:
                doc_freqs[term] += 1
        return doc_freqs

    def idf(self, term: str) -> float:
        doc_freq = self.doc_freqs.get(term, 0)
        return math.log(1 + (self.total_docs - doc_freq + 0.5) / (doc_freq + 0.5))

    def score_chunk(self, query_terms: list[str], chunk_index: int) -> float:
        freqs = self.term_freqs[chunk_index]
        doc_length = self.doc_lengths[chunk_index]
        score = 0.0

        for term in query_terms:
            term_frequency = freqs.get(term, 0)
            if term_frequency == 0:
                continue

            numerator = term_frequency * (self.k1 + 1)
            denominator = term_frequency + self.k1 * (
                1 - self.b + self.b * doc_length / self.avg_doc_length
            )
            score += self.idf(term) * numerator / denominator

        return score

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_terms = tokenize(query)
        results = []

        for index, chunk in enumerate(self.chunks):
            score = self.score_chunk(query_terms, index)
            if score > 0:
                results.append(
                    SearchResult(
                        chunk=chunk,
                        score=score,
                        method="bm25",
                        debug={"matched_terms": len(set(query_terms) & set(self.term_freqs[index]))},
                    )
                )

        return rank_results(results, top_k=top_k)


class TfidfEmbeddingLikeRetriever:
    """A TF-IDF cosine retriever used as an embedding-like teaching baseline.

    Real embedding retrieval encodes text into dense vectors with a neural model.
    This class uses TF-IDF sparse vectors so the demo can run without network or
    model downloads, while still showing the same retrieve-by-vector shape.
    """

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.term_freqs = [Counter(analysis_terms(chunk.text)) for chunk in chunks]
        self.doc_freqs = self._build_doc_freqs()
        self.total_docs = len(chunks)
        self.vectors = [self._to_tfidf_vector(freqs) for freqs in self.term_freqs]
        self.norms = [vector_norm(vector) for vector in self.vectors]

    def _build_doc_freqs(self) -> Counter:
        doc_freqs = Counter()
        for freqs in self.term_freqs:
            for term in freqs:
                doc_freqs[term] += 1
        return doc_freqs

    def idf(self, term: str) -> float:
        return math.log((self.total_docs + 1) / (self.doc_freqs.get(term, 0) + 1)) + 1

    def _to_tfidf_vector(self, freqs: Counter) -> dict[str, float]:
        total_terms = sum(freqs.values()) or 1
        return {
            term: (count / total_terms) * self.idf(term)
            for term, count in freqs.items()
        }

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_freqs = Counter(analysis_terms(query))
        query_vector = self._to_tfidf_vector(query_freqs)
        query_norm = vector_norm(query_vector)
        results = []

        for index, chunk in enumerate(self.chunks):
            score = cosine_similarity(query_vector, query_norm, self.vectors[index], self.norms[index])
            if score > 0:
                results.append(
                    SearchResult(
                        chunk=chunk,
                        score=score,
                        method="embedding_like",
                        debug={"vector_terms": len(query_vector)},
                    )
                )

        return rank_results(results, top_k=top_k)


def vector_norm(vector: dict[str, float]) -> float:
    return math.sqrt(sum(value * value for value in vector.values()))


def cosine_similarity(
    left: dict[str, float],
    left_norm: float,
    right: dict[str, float],
    right_norm: float,
) -> float:
    if left_norm == 0 or right_norm == 0:
        return 0.0

    if len(left) > len(right):
        left, right = right, left

    dot = sum(value * right.get(term, 0.0) for term, value in left.items())
    return dot / (left_norm * right_norm)


def rank_results(results: list[SearchResult], top_k: int) -> list[SearchResult]:
    ranked = sorted(results, key=lambda item: item.score, reverse=True)[:top_k]
    for rank, result in enumerate(ranked, start=1):
        result.rank = rank
    return ranked


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    top_k: int = 10,
    rrf_k: int = 60,
) -> list[SearchResult]:
    fused_scores: dict[str, float] = defaultdict(float)
    chunks_by_id: dict[str, Chunk] = {}
    sources_by_id: dict[str, list[str]] = defaultdict(list)

    for results in result_lists:
        for result in results:
            chunk_id = result.chunk.chunk_id
            chunks_by_id[chunk_id] = result.chunk
            fused_scores[chunk_id] += 1.0 / (rrf_k + result.rank)
            sources_by_id[chunk_id].append(f"{result.method}@{result.rank}")

    fused_results = [
        SearchResult(
            chunk=chunks_by_id[chunk_id],
            score=score,
            method="rrf_fusion",
            debug={"sources": ",".join(sources_by_id[chunk_id])},
        )
        for chunk_id, score in fused_scores.items()
    ]

    return rank_results(fused_results, top_k=top_k)


class LightweightReranker:
    """A small feature-based reranker that mimics second-stage ranking.

    It scores query-chunk pairs with features that a real cross-encoder reranker
    would learn automatically: term coverage, phrase match, source title match,
    and the first-stage fusion score.
    """

    def rerank(self, query: str, candidates: list[SearchResult], top_n: int = 5) -> list[SearchResult]:
        query_terms = set(tokenize(query))
        query_bigrams = set(make_char_ngrams(query))
        query_keyword_terms = {
            term for term in query_terms if re.search(r"[a-z0-9_\-\.]", term) and len(term) >= 3
        }
        reranked = []

        for candidate in candidates:
            text = candidate.chunk.text.lower()
            chunk_terms = set(tokenize(text))
            chunk_bigrams = set(make_char_ngrams(text))

            term_coverage = safe_divide(len(query_terms & chunk_terms), len(query_terms))
            bigram_coverage = safe_divide(len(query_bigrams & chunk_bigrams), len(query_bigrams))
            keyword_coverage = safe_divide(
                len(query_keyword_terms & chunk_terms),
                len(query_keyword_terms),
            )
            exact_bonus = 1.0 if query.lower() in text else 0.0
            title_bonus = safe_divide(
                sum(1 for term in query_keyword_terms if term in candidate.chunk.source.lower()),
                len(query_keyword_terms),
            )

            rerank_score = (
                0.35 * term_coverage
                + 0.25 * keyword_coverage
                + 0.15 * bigram_coverage
                + 0.15 * normalize_first_stage_score(candidate.score)
                + 0.07 * exact_bonus
                + 0.03 * title_bonus
            )

            reranked.append(
                SearchResult(
                    chunk=candidate.chunk,
                    score=rerank_score,
                    method="lightweight_rerank",
                    debug={
                        "term_coverage": round(term_coverage, 4),
                        "keyword_coverage": round(keyword_coverage, 4),
                        "bigram_coverage": round(bigram_coverage, 4),
                        "first_stage_score": round(candidate.score, 6),
                        "exact_bonus": exact_bonus,
                        "title_bonus": title_bonus,
                    },
                )
            )

        return rank_results(reranked, top_k=top_n)


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def normalize_first_stage_score(score: float) -> float:
    return score / (score + 1.0)


def format_results(title: str, results: list[SearchResult]) -> str:
    lines = [f"\n=== {title} ==="]
    for result in results:
        preview = " ".join(result.chunk.text.split())[:160]
        lines.append(
            f"{result.rank:>2}. score={result.score:.4f} "
            f"source={result.chunk.source} chunk={result.chunk.chunk_id}"
        )
        lines.append(f"    {preview}")
        if result.debug:
            lines.append(f"    debug={result.debug}")
    return "\n".join(lines)


def serialize_results(results: list[SearchResult]) -> list[dict]:
    return [
        {
            "rank": result.rank,
            "score": result.score,
            "method": result.method,
            "chunk_id": result.chunk.chunk_id,
            "source": result.chunk.source,
            "debug": result.debug,
            "preview": " ".join(result.chunk.text.split())[:240],
        }
        for result in results
    ]


def save_trace(record: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with TRACE_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def answer_from_context(query: str, reranked_results: list[SearchResult]) -> str:
    """A tiny answer composer for demonstration.

    In a real RAG system this step is handled by an LLM. Here we produce a
    transparent extractive answer so the script stays offline and deterministic.
    """

    lines = [
        f"问题：{query}",
        "回答：混合检索先用 BM25 捕捉关键词、术语、字段名等精确匹配，再用 Embedding 类检索补充语义相关片段；"
        "随后 Reranker 对融合后的候选进行 query-chunk 级别精排，把最相关的上下文放到前面。",
        "依据片段：",
    ]

    for result in reranked_results[:3]:
        preview = " ".join(result.chunk.text.split())[:220]
        lines.append(f"- {result.chunk.source} / {result.chunk.chunk_id}: {preview}")

    return "\n".join(lines)


def run_pipeline(query: str, top_k: int, fusion_top_k: int, rerank_top_n: int) -> dict:
    documents = load_documents(DATA_DIR)
    chunks = split_into_chunks(documents)

    bm25 = BM25Retriever(chunks)
    embedding_like = TfidfEmbeddingLikeRetriever(chunks)
    reranker = LightweightReranker()

    bm25_results = bm25.retrieve(query, top_k=top_k)
    embedding_results = embedding_like.retrieve(query, top_k=top_k)
    fused_results = reciprocal_rank_fusion(
        [bm25_results, embedding_results],
        top_k=fusion_top_k,
        rrf_k=60,
    )
    reranked_results = reranker.rerank(query, fused_results, top_n=rerank_top_n)
    answer = answer_from_context(query, reranked_results)

    record = {
        "query": query,
        "num_documents": len(documents),
        "num_chunks": len(chunks),
        "bm25": serialize_results(bm25_results),
        "embedding_like": serialize_results(embedding_results),
        "hybrid_rrf": serialize_results(fused_results),
        "reranked": serialize_results(reranked_results),
        "answer": answer,
    }
    save_trace(record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        default="为什么生产级 RAG 需要 BM25 和 Embedding 混合检索，并在最后加入 Rerank？",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--fusion-top-k", type=int, default=8)
    parser.add_argument("--rerank-top-n", type=int, default=4)
    args = parser.parse_args()

    record = run_pipeline(
        query=args.query,
        top_k=args.top_k,
        fusion_top_k=args.fusion_top_k,
        rerank_top_n=args.rerank_top_n,
    )

    print(format_results("BM25 sparse retrieval", deserialize_for_display(record["bm25"])))
    print(format_results("Embedding-like TF-IDF retrieval", deserialize_for_display(record["embedding_like"])))
    print(format_results("Hybrid retrieval with RRF", deserialize_for_display(record["hybrid_rrf"])))
    print(format_results("Reranked final contexts", deserialize_for_display(record["reranked"])))
    print("\n=== Offline answer demo ===")
    print(record["answer"])
    print(f"\nDebug trace saved to: {TRACE_PATH}")


def deserialize_for_display(rows: list[dict]) -> list[SearchResult]:
    results = []
    for row in rows:
        chunk = Chunk(
            chunk_id=row["chunk_id"],
            doc_id=row["chunk_id"].split("::")[0],
            source=row["source"],
            text=row["preview"],
            metadata={"source": row["source"]},
        )
        results.append(
            SearchResult(
                chunk=chunk,
                score=row["score"],
                rank=row["rank"],
                method=row["method"],
                debug=row["debug"],
            )
        )
    return results


if __name__ == "__main__":
    main()
