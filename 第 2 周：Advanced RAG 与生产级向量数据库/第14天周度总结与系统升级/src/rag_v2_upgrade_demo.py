"""
Day 14 RAG v2 upgrade demo.

This file implements a complete, teaching-oriented RAG retrieval pipeline:

1. Load sample Markdown documents.
2. Split them into chunks with metadata.
3. Generate deterministic hashing embeddings.
4. Retrieve with BM25 sparse retrieval.
5. Retrieve with local dense retrieval, or optional Milvus dense retrieval.
6. Fuse candidates with Reciprocal Rank Fusion.
7. Rerank fused candidates with a lightweight feature-based reranker.
8. Build a citation-friendly context.
9. Compose an offline answer.
10. Save a retrieval trace JSONL record.

Run offline:
    python src/rag_v2_upgrade_demo.py

Run with a custom query:
    python src/rag_v2_upgrade_demo.py --query "Milvus 在 RAG 中负责什么？"

Run with optional Milvus backend:
    python src/rag_v2_upgrade_demo.py --vector-backend milvus --rebuild-milvus
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"
TRACE_PATH = OUTPUT_DIR / "rag_v2_retrieval_trace.jsonl"


@dataclass(frozen=True)
class Document:
    doc_id: str
    source: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class TransformedQuery:
    text: str
    strategy: str
    weight: float = 1.0
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    rank: int = 0
    method: str = ""
    debug: dict[str, object] = field(default_factory=dict)


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[SearchResult]:
        ...


def now_ms() -> int:
    return int(time.perf_counter() * 1000)


def tokenize(text: str) -> list[str]:
    """Small tokenizer for Chinese, English identifiers, numbers, and config keys."""

    text = text.lower()
    english_terms = re.findall(r"[a-z][a-z0-9_\-\.]*|\d+(?:\.\d+)?", text)
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    return english_terms + chinese_chars


def chinese_ngrams(text: str, n: int = 2) -> list[str]:
    chars = re.findall(r"[\u4e00-\u9fff]", text)
    return ["".join(chars[i : i + n]) for i in range(max(0, len(chars) - n + 1))]


def analysis_terms(text: str) -> list[str]:
    return tokenize(text) + chinese_ngrams(text, n=2)


def stable_int_id(text: str, max_value: int = 2_147_483_647) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % max_value


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    documents: list[Document] = []
    paths = sorted(path for path in data_dir.rglob("*") if path.suffix.lower() in {".md", ".txt"})

    for path in paths:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        relative_source = path.relative_to(ROOT).as_posix()
        documents.append(
            Document(
                doc_id=path.stem,
                source=relative_source,
                text=text,
                metadata={
                    "source": relative_source,
                    "file_name": path.name,
                    "corpus_version": "day14_demo_v1",
                },
            )
        )

    if not documents:
        raise RuntimeError(f"No Markdown or text documents found in {data_dir}.")

    return documents


def split_into_chunks(
    documents: Iterable[Document],
    chunk_size: int = 720,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    """Paragraph-aware chunking with stable IDs and inherited metadata."""

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
                    build_chunk(
                        document=document,
                        local_index=local_index,
                        text=current,
                    )
                )
                local_index += 1

            overlap = current[-chunk_overlap:] if chunk_overlap > 0 else ""
            current = f"{overlap}\n\n{paragraph}".strip()

        if current:
            chunks.append(
                build_chunk(
                    document=document,
                    local_index=local_index,
                    text=current,
                )
            )

    return chunks


def build_chunk(document: Document, local_index: int, text: str) -> Chunk:
    chunk_id = f"{document.doc_id}::chunk_{local_index:03d}"
    title = extract_title(document.text)
    return Chunk(
        chunk_id=chunk_id,
        doc_id=document.doc_id,
        source=document.source,
        text=text,
        metadata={
            **document.metadata,
            "chunk_id": chunk_id,
            "doc_id": document.doc_id,
            "title": title,
            "chunk_type": "text",
            "section_path": title,
        },
    )


def extract_title(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return ""


class HashingEmbedder:
    """Deterministic local embedding substitute.

    A real project would call OpenAI, BGE, GTE, E5, or another embedding model.
    This hash embedding keeps the demo offline and deterministic while preserving
    the same API shape: text -> dense vector.
    """

    def __init__(self, dimension: int = 128):
        self.dimension = dimension

    def embed_query(self, text: str) -> list[float]:
        return self.embed_text(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]

    def embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for term in analysis_terms(text):
            digest = hashlib.md5(term.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.dimension
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class BM25Retriever:
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
            tf = freqs.get(term, 0)
            if tf == 0:
                continue

            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
            score += self.idf(term) * numerator / denominator

        return score

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_terms = tokenize(query)
        results: list[SearchResult] = []

        for index, chunk in enumerate(self.chunks):
            score = self.score_chunk(query_terms, index)
            if score <= 0:
                continue

            matched = sorted(set(query_terms) & set(self.term_freqs[index]))
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                    method="bm25",
                    debug={"matched_terms": matched[:20]},
                )
            )

        return rank_results(results, top_k)


class LocalDenseRetriever:
    def __init__(self, chunks: list[Chunk], embedder: HashingEmbedder):
        self.chunks = chunks
        self.embedder = embedder
        self.vectors = embedder.embed_documents([chunk.text for chunk in chunks])

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_vector = self.embedder.embed_query(query)
        results: list[SearchResult] = []

        for chunk, vector in zip(self.chunks, self.vectors):
            score = dot(query_vector, vector)
            if score <= 0:
                continue

            results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                    method="local_dense",
                    debug={"embedding": "hashing"},
                )
            )

        return rank_results(results, top_k)


class MilvusDenseRetriever:
    """Optional Milvus retriever.

    It stores the same deterministic hash vectors in Milvus, so you can verify
    the real vector database flow without calling an external embedding API.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        embedder: HashingEmbedder,
        uri: str = "http://localhost:19530",
        token: str = "root:Milvus",
        collection_name: str = "rag_chunks_day14",
        rebuild: bool = False,
    ):
        try:
            from pymilvus import DataType, MilvusClient
        except ImportError as exc:
            raise RuntimeError("pymilvus is not installed. Run: pip install pymilvus") from exc

        self.DataType = DataType
        self.client = MilvusClient(uri=uri, token=token)
        self.chunks = chunks
        self.embedder = embedder
        self.collection_name = collection_name

        if rebuild or not self.client.has_collection(collection_name):
            self.recreate_collection()
            self.insert_chunks()

        self.client.load_collection(collection_name=collection_name)

    def recreate_collection(self) -> None:
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("id", self.DataType.INT64, is_primary=True)
        schema.add_field("vector", self.DataType.FLOAT_VECTOR, dim=self.embedder.dimension)
        schema.add_field("chunk_id", self.DataType.VARCHAR, max_length=256)
        schema.add_field("doc_id", self.DataType.VARCHAR, max_length=256)
        schema.add_field("text", self.DataType.VARCHAR, max_length=8192)
        schema.add_field("source", self.DataType.VARCHAR, max_length=1024)
        schema.add_field("title", self.DataType.VARCHAR, max_length=512)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def insert_chunks(self) -> None:
        vectors = self.embedder.embed_documents([chunk.text for chunk in self.chunks])
        rows = []
        for chunk, vector in zip(self.chunks, vectors):
            rows.append(
                {
                    "id": stable_int_id(chunk.chunk_id),
                    "vector": vector,
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "source": chunk.source,
                    "title": chunk.metadata.get("title", ""),
                }
            )

        self.client.insert(collection_name=self.collection_name, data=rows)
        self.client.flush(collection_name=self.collection_name)

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_vector = self.embedder.embed_query(query)
        raw_results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            anns_field="vector",
            limit=top_k,
            output_fields=["chunk_id", "doc_id", "text", "source", "title"],
            search_params={"metric_type": "COSINE"},
        )

        results: list[SearchResult] = []
        for hits in raw_results:
            for hit in hits:
                entity = hit.get("entity", {})
                chunk = Chunk(
                    chunk_id=entity["chunk_id"],
                    doc_id=entity["doc_id"],
                    source=entity["source"],
                    text=entity["text"],
                    metadata={
                        "title": entity.get("title", ""),
                        "source": entity["source"],
                    },
                )
                results.append(
                    SearchResult(
                        chunk=chunk,
                        score=float(hit.get("distance", 0.0)),
                        method="milvus_dense",
                        debug={"backend": "milvus"},
                    )
                )

        return rank_results(results, top_k)


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def rank_results(results: list[SearchResult], top_k: int) -> list[SearchResult]:
    ranked = sorted(results, key=lambda item: item.score, reverse=True)[:top_k]
    for rank, result in enumerate(ranked, start=1):
        result.rank = rank
    return ranked


class RuleBasedQueryTransformer:
    """Offline query transformation examples.

    Real systems usually call an LLM here. This class keeps the demo runnable
    while showing the exact data shape and pipeline position.
    """

    def transform(self, query: str, strategies: list[str]) -> list[TransformedQuery]:
        transformed: list[TransformedQuery] = []

        if "original" in strategies:
            transformed.append(TransformedQuery(text=query, strategy="original", weight=1.0))

        if "hyde" in strategies:
            transformed.append(
                TransformedQuery(
                    text=(
                        f"{query}。这通常涉及 RAG 系统中的文档切分、Embedding、BM25、"
                        "Milvus 向量检索、混合检索、RRF 融合、Reranker 精排、评估指标和 retrieval trace。"
                    ),
                    strategy="hyde",
                    weight=0.8,
                    metadata={"note": "offline hypothetical document"},
                )
            )

        if "multi_query" in strategies:
            transformed.extend(
                [
                    TransformedQuery(
                        text=f"{query} 的工程实现步骤是什么？",
                        strategy="multi_query",
                        weight=0.9,
                    ),
                    TransformedQuery(
                        text=f"{query} 在生产级 RAG 系统中的作用和风险是什么？",
                        strategy="multi_query",
                        weight=0.9,
                    ),
                ]
            )

        if "decomposition" in strategies:
            transformed.extend(self.decompose(query))

        return deduplicate_transformed_queries(transformed)

    def decompose(self, query: str) -> list[TransformedQuery]:
        if "和" not in query and "以及" not in query and "如何" not in query:
            return []

        return [
            TransformedQuery(
                text="RAG 系统中 Milvus 负责什么？",
                strategy="decomposition",
                weight=0.85,
            ),
            TransformedQuery(
                text="RAG 系统中 BM25、Embedding 和 Reranker 如何配合？",
                strategy="decomposition",
                weight=0.85,
            ),
        ]


def deduplicate_transformed_queries(queries: list[TransformedQuery]) -> list[TransformedQuery]:
    seen = set()
    output = []
    for item in queries:
        key = item.text.strip()
        if key and key not in seen:
            seen.add(key)
            output.append(item)
    return output


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    top_k: int = 8,
    rrf_k: int = 60,
) -> list[SearchResult]:
    scores: dict[str, float] = defaultdict(float)
    chunks_by_id: dict[str, Chunk] = {}
    traces_by_id: dict[str, list[dict[str, object]]] = defaultdict(list)

    for results in result_lists:
        for result in results:
            chunk_id = result.chunk.chunk_id
            chunks_by_id[chunk_id] = result.chunk
            scores[chunk_id] += 1.0 / (rrf_k + result.rank)
            traces_by_id[chunk_id].append(
                {
                    "method": result.method,
                    "rank": result.rank,
                    "score": round(result.score, 6),
                    "query_strategy": result.debug.get("query_strategy"),
                }
            )

    fused = [
        SearchResult(
            chunk=chunks_by_id[chunk_id],
            score=score,
            method="rrf_fusion",
            debug={"matched_by": traces_by_id[chunk_id]},
        )
        for chunk_id, score in scores.items()
    ]

    return rank_results(fused, top_k)


class LightweightReranker:
    def rerank(self, original_query: str, candidates: list[SearchResult], top_n: int = 4) -> list[SearchResult]:
        query_terms = set(tokenize(original_query))
        query_bigrams = set(chinese_ngrams(original_query))
        keyword_terms = {
            term for term in query_terms if re.search(r"[a-z0-9_\-\.]", term) or len(term) >= 2
        }
        reranked: list[SearchResult] = []

        for candidate in candidates:
            text = candidate.chunk.text.lower()
            chunk_terms = set(tokenize(text))
            chunk_bigrams = set(chinese_ngrams(text))

            term_coverage = safe_divide(len(query_terms & chunk_terms), len(query_terms))
            keyword_coverage = safe_divide(len(keyword_terms & chunk_terms), len(keyword_terms))
            bigram_coverage = safe_divide(len(query_bigrams & chunk_bigrams), len(query_bigrams))
            title_text = candidate.chunk.metadata.get("title", "").lower()
            title_bonus = 1.0 if any(term in title_text for term in keyword_terms) else 0.0
            first_stage = candidate.score / (candidate.score + 1.0)

            rerank_score = (
                0.34 * term_coverage
                + 0.24 * keyword_coverage
                + 0.18 * bigram_coverage
                + 0.16 * first_stage
                + 0.08 * title_bonus
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
                        "title_bonus": title_bonus,
                        "fusion_trace": candidate.debug.get("matched_by", []),
                    },
                )
            )

        return rank_results(reranked, top_n)


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def build_context(results: list[SearchResult], max_chars: int = 2600) -> str:
    sections = []
    used_chars = 0

    for index, result in enumerate(results, start=1):
        content = " ".join(result.chunk.text.split())
        block = (
            f"[Source {index}]\n"
            f"source: {result.chunk.source}\n"
            f"chunk_id: {result.chunk.chunk_id}\n"
            f"title: {result.chunk.metadata.get('title', '')}\n"
            f"content: {content}\n"
        )
        if used_chars + len(block) > max_chars:
            break
        sections.append(block)
        used_chars += len(block)

    return "\n".join(sections)


def compose_offline_answer(query: str, reranked_results: list[SearchResult]) -> str:
    """Extractive answer composer used instead of an LLM for offline execution."""

    lines = [
        f"问题：{query}",
        "",
        "离线示例回答：",
        "升级后的 RAG v2 会把第一周的单路向量检索改造成可观测、可评估的多阶段检索链路。"
        "Milvus 负责生产级向量存储和 dense retrieval，BM25 负责精确术语召回，RRF 负责融合多路候选，"
        "Reranker 负责把最相关的 chunk 排到前面，Retrieval Trace 和评估指标负责验证系统是否真的变好。",
        "",
        "引用依据：",
    ]

    for index, result in enumerate(reranked_results[:3], start=1):
        preview = " ".join(result.chunk.text.split())[:220]
        lines.append(f"[Source {index}] {result.chunk.source} / {result.chunk.chunk_id}: {preview}")

    return "\n".join(lines)


def serialize_results(results: list[SearchResult]) -> list[dict[str, object]]:
    return [
        {
            "rank": result.rank,
            "method": result.method,
            "score": round(result.score, 6),
            "chunk_id": result.chunk.chunk_id,
            "source": result.chunk.source,
            "title": result.chunk.metadata.get("title", ""),
            "preview": " ".join(result.chunk.text.split())[:240],
            "debug": result.debug,
        }
        for result in results
    ]


def save_trace(record: dict[str, object], trace_path: Path = TRACE_PATH) -> None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def format_results(title: str, results: list[dict[str, object]]) -> str:
    lines = [f"\n=== {title} ==="]
    for row in results:
        lines.append(
            f"{row['rank']:>2}. score={row['score']} source={row['source']} chunk={row['chunk_id']}"
        )
        lines.append(f"    {row['preview']}")
        if row.get("debug"):
            lines.append(f"    debug={row['debug']}")
    return "\n".join(lines)


def build_dense_retriever(
    vector_backend: str,
    chunks: list[Chunk],
    embedder: HashingEmbedder,
    rebuild_milvus: bool,
) -> Retriever:
    if vector_backend == "local":
        return LocalDenseRetriever(chunks, embedder)

    if vector_backend == "milvus":
        return MilvusDenseRetriever(
            chunks=chunks,
            embedder=embedder,
            rebuild=rebuild_milvus,
        )

    raise ValueError(f"Unsupported vector backend: {vector_backend}")


def run_pipeline(
    query: str,
    vector_backend: str = "local",
    strategies: list[str] | None = None,
    bm25_top_k: int = 5,
    dense_top_k: int = 5,
    fusion_top_k: int = 8,
    rerank_top_n: int = 4,
    rebuild_milvus: bool = False,
    save_debug_trace: bool = True,
) -> dict[str, object]:
    timings: dict[str, int] = {}
    t0 = now_ms()

    documents = load_documents()
    chunks = split_into_chunks(documents)
    embedder = HashingEmbedder(dimension=128)
    bm25 = BM25Retriever(chunks)
    dense = build_dense_retriever(vector_backend, chunks, embedder, rebuild_milvus)
    transformer = RuleBasedQueryTransformer()
    reranker = LightweightReranker()
    timings["build_ms"] = now_ms() - t0

    strategies = strategies or ["original", "hyde", "multi_query"]

    t1 = now_ms()
    transformed_queries = transformer.transform(query, strategies=strategies)
    timings["query_transform_ms"] = now_ms() - t1

    all_result_lists: list[list[SearchResult]] = []
    bm25_serialized: list[dict[str, object]] = []
    dense_serialized: list[dict[str, object]] = []

    t2 = now_ms()
    for transformed in transformed_queries:
        bm25_results = bm25.retrieve(transformed.text, top_k=bm25_top_k)
        for result in bm25_results:
            result.debug["query_strategy"] = transformed.strategy
            result.debug["transformed_query"] = transformed.text
        all_result_lists.append(bm25_results)
        bm25_serialized.extend(serialize_results(bm25_results))
    timings["bm25_ms"] = now_ms() - t2

    t3 = now_ms()
    for transformed in transformed_queries:
        dense_results = dense.retrieve(transformed.text, top_k=dense_top_k)
        for result in dense_results:
            result.debug["query_strategy"] = transformed.strategy
            result.debug["transformed_query"] = transformed.text
        all_result_lists.append(dense_results)
        dense_serialized.extend(serialize_results(dense_results))
    timings["dense_ms"] = now_ms() - t3

    t4 = now_ms()
    fused = reciprocal_rank_fusion(all_result_lists, top_k=fusion_top_k, rrf_k=60)
    timings["fusion_ms"] = now_ms() - t4

    t5 = now_ms()
    reranked = reranker.rerank(query, fused, top_n=rerank_top_n)
    timings["rerank_ms"] = now_ms() - t5

    context = build_context(reranked)
    answer = compose_offline_answer(query, reranked)

    record: dict[str, object] = {
        "trace_id": f"day14_{int(time.time() * 1000)}",
        "query": query,
        "vector_backend": vector_backend,
        "num_documents": len(documents),
        "num_chunks": len(chunks),
        "transformed_queries": [item.__dict__ for item in transformed_queries],
        "bm25_results": bm25_serialized,
        "dense_results": dense_serialized,
        "fused_results": serialize_results(fused),
        "reranked_results": serialize_results(reranked),
        "context": context,
        "answer": answer,
        "latency_ms": timings,
        "config": {
            "bm25_top_k": bm25_top_k,
            "dense_top_k": dense_top_k,
            "fusion_top_k": fusion_top_k,
            "rerank_top_n": rerank_top_n,
            "strategies": strategies,
        },
    }

    if save_debug_trace:
        save_trace(record)

    return record


def parse_strategies(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query",
        default="如何将第一周基础 RAG 系统升级为集成 Milvus、混合检索和 Reranker 的生产级 RAG？",
    )
    parser.add_argument("--vector-backend", choices=["local", "milvus"], default="local")
    parser.add_argument("--strategies", default="original,hyde,multi_query")
    parser.add_argument("--bm25-top-k", type=int, default=5)
    parser.add_argument("--dense-top-k", type=int, default=5)
    parser.add_argument("--fusion-top-k", type=int, default=8)
    parser.add_argument("--rerank-top-n", type=int, default=4)
    parser.add_argument("--rebuild-milvus", action="store_true")
    args = parser.parse_args()

    record = run_pipeline(
        query=args.query,
        vector_backend=args.vector_backend,
        strategies=parse_strategies(args.strategies),
        bm25_top_k=args.bm25_top_k,
        dense_top_k=args.dense_top_k,
        fusion_top_k=args.fusion_top_k,
        rerank_top_n=args.rerank_top_n,
        rebuild_milvus=args.rebuild_milvus,
    )

    print(format_results("BM25 sparse retrieval", record["bm25_results"]))
    print(format_results("Dense retrieval", record["dense_results"]))
    print(format_results("Hybrid RRF fusion", record["fused_results"]))
    print(format_results("Reranked final contexts", record["reranked_results"]))
    print("\n=== Context Sent To LLM Demo ===")
    print(record["context"])
    print("\n=== Offline Answer Demo ===")
    print(record["answer"])
    print(f"\nTrace saved to: {TRACE_PATH}")


if __name__ == "__main__":
    main()

