"""
Evaluate the Day 14 RAG v2 upgrade demo.

Run:
    python src/evaluate_rag_v2.py
"""

from __future__ import annotations

import json
from pathlib import Path

from rag_v2_upgrade_demo import (
    BM25Retriever,
    HashingEmbedder,
    LightweightReranker,
    LocalDenseRetriever,
    reciprocal_rank_fusion,
    load_documents,
    run_pipeline,
    split_into_chunks,
)


ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / "data" / "eval" / "eval_queries.jsonl"
REPORT_PATH = ROOT / "outputs" / "rag_v2_evaluation_report.md"


def load_eval_queries() -> list[dict]:
    rows = []
    with EVAL_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def hit_at_k(results, relevant_sources: set[str], k: int) -> int:
    return int(any(result.chunk.source in relevant_sources for result in results[:k]))


def recall_at_k(results, relevant_sources: set[str], k: int) -> float:
    if not relevant_sources:
        return 0.0
    found = {result.chunk.source for result in results[:k] if result.chunk.source in relevant_sources}
    return len(found) / len(relevant_sources)


def first_relevant_rank(results, relevant_sources: set[str]) -> int | None:
    for rank, result in enumerate(results, start=1):
        if result.chunk.source in relevant_sources:
            return rank
    return None


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_methods() -> tuple[dict[str, list[dict[str, float]]], list[dict[str, str]]]:
    documents = load_documents()
    chunks = split_into_chunks(documents)
    embedder = HashingEmbedder(dimension=128)

    bm25 = BM25Retriever(chunks)
    dense = LocalDenseRetriever(chunks, embedder)
    reranker = LightweightReranker()

    methods: dict[str, list[dict[str, float]]] = {
        "bm25_only": [],
        "dense_only": [],
        "hybrid_rrf": [],
        "hybrid_rerank": [],
        "transform_hybrid_rerank": [],
    }
    per_query_rows: list[dict[str, str]] = []

    for item in load_eval_queries():
        query = item["query"]
        relevant_sources = set(item["relevant_sources"])

        bm25_results = bm25.retrieve(query, top_k=5)
        dense_results = dense.retrieve(query, top_k=5)
        hybrid_results = reciprocal_rank_fusion([bm25_results, dense_results], top_k=8)
        reranked_results = reranker.rerank(query, hybrid_results, top_n=5)

        transformed_record = run_pipeline(
            query=query,
            vector_backend="local",
            strategies=["original", "hyde", "multi_query"],
            save_debug_trace=False,
        )
        transform_results = rows_to_result_sources(transformed_record["reranked_results"])

        result_map = {
            "bm25_only": result_sources(bm25_results),
            "dense_only": result_sources(dense_results),
            "hybrid_rrf": result_sources(hybrid_results),
            "hybrid_rerank": result_sources(reranked_results),
            "transform_hybrid_rerank": transform_results,
        }

        for method_name, source_list in result_map.items():
            rank = first_relevant_rank_from_sources(source_list, relevant_sources)
            methods[method_name].append(
                {
                    "hit@1": hit_at_k_sources(source_list, relevant_sources, 1),
                    "hit@3": hit_at_k_sources(source_list, relevant_sources, 3),
                    "hit@5": hit_at_k_sources(source_list, relevant_sources, 5),
                    "recall@5": recall_at_k_sources(source_list, relevant_sources, 5),
                    "mrr": 1 / rank if rank else 0.0,
                }
            )

        per_query_rows.append(
            {
                "query_id": item["query_id"],
                "query": query,
                "relevant": ", ".join(sorted(relevant_sources)),
                "bm25_top1": first_or_empty(result_map["bm25_only"]),
                "dense_top1": first_or_empty(result_map["dense_only"]),
                "hybrid_top1": first_or_empty(result_map["hybrid_rrf"]),
                "rerank_top1": first_or_empty(result_map["hybrid_rerank"]),
                "transform_top1": first_or_empty(result_map["transform_hybrid_rerank"]),
            }
        )

    return methods, per_query_rows


def result_sources(results) -> list[str]:
    return [result.chunk.source for result in results]


def rows_to_result_sources(rows: list[dict]) -> list[str]:
    return [str(row["source"]) for row in rows]


def first_relevant_rank_from_sources(sources: list[str], relevant_sources: set[str]) -> int | None:
    for rank, source in enumerate(sources, start=1):
        if source in relevant_sources:
            return rank
    return None


def hit_at_k_sources(sources: list[str], relevant_sources: set[str], k: int) -> int:
    return int(any(source in relevant_sources for source in sources[:k]))


def recall_at_k_sources(sources: list[str], relevant_sources: set[str], k: int) -> float:
    if not relevant_sources:
        return 0.0
    found = {source for source in sources[:k] if source in relevant_sources}
    return len(found) / len(relevant_sources)


def first_or_empty(values: list[str]) -> str:
    return values[0] if values else ""


def write_report(methods: dict[str, list[dict[str, float]]], rows: list[dict[str, str]]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# RAG v2 Upgrade Evaluation Report",
        "",
        "## Summary",
        "",
        "| Method | Hit@1 | Hit@3 | Hit@5 | Recall@5 | MRR |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for method_name, metrics in methods.items():
        lines.append(
            "| {method} | {hit1:.3f} | {hit3:.3f} | {hit5:.3f} | {recall5:.3f} | {mrr:.3f} |".format(
                method=method_name,
                hit1=mean([row["hit@1"] for row in metrics]),
                hit3=mean([row["hit@3"] for row in metrics]),
                hit5=mean([row["hit@5"] for row in metrics]),
                recall5=mean([row["recall@5"] for row in metrics]),
                mrr=mean([row["mrr"] for row in metrics]),
            )
        )

    lines.extend(
        [
            "",
            "## Per Query Top-1",
            "",
            "| Query ID | Query | Relevant | BM25 | Dense | Hybrid | Rerank | Transform+Hybrid+Rerank |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )

    for row in rows:
        lines.append(
            "| {query_id} | {query} | {relevant} | {bm25_top1} | {dense_top1} | {hybrid_top1} | {rerank_top1} | {transform_top1} |".format(
                **{key: str(value).replace("|", "/") for key, value in row.items()}
            )
        )

    lines.extend(
        [
            "",
            "## How To Read This Report",
            "",
            "- `bm25_only` tests exact lexical retrieval.",
            "- `dense_only` tests local dense-vector-shaped retrieval.",
            "- `hybrid_rrf` tests sparse + dense fusion.",
            "- `hybrid_rerank` tests second-stage ranking after fusion.",
            "- `transform_hybrid_rerank` tests query transformation before hybrid retrieval.",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    methods, rows = evaluate_methods()
    write_report(methods, rows)
    print(f"Evaluation report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()

