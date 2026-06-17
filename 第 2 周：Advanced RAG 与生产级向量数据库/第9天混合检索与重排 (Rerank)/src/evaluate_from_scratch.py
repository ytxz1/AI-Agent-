"""
Evaluate the from-scratch hybrid retrieval pipeline.

Run:
    python src/evaluate_from_scratch.py
"""

from __future__ import annotations

import json
from pathlib import Path

from hybrid_rerank_from_scratch import (
    BM25Retriever,
    DATA_DIR,
    LightweightReranker,
    TfidfEmbeddingLikeRetriever,
    load_documents,
    reciprocal_rank_fusion,
    split_into_chunks,
)


ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ROOT / "eval_queries.jsonl"
REPORT_PATH = ROOT / "outputs" / "from_scratch_evaluation_report.md"


def load_eval_queries() -> list[dict]:
    rows = []
    with EVAL_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def hit_at_k(results, relevant_sources: set[str], k: int) -> int:
    for result in results[:k]:
        if result.chunk.source in relevant_sources:
            return 1
    return 0


def first_relevant_rank(results, relevant_sources: set[str]) -> int | None:
    for rank, result in enumerate(results, start=1):
        if result.chunk.source in relevant_sources:
            return rank
    return None


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate():
    documents = load_documents(DATA_DIR)
    chunks = split_into_chunks(documents)

    bm25 = BM25Retriever(chunks)
    dense = TfidfEmbeddingLikeRetriever(chunks)
    reranker = LightweightReranker()

    queries = load_eval_queries()
    rows = []

    methods = {
        "BM25": [],
        "Embedding-like": [],
        "Hybrid RRF": [],
        "Hybrid RRF + Rerank": [],
    }

    for item in queries:
        query = item["query"]
        relevant_sources = set(item["relevant_sources"])

        bm25_results = bm25.retrieve(query, top_k=5)
        dense_results = dense.retrieve(query, top_k=5)
        hybrid_results = reciprocal_rank_fusion([bm25_results, dense_results], top_k=8)
        reranked_results = reranker.rerank(query, hybrid_results, top_n=5)

        result_map = {
            "BM25": bm25_results,
            "Embedding-like": dense_results,
            "Hybrid RRF": hybrid_results,
            "Hybrid RRF + Rerank": reranked_results,
        }

        for method_name, results in result_map.items():
            rank = first_relevant_rank(results, relevant_sources)
            methods[method_name].append(
                {
                    "hit@1": hit_at_k(results, relevant_sources, 1),
                    "hit@3": hit_at_k(results, relevant_sources, 3),
                    "hit@5": hit_at_k(results, relevant_sources, 5),
                    "mrr": 1 / rank if rank else 0.0,
                }
            )

        rows.append(
            {
                "query": query,
                "relevant_sources": sorted(relevant_sources),
                "bm25_top1": bm25_results[0].chunk.source if bm25_results else "",
                "dense_top1": dense_results[0].chunk.source if dense_results else "",
                "hybrid_top1": hybrid_results[0].chunk.source if hybrid_results else "",
                "rerank_top1": reranked_results[0].chunk.source if reranked_results else "",
            }
        )

    return methods, rows


def write_report(methods: dict, rows: list[dict]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# From-Scratch Hybrid Retrieval Evaluation",
        "",
        "## Summary",
        "",
        "| Method | Hit@1 | Hit@3 | Hit@5 | MRR |",
        "|---|---:|---:|---:|---:|",
    ]

    for method_name, metrics in methods.items():
        lines.append(
            "| {method} | {hit1:.3f} | {hit3:.3f} | {hit5:.3f} | {mrr:.3f} |".format(
                method=method_name,
                hit1=mean([row["hit@1"] for row in metrics]),
                hit3=mean([row["hit@3"] for row in metrics]),
                hit5=mean([row["hit@5"] for row in metrics]),
                mrr=mean([row["mrr"] for row in metrics]),
            )
        )

    lines.extend(
        [
            "",
            "## Per Query Top-1",
            "",
            "| Query | BM25 Top1 | Embedding-like Top1 | Hybrid Top1 | Rerank Top1 |",
            "|---|---|---|---|---|",
        ]
    )

    for row in rows:
        lines.append(
            "| {query} | {bm25} | {dense} | {hybrid} | {rerank} |".format(
                query=row["query"].replace("|", "/"),
                bm25=row["bm25_top1"],
                dense=row["dense_top1"],
                hybrid=row["hybrid_top1"],
                rerank=row["rerank_top1"],
            )
        )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    methods, rows = evaluate()
    write_report(methods, rows)
    print(f"Evaluation report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
