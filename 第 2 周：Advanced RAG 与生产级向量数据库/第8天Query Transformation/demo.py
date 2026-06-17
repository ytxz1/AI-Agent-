from __future__ import annotations

import argparse
from pathlib import Path

from src.logger import write_jsonl
from src.mock_llm import RuleBasedLLM
from src.pipeline import RetrievalPipeline
from src.query_transformers import (
    DecompositionTransformer,
    HyDETransformer,
    MultiQueryTransformer,
    OriginalQueryTransformer,
    QueryTransformationPipeline,
)
from src.simple_retriever import SimpleTfidfRetriever
from src.toy_corpus import load_toy_documents


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"


def build_transformer(strategy: str, llm: RuleBasedLLM):
    if strategy == "original":
        return QueryTransformationPipeline([OriginalQueryTransformer()])

    if strategy == "hyde":
        return QueryTransformationPipeline([HyDETransformer(llm, include_original=True)])

    if strategy == "multi_query":
        return QueryTransformationPipeline([MultiQueryTransformer(llm, num_queries=4, include_original=True)])

    if strategy == "decomposition":
        return QueryTransformationPipeline([DecompositionTransformer(llm, include_original=True)])

    if strategy == "hyde_multi_query":
        return QueryTransformationPipeline(
            [
                HyDETransformer(llm, include_original=True),
                MultiQueryTransformer(llm, num_queries=4, include_original=False),
            ]
        )

    raise ValueError(f"Unsupported strategy: {strategy}")


def run_once(strategy: str, query: str) -> dict:
    documents = load_toy_documents()
    llm = RuleBasedLLM()
    transformer = build_transformer(strategy, llm)
    retriever = SimpleTfidfRetriever(documents)
    pipeline = RetrievalPipeline(retriever, per_query_top_k=4, final_top_k=5)

    transformed_queries = transformer.transform(query)
    raw_results, fused_results = pipeline.retrieve(transformed_queries)

    print_section("Original Query")
    print(query)

    print_section("Transformed Queries")
    for index, item in enumerate(transformed_queries, start=1):
        print(f"{index}. [{item.strategy}] weight={item.weight}")
        print(f"   {item.text}")

    print_section("Raw Retrieval Results")
    for item in raw_results:
        print(
            f"[{item.strategy}] rank={item.rank} score={item.score:.4f} "
            f"doc={item.doc.doc_id} title={item.doc.title}"
        )

    print_section("Fused Results")
    for rank, item in enumerate(fused_results, start=1):
        print(
            f"{rank}. fusion={item.fusion_score:.4f} best={item.best_score:.4f} "
            f"doc={item.doc.doc_id} strategies={','.join(item.strategies)}"
        )
        print(f"   title: {item.doc.title}")
        print(f"   preview: {item.doc.text[:120]}")

    save_outputs(strategy, query, transformed_queries, raw_results, fused_results)

    return {
        "strategy": strategy,
        "query": query,
        "transformed_count": len(transformed_queries),
        "raw_result_count": len(raw_results),
        "fused_result_count": len(fused_results),
    }


def save_outputs(strategy, query, transformed_queries, raw_results, fused_results) -> None:
    safe_strategy = strategy.replace("/", "_")

    write_jsonl(
        OUTPUT_DIR / f"{safe_strategy}_transformed_queries.jsonl",
        [
            {
                "original_query": query,
                "strategy": item.strategy,
                "text": item.text,
                "weight": item.weight,
                "metadata": item.metadata,
            }
            for item in transformed_queries
        ],
    )

    write_jsonl(
        OUTPUT_DIR / f"{safe_strategy}_raw_results.jsonl",
        [
            {
                "strategy": item.strategy,
                "query_text": item.query_text,
                "rank": item.rank,
                "score": item.score,
                "doc_id": item.doc.doc_id,
                "title": item.doc.title,
                "metadata": item.doc.metadata,
                "text_preview": item.doc.text[:160],
            }
            for item in raw_results
        ],
    )

    write_jsonl(
        OUTPUT_DIR / f"{safe_strategy}_fused_results.jsonl",
        [
            {
                "rank": rank,
                "doc_id": item.doc.doc_id,
                "title": item.doc.title,
                "fusion_score": item.fusion_score,
                "best_score": item.best_score,
                "strategies": item.strategies,
                "matched_queries": item.matched_queries,
                "ranks": item.ranks,
                "text_preview": item.doc.text[:160],
            }
            for rank, item in enumerate(fused_results, start=1)
        ],
    )


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query Transformation demo: HyDE, Multi-Query, Decomposition, RRF.")
    parser.add_argument(
        "--strategy",
        choices=["original", "hyde", "multi_query", "decomposition", "hyde_multi_query", "compare"],
        default="compare",
    )
    parser.add_argument("--query", default="如何提升 RAG 的召回率？")
    args = parser.parse_args()

    if args.strategy == "compare":
        summaries = []
        for strategy in ["original", "hyde", "multi_query", "decomposition", "hyde_multi_query"]:
            print_section(f"Running Strategy: {strategy}")
            summaries.append(run_once(strategy, args.query))

        print_section("Compare Summary")
        for item in summaries:
            print(
                f"{item['strategy']}: transformed={item['transformed_count']} "
                f"raw={item['raw_result_count']} fused={item['fused_result_count']}"
            )
    else:
        run_once(args.strategy, args.query)


if __name__ == "__main__":
    main()

