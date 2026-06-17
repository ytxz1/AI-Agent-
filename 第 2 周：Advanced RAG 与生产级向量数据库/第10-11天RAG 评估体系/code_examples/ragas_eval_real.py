"""
Real RAGAs evaluation script.

This file shows how the same baseline/optimized outputs can be evaluated with
RAGAs LLM-based metrics. It requires API access and the current RAGAs package.

Install:
    pip install ragas langchain-openai openai

PowerShell:
    $env:OPENAI_API_KEY="your-key"

Run:
    python code_examples/ragas_eval_real.py

Notes:
    RAGAs evolves quickly. The current official quickstart uses:
    - EvaluationDataset.from_list(dataset)
    - evaluate(dataset=evaluation_dataset, metrics=[...], llm=evaluator_llm)
    - LangchainLLMWrapper(ChatOpenAI(...))
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, FactualCorrectness, LLMContextRecall

from local_rag_eval import DATA_DIR, load_documents, load_eval_cases, run_rag


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs"


def build_ragas_dataset(mode: str) -> EvaluationDataset:
    docs = load_documents()
    cases = load_eval_cases()
    rows = []
    for case in cases:
        result = run_rag(case, docs, mode=mode)
        rows.append(
            {
                "user_input": case.question,
                "retrieved_contexts": result.retrieved_contexts,
                "response": result.response,
                "reference": case.reference,
            }
        )
    return EvaluationDataset.from_list(rows)


def save_result_json(path: Path, result) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(result, "to_pandas"):
        result.to_pandas().to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(result), f, ensure_ascii=False, indent=2)


def run_ragas_eval(mode: str):
    evaluation_dataset = build_ragas_dataset(mode)
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=[
            LLMContextRecall(),
            Faithfulness(),
            FactualCorrectness(),
        ],
        llm=evaluator_llm,
    )
    save_result_json(OUT_DIR / f"ragas_{mode}.json", result)
    return result


def main() -> None:
    baseline = run_ragas_eval("baseline")
    optimized = run_ragas_eval("optimized")
    print("Baseline RAGAs result:")
    print(baseline)
    print("\nOptimized RAGAs result:")
    print(optimized)
    print(f"\nSaved RAGAs outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
