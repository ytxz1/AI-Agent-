"""
Local, dependency-light RAG evaluation demo.

This script does not call any LLM API. It demonstrates the full evaluation loop:
1. load a tiny knowledge base
2. run a baseline RAG pipeline
3. run an optimized RAG pipeline
4. compute retrieval and answer-quality proxy metrics
5. save per-sample and summary CSV files

Run from this folder:
    python code_examples/local_rag_eval.py
"""

from __future__ import annotations

import csv
import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", re.UNICODE)


@dataclass
class Document:
    id: str
    title: str
    text: str


@dataclass
class EvalCase:
    id: str
    question: str
    reference: str
    reference_context_ids: list[str]
    must_have_facts: list[str]
    forbidden_claims: list[str]
    question_type: str
    difficulty: str


@dataclass
class RagResult:
    response: str
    retrieved_context_ids: list[str]
    retrieved_contexts: list[str]
    latency_ms: int


def tokenize(text: str) -> list[str]:
    """A tiny tokenizer that works for both English terms and Chinese characters."""
    return [token.lower() for token in TOKEN_RE.findall(text)]


def term_frequency(tokens: Iterable[str]) -> Counter:
    return Counter(tokens)


def cosine_similarity(left: Counter, right: Counter) -> float:
    common = set(left) & set(right)
    numerator = sum(left[t] * right[t] for t in common)
    left_norm = math.sqrt(sum(v * v for v in left.values()))
    right_norm = math.sqrt(sum(v * v for v in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_documents() -> list[Document]:
    return [Document(**row) for row in load_jsonl(DATA_DIR / "knowledge_base.jsonl")]


def load_eval_cases() -> list[EvalCase]:
    return [EvalCase(**row) for row in load_jsonl(DATA_DIR / "rag_eval_examples.jsonl")]


def retrieve_baseline(question: str, docs: list[Document], top_k: int = 3) -> list[Document]:
    """
    Baseline retriever: pure lexical similarity.

    This intentionally behaves like a naive retriever. It can be distracted by
    overlapping words and does not know that terms like '忠实性' and 'Faithfulness'
    belong together unless both appear in the same document.
    """
    query_vec = term_frequency(tokenize(question))
    scored = []
    for doc in docs:
        doc_vec = term_frequency(tokenize(doc.title + " " + doc.text))
        scored.append((cosine_similarity(query_vec, doc_vec), doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


QUERY_EXPANSIONS = {
    "忠实": ["faithfulness", "上下文支持", "幻觉"],
    "幻觉": ["faithfulness", "上下文支持"],
    "召回": ["context recall", "找全", "必要证据"],
    "找全": ["context recall", "召回"],
    "排序": ["context precision", "靠前", "重排"],
    "重排": ["reranker", "context precision"],
    "抗噪": ["noise sensitivity", "干扰", "无关"],
    "噪声": ["noise sensitivity", "干扰", "无关"],
    "测试": ["deepeval", "pytest", "阈值"],
    "benchmark": ["lighteval", "模型能力"],
    "复现": ["flashrag", "pipeline", "组件"],
}


def expand_query(question: str) -> str:
    additions: list[str] = []
    for key, values in QUERY_EXPANSIONS.items():
        if key.lower() in question.lower():
            additions.extend(values)
    return question + " " + " ".join(additions)


def rerank(question: str, candidates: list[Document], top_n: int = 3) -> list[Document]:
    """
    Toy reranker: combines lexical score with domain keyword bonuses.

    A real project would replace this with bge-reranker, Cohere Rerank,
    Jina Reranker, FlashRank, or another cross-encoder reranker.
    """
    expanded_question = expand_query(question)
    query_vec = term_frequency(tokenize(expanded_question))
    scored = []
    for doc in candidates:
        doc_text = doc.title + " " + doc.text
        score = cosine_similarity(query_vec, term_frequency(tokenize(doc_text)))
        if "Faithfulness" in question and "faithfulness" in doc.id:
            score += 0.35
        if "Context Recall" in question and "context_recall" in doc.id:
            score += 0.25
        if "DeepEval" in question and "deepeval" in doc.id:
            score += 0.25
        if "LightEval" in question and "lighteval" in doc.id:
            score += 0.35
        if "FlashRAG" in question and "flashrag" in doc.id:
            score += 0.35
        if "PDF" in question and "prompt" in doc.id:
            score += 0.2
        scored.append((score, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored[:top_n]]


def retrieve_optimized(question: str, docs: list[Document], top_k: int = 8, top_n: int = 3) -> list[Document]:
    """
    Optimized retriever:
    1. query expansion improves recall
    2. larger top_k gives reranker more candidates
    3. reranker improves precision at the final context window
    """
    expanded_question = expand_query(question)
    candidates = retrieve_baseline(expanded_question, docs, top_k=top_k)
    return rerank(question, candidates, top_n=top_n)


def generate_baseline_answer(question: str, contexts: list[Document]) -> str:
    """
    Baseline generator: answers from the top context but is deliberately loose.

    This simulates a common naive RAG prompt:
    'Based on the following documents, answer the question.'
    It often works, but may over-answer no-answer questions.
    """
    top_text = contexts[0].text if contexts else ""
    if "PDF" in question:
        return "当前产品支持 PDF 导出。"
    if "Faithfulness" in question:
        return "Faithfulness 主要评估答案是否相关，也可以看答案是否自然流畅。"
    if "Context Recall" in question:
        return "应该优先优化生成模型，因为 Faithfulness 高说明检索没有问题。"
    return top_text


def generate_optimized_answer(question: str, contexts: list[Document]) -> str:
    """
    Optimized generator: grounded and conservative.

    This simulates a stronger prompt:
    - answer only from context
    - refuse when context is insufficient
    - avoid migrating facts from unrelated documents
    """
    joined = "\n".join(doc.text for doc in contexts)
    ids = {doc.id for doc in contexts}

    if "Faithfulness" in question and "ragas#faithfulness#001" in ids:
        return "Faithfulness 评估生成答案中的声明是否能被检索到的上下文支持，用于衡量答案是否忠实于上下文、是否存在幻觉。"
    if "Context Recall" in question and "rag#diagnosis#001" in ids:
        return "这通常说明答案比较忠实，但检索没有找全必要证据，应优先优化检索召回，例如 query rewrite、hybrid search、top_k、embedding 或 chunk 策略。"
    if "RAGAs" in question and "DeepEval" in question:
        return "RAGAs 更聚焦 RAG/LLM 应用的指标评估和实验闭环；DeepEval 更像面向 LLM 应用的测试框架，风格接近 Pytest，适合写测试并设置阈值。"
    if "LightEval" in question and "lighteval#intro#001" in ids:
        return "不能直接判断。LightEval 主要用于在标准 benchmark 上评估大语言模型能力；具体 RAG 知识库 chunk 是否检索正确，更适合用 RAGAs、DeepEval 或自定义检索指标。"
    if "PDF" in question:
        if "rag#prompt#001" in ids:
            return "根据已提供资料无法确认当前产品是否支持 PDF 导出，不能编造，也不能把其他产品的 PDF 能力迁移到当前产品。"
        return "根据已提供资料无法确认。"
    if "FlashRAG" in question and "flashrag#intro#001" in ids:
        return "FlashRAG 更适合 RAG 研究复现和 pipeline 组件比较，例如比较 retriever、reranker、generator、refiner、compressor 等模块。"

    if not joined:
        return "根据已提供资料无法确认。"
    return joined.split("。")[0] + "。"


def run_rag(case: EvalCase, docs: list[Document], mode: str) -> RagResult:
    start = time.perf_counter()
    if mode == "baseline":
        retrieved = retrieve_baseline(case.question, docs, top_k=3)
        response = generate_baseline_answer(case.question, retrieved)
    elif mode == "optimized":
        retrieved = retrieve_optimized(case.question, docs, top_k=8, top_n=3)
        response = generate_optimized_answer(case.question, retrieved)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    latency_ms = int((time.perf_counter() - start) * 1000)
    return RagResult(
        response=response,
        retrieved_context_ids=[doc.id for doc in retrieved],
        retrieved_contexts=[doc.text for doc in retrieved],
        latency_ms=latency_ms,
    )


def hit_rate_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    return 1.0 if set(retrieved_ids[:k]) & set(gold_ids) else 0.0


def recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    return len(set(retrieved_ids[:k]) & set(gold_ids)) / len(set(gold_ids))


def precision_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(retrieved_ids[:k]) & set(gold_ids)) / k


def mrr(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    gold = set(gold_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in gold:
            return 1.0 / rank
    return 0.0


def context_precision_proxy(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    """
    RAGAs Context Precision is LLM-based or ID-based depending on setup.
    This proxy uses gold document IDs and rewards relevant docs appearing early.
    """
    gold = set(gold_ids)
    if not retrieved_ids or not gold:
        return 0.0
    running_hits = 0
    precision_sum = 0.0
    relevant_seen = 0
    for index, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in gold:
            running_hits += 1
            relevant_seen += 1
            precision_sum += running_hits / index
    return precision_sum / max(1, relevant_seen)


def fact_coverage(response: str, must_have_facts: list[str]) -> float:
    if not must_have_facts:
        return 1.0
    response_tokens = set(tokenize(response))
    covered = 0
    for fact in must_have_facts:
        fact_tokens = set(tokenize(fact))
        if fact_tokens and len(response_tokens & fact_tokens) / len(fact_tokens) >= 0.35:
            covered += 1
    return covered / len(must_have_facts)


def forbidden_claim_rate(response: str, forbidden_claims: list[str]) -> float:
    if not forbidden_claims:
        return 0.0
    hits = 0
    for claim in forbidden_claims:
        normalized_response = response.replace(" ", "")
        normalized_claim = claim.replace(" ", "").replace("一定", "").replace("只", "")
        if "PDF导出" in normalized_claim:
            has_unsafe_pdf_claim = "支持PDF导出" in normalized_response and "无法确认" not in normalized_response
            if has_unsafe_pdf_claim:
                hits += 1
            continue
        if normalized_claim and normalized_claim in normalized_response:
            hits += 1
    return hits / len(forbidden_claims)


def answer_relevancy_proxy(question: str, response: str) -> float:
    return cosine_similarity(term_frequency(tokenize(question)), term_frequency(tokenize(response)))


def faithfulness_proxy(response: str, retrieved_contexts: list[str]) -> float:
    """
    A simple proxy: how much of the response is lexically supported by retrieved contexts.
    Real RAGAs Faithfulness uses an LLM judge to extract and verify claims.
    """
    if not response:
        return 0.0
    response_tokens = set(tokenize(response))
    context_tokens = set(tokenize(" ".join(retrieved_contexts)))
    if not response_tokens:
        return 0.0
    return len(response_tokens & context_tokens) / len(response_tokens)


def evaluate_case(case: EvalCase, result: RagResult) -> dict:
    retrieved_ids = result.retrieved_context_ids
    gold_ids = case.reference_context_ids
    coverage = fact_coverage(result.response, case.must_have_facts)
    forbidden = forbidden_claim_rate(result.response, case.forbidden_claims)
    faithfulness = faithfulness_proxy(result.response, result.retrieved_contexts)
    pass_flag = (
        recall_at_k(retrieved_ids, gold_ids, k=len(retrieved_ids)) >= 0.5
        and coverage >= 0.5
        and forbidden == 0.0
        and faithfulness >= 0.45
    )
    return {
        "id": case.id,
        "question": case.question,
        "question_type": case.question_type,
        "difficulty": case.difficulty,
        "reference": case.reference,
        "response": result.response,
        "retrieved_context_ids": "|".join(retrieved_ids),
        "reference_context_ids": "|".join(gold_ids),
        "hit_rate_at_3": hit_rate_at_k(retrieved_ids, gold_ids, 3),
        "recall_at_3": recall_at_k(retrieved_ids, gold_ids, 3),
        "precision_at_3": precision_at_k(retrieved_ids, gold_ids, 3),
        "mrr": mrr(retrieved_ids, gold_ids),
        "context_precision_proxy": context_precision_proxy(retrieved_ids, gold_ids),
        "context_recall_proxy": recall_at_k(retrieved_ids, gold_ids, len(retrieved_ids)),
        "faithfulness_proxy": faithfulness,
        "answer_relevancy_proxy": answer_relevancy_proxy(case.question, result.response),
        "fact_coverage": coverage,
        "forbidden_claim_rate": forbidden,
        "latency_ms": result.latency_ms,
        "pass": pass_flag,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], mode: str) -> dict:
    metric_names = [
        "hit_rate_at_3",
        "recall_at_3",
        "precision_at_3",
        "mrr",
        "context_precision_proxy",
        "context_recall_proxy",
        "faithfulness_proxy",
        "answer_relevancy_proxy",
        "fact_coverage",
        "forbidden_claim_rate",
        "latency_ms",
    ]
    summary = {"experiment": mode, "sample_count": len(rows)}
    for metric in metric_names:
        summary[metric] = sum(float(row[metric]) for row in rows) / len(rows)
    summary["pass_rate"] = sum(1 for row in rows if row["pass"]) / len(rows)
    return summary


def compare_summaries(baseline: dict, optimized: dict) -> list[dict]:
    rows = []
    for metric, baseline_value in baseline.items():
        if metric in {"experiment", "sample_count"}:
            continue
        optimized_value = optimized[metric]
        rows.append(
            {
                "metric": metric,
                "baseline": round(float(baseline_value), 4),
                "optimized": round(float(optimized_value), 4),
                "delta": round(float(optimized_value) - float(baseline_value), 4),
            }
        )
    return rows


def main() -> None:
    docs = load_documents()
    cases = load_eval_cases()

    all_summaries = []
    all_rows_by_mode: dict[str, list[dict]] = {}

    for mode in ["baseline", "optimized"]:
        rows = []
        for case in cases:
            result = run_rag(case, docs, mode=mode)
            rows.append(evaluate_case(case, result))
        all_rows_by_mode[mode] = rows
        write_csv(OUT_DIR / f"{mode}_per_sample.csv", rows)
        all_summaries.append(summarize(rows, mode))

    write_csv(OUT_DIR / "summary.csv", all_summaries)
    comparison = compare_summaries(all_summaries[0], all_summaries[1])
    write_csv(OUT_DIR / "comparison.csv", comparison)

    print("Saved outputs:")
    print(f"- {OUT_DIR / 'baseline_per_sample.csv'}")
    print(f"- {OUT_DIR / 'optimized_per_sample.csv'}")
    print(f"- {OUT_DIR / 'summary.csv'}")
    print(f"- {OUT_DIR / 'comparison.csv'}")
    print("\nMetric comparison:")
    for row in comparison:
        print(f"{row['metric']}: {row['baseline']} -> {row['optimized']} ({row['delta']:+.4f})")


if __name__ == "__main__":
    main()
