# 用 RAGAs 评估优化前后的系统性能

## 一、实战目标

本实战的目标是搭建一个可复用的 RAG 评估闭环：

1. 准备评估数据集。
2. 运行 baseline RAG。
3. 用 RAGAs 计算核心指标。
4. 分析失败样本。
5. 做一个有明确假设的优化。
6. 再次运行 RAGAs。
7. 输出 baseline vs optimized 对比报告。

## 二、实验设计

### 1. 实验假设

示例假设：

```text
当前 RAG 系统的主要问题是检索结果排序不佳，相关 chunk 没有排在前面。
加入 reranker 后，Context Precision 和 Faithfulness 应该提升；
由于增加了重排步骤，平均延迟会增加。
```

如果你当前系统没有 reranker，可以把优化设为：

```text
baseline: vector search top_k=4
optimized: vector search top_k=10 + reranker top_n=4 + grounded prompt
```

### 2. 控制变量

为了让对比可信，保持这些不变：

- 评估集不变。
- 知识库版本不变。
- embedding 模型不变，除非你正在评估 embedding。
- generator 模型不变，除非你正在评估模型替换。
- RAGAs evaluator LLM 不变。
- prompt 只在你明确评估 prompt 优化时改变。

### 3. 实验记录模板

```yaml
eval_dataset:
  name: rag_eval_v1
  size: 50
  created_at: 2026-06-15

baseline:
  name: baseline_vector_top4
  retriever: vector
  top_k: 4
  reranker: null
  prompt_version: rag_prompt_v1
  generator_model: gpt-4o-mini

optimized:
  name: optimized_top10_rerank4_prompt_v2
  retriever: vector
  top_k: 10
  reranker: bge-reranker-large
  rerank_top_n: 4
  prompt_version: rag_prompt_v2_grounded
  generator_model: gpt-4o-mini

evaluator:
  framework: ragas
  llm: gpt-4o-mini
  embeddings: text-embedding-3-small
```

## 三、环境准备

### 1. 安装依赖

```bash
pip install ragas openai pandas datasets python-dotenv
```

如果你要计算非 LLM 字符串相似类指标：

```bash
pip install rapidfuzz
```

如果你需要记录实验：

```bash
pip install mlflow
```

### 2. 环境变量

```bash
export OPENAI_API_KEY="your_api_key"
```

Windows PowerShell：

```powershell
$env:OPENAI_API_KEY="your_api_key"
```

## 四、评估数据集设计

### 1. CSV 格式

建议先用 CSV，简单直观。

```csv
id,question,reference,reference_contexts,reference_context_ids,question_type,difficulty
q001,"什么是 RAGAs？","RAGAs 是用于评估 LLM/RAG 应用的框架。","RAGAs provides evaluation metrics for LLM applications.","doc_001#chunk_001","definition","easy"
```

注意：

- `reference_contexts` 如果有多个，可以用 JSON 字符串保存。
- `reference_context_ids` 如果有多个，也用 JSON 字符串保存。
- 如果暂时没有 gold context，可以先只维护 `question` 和 `reference`，用 LLM-based Context Recall/Precision 评估。

### 2. JSONL 格式

更推荐工程化使用 JSONL：

```json
{"id":"q001","question":"什么是 RAGAs？","reference":"RAGAs 是用于评估 LLM/RAG 应用的框架。","reference_contexts":["RAGAs provides evaluation metrics for LLM applications."],"reference_context_ids":["doc_001#chunk_001"],"question_type":"definition","difficulty":"easy"}
{"id":"q002","question":"为什么 RAG 需要评估检索和生成两个环节？","reference":"因为答案质量同时受检索证据和生成模型影响，需要分别定位瓶颈。","reference_contexts":["RAG answers depend on retrieved contexts and generation."],"reference_context_ids":["doc_002#chunk_003"],"question_type":"reasoning","difficulty":"medium"}
```

### 3. 样本数量建议

| 阶段 | 样本数 | 目标 |
|---|---:|---|
| smoke test | 5-10 | 验证脚本能跑 |
| dev eval | 30-50 | 快速定位主要问题 |
| regression eval | 100-300 | 每次发版稳定回归 |
| production eval | 500+ | 分场景监控和抽检 |

## 五、让你的 RAG 系统返回评估字段

不管你用 LangChain、LlamaIndex、Haystack、FlashRAG 还是自己写的 RAG pipeline，都建议封装一个统一函数：

```python
def run_rag(question: str, config: dict) -> dict:
    """
    Return fields required by RAG evaluation.
    """
    # 1. retrieve
    # 2. optionally rerank/compress
    # 3. generate
    return {
        "response": "...",
        "retrieved_contexts": ["...", "..."],
        "retrieved_context_ids": ["doc_1#chunk_2", "doc_3#chunk_1"],
        "latency_ms": 1234,
        "input_tokens": 1200,
        "output_tokens": 180,
        "cost_usd": 0.0012,
    }
```

关键点：

- 一定要保存 retrieved_contexts。
- 最好保存 retrieved_context_ids，方便做非 LLM 检索评估。
- 保存原始 retriever score 和 reranker score，方便失败分析。
- 保存 prompt_version、index_version、model_version，方便复现实验。

## 六、RAGAs 指标评估脚本

下面是一个偏工程化的脚本骨架，你可以放到自己的项目里改。

```python
import asyncio
import json
import time
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI
from ragas import EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    NoiseSensitivity,
)


def load_eval_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


async def run_rag(question: str, config: dict) -> dict:
    """
    Replace this with your own RAG pipeline.
    The function must return response and retrieved_contexts.
    """
    start = time.perf_counter()

    # Example placeholder.
    response = "TODO: call your RAG system"
    retrieved_contexts = ["TODO: retrieved chunk 1", "TODO: retrieved chunk 2"]
    retrieved_context_ids = ["doc_todo#chunk_1", "doc_todo#chunk_2"]

    latency_ms = int((time.perf_counter() - start) * 1000)
    return {
        "response": response,
        "retrieved_contexts": retrieved_contexts,
        "retrieved_context_ids": retrieved_context_ids,
        "latency_ms": latency_ms,
        "input_tokens": None,
        "output_tokens": None,
        "cost_usd": None,
    }


async def build_eval_samples(eval_rows: list[dict], rag_config: dict) -> tuple[EvaluationDataset, list[dict]]:
    samples = []
    raw_results = []

    for row in eval_rows:
        rag_result = await run_rag(row["question"], rag_config)
        raw = {
            **row,
            **rag_result,
            "experiment_name": rag_config["name"],
        }
        raw_results.append(raw)

        samples.append(
            SingleTurnSample(
                user_input=row["question"],
                response=rag_result["response"],
                retrieved_contexts=rag_result["retrieved_contexts"],
                reference=row.get("reference"),
                reference_contexts=row.get("reference_contexts"),
                retrieved_context_ids=rag_result.get("retrieved_context_ids"),
                reference_context_ids=row.get("reference_context_ids"),
            )
        )

    return EvaluationDataset(samples=samples), raw_results


async def score_dataset(dataset: EvaluationDataset) -> list[dict]:
    client = AsyncOpenAI()
    evaluator_llm = llm_factory("gpt-4o-mini", client=client)
    evaluator_embeddings = embedding_factory(
        "openai",
        model="text-embedding-3-small",
        client=client,
    )

    metrics = [
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm),
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        NoiseSensitivity(llm=evaluator_llm),
    ]

    scored_rows = []
    for sample in dataset.samples:
        row = {
            "user_input": sample.user_input,
            "response": sample.response,
        }
        for metric in metrics:
            result = await metric.ascore(
                user_input=sample.user_input,
                response=sample.response,
                retrieved_contexts=sample.retrieved_contexts,
                reference=sample.reference,
            )
            row[metric.name] = result.value
            if getattr(result, "reason", None):
                row[f"{metric.name}_reason"] = result.reason
        scored_rows.append(row)

    return scored_rows


async def run_experiment(eval_path: str, rag_config: dict, out_dir: str) -> pd.DataFrame:
    rows = load_eval_jsonl(eval_path)
    dataset, raw_results = await build_eval_samples(rows, rag_config)
    metric_rows = await score_dataset(dataset)

    merged = []
    for raw, metrics in zip(raw_results, metric_rows):
        merged.append({**raw, **metrics})

    df = pd.DataFrame(merged)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(out_dir) / f"{rag_config['name']}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_path}")
    return df


async def main():
    baseline_config = {
        "name": "baseline_vector_top4",
        "top_k": 4,
        "reranker": None,
        "prompt_version": "rag_prompt_v1",
    }

    optimized_config = {
        "name": "optimized_top10_rerank4_prompt_v2",
        "top_k": 10,
        "reranker": "bge-reranker-large",
        "rerank_top_n": 4,
        "prompt_version": "rag_prompt_v2_grounded",
    }

    eval_path = "eval_dataset/rag_eval_v1.jsonl"
    out_dir = "eval_results"

    baseline_df = await run_experiment(eval_path, baseline_config, out_dir)
    optimized_df = await run_experiment(eval_path, optimized_config, out_dir)

    summary = compare_results(baseline_df, optimized_df)
    print(summary)


def compare_results(baseline_df: pd.DataFrame, optimized_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_relevancy",
        "noise_sensitivity",
        "latency_ms",
    ]

    rows = []
    for col in metric_cols:
        if col not in baseline_df.columns or col not in optimized_df.columns:
            continue
        baseline = baseline_df[col].dropna().mean()
        optimized = optimized_df[col].dropna().mean()
        rows.append(
            {
                "metric": col,
                "baseline": baseline,
                "optimized": optimized,
                "delta": optimized - baseline,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    asyncio.run(main())
```

注意：RAGAs 不同版本的 API 可能存在小幅差异。新项目优先参考官方当前文档中的 collections-based API；如果你的本地版本仍使用 legacy API，就按本地版本调整 import 和 score 调用。

## 七、优化策略一：加入 reranker

### 1. 为什么 reranker 常常有效

向量检索擅长召回语义相近内容，但不一定擅长精排。reranker 会重新判断 query 和每个 chunk 的匹配程度，把真正能回答问题的内容排到前面。

预期变化：

| 指标 | 预期 |
|---|---|
| Context Precision | 上升 |
| Faithfulness | 通常上升 |
| Answer Relevancy | 可能上升 |
| Context Recall | 不一定上升 |
| Latency | 上升 |
| Cost | 视 reranker 部署方式而定 |

### 2. 伪代码

```python
def retrieve_with_rerank(question: str, top_k: int = 10, rerank_top_n: int = 4):
    candidates = vector_store.similarity_search(question, k=top_k)
    reranked = reranker.rerank(question, candidates)
    return reranked[:rerank_top_n]
```

### 3. 失败样本检查

如果 reranker 后 Context Precision 没提升：

- 检查 reranker 是否适配中文或你的领域。
- 检查 chunk 是否过长，导致 reranker 难判断。
- 检查 query 是否过短或歧义严重。
- 检查 reference_contexts 标注是否准确。

## 八、优化策略二：改 grounded prompt

### 1. Baseline prompt

```text
请根据以下资料回答问题。

问题：
{question}

资料：
{context}

回答：
```

### 2. Optimized prompt

```text
你是一个严谨的知识库问答助手。请只基于给定资料回答问题。

规则：
1. 如果资料中没有足够信息，请回答“根据已提供资料无法确认”，不要编造。
2. 回答必须直接回应问题，不要加入无关背景。
3. 涉及事实、数字、步骤、定义时，必须能在资料中找到依据。
4. 如果资料存在冲突，请指出冲突，不要自行猜测。

问题：
{question}

资料：
{context}

回答：
```

预期变化：

| 指标 | 预期 |
|---|---|
| Faithfulness | 上升 |
| Noise Sensitivity | 下降 |
| Answer Relevancy | 可能上升 |
| Answer Correctness | 取决于检索是否正确 |
| 拒答率 | 可能上升 |

### 3. 注意副作用

过强的 grounded prompt 可能让模型太保守。你需要同时观察：

- no-answer accuracy 是否提升。
- 可回答问题是否被错误拒答。
- 答案完整性是否下降。

## 九、优化策略三：调整 top_k

### 1. top_k 太小

症状：

- Context Recall 低。
- 多跳问题失败。
- 答案经常说资料不足。

优化：

```text
top_k: 3 -> 5/8/10
```

### 2. top_k 太大

症状：

- Context Precision 低。
- Noise Sensitivity 高。
- Faithfulness 下降。
- token 成本高。

优化：

```text
top_k: 10 -> rerank_top_n: 4
或
top_k: 10 -> context compression -> top_n: 4
```

## 十、结果分析模板

### 1. 总体指标

| metric | baseline | optimized | delta | interpretation |
|---|---:|---:|---:|---|
| context_precision |  |  |  |  |
| context_recall |  |  |  |  |
| faithfulness |  |  |  |  |
| answer_relevancy |  |  |  |  |
| noise_sensitivity |  |  |  |  |
| latency_ms |  |  |  |  |
| cost_usd |  |  |  |  |

### 2. 按问题类型拆分

| question_type | metric | baseline | optimized | delta |
|---|---|---:|---:|---:|
| single_hop | faithfulness |  |  |  |
| multi_hop | context_recall |  |  |  |
| no_answer | answer_correctness |  |  |  |
| comparison | answer_relevancy |  |  |  |

### 3. 失败样本表

| id | question | baseline_error | optimized_error | root_cause | next_action |
|---|---|---|---|---|---|
| q001 |  |  |  | retrieval_miss | improve query rewrite |
| q002 |  |  |  | hallucination | strengthen prompt |

### 4. 根因分类

建议枚举：

```text
retrieval_miss
retrieval_bad_ranking
context_noise
chunk_boundary_issue
outdated_knowledge
generation_hallucination
incomplete_answer
wrong_citation
ambiguous_question
bad_reference
judge_error
```

## 十一、如何判断优化是否真的有效

不要只看平均分。至少满足：

1. 核心目标指标提升，例如 Context Precision +0.10。
2. 高优先级问题类型没有明显下降。
3. 失败样本数量下降，且根因符合预期。
4. latency/cost 增加在可接受范围内。
5. 人工抽检没有发现明显 judge 误判。

## 十二、RAGAs 与其他工具怎么结合

### 1. 与 FlashRAG 结合

FlashRAG 适合做研究式 pipeline 比较。你可以用它：

- 快速试验不同 retriever。
- 比较 reranker、compressor、generator。
- 复现高级 RAG 算法。
- 使用 benchmark 数据集观察通用能力。

然后把 FlashRAG pipeline 的输出转成 RAGAs 需要的字段：

```python
{
    "user_input": item.question,
    "response": item.pred,
    "retrieved_contexts": item.retrieval_result,
    "reference": item.golden_answer,
}
```

### 2. 与 DeepEval 结合

DeepEval 更像测试框架，适合写成 CI 里的测试：

```python
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund.",
    retrieval_context=["All customers are eligible for a 30 day full refund."],
)

evaluate(
    [test_case],
    [
        AnswerRelevancyMetric(threshold=0.7),
        FaithfulnessMetric(threshold=0.7),
    ],
)
```

适合：

- 把 eval 写进测试。
- 设 threshold，低于阈值则失败。
- 和 trace/observability 结合。

### 3. 与 LightEval 区分

LightEval 主要评估 LLM 模型在标准任务上的能力，不是专门评估你的 RAG 应用。

适合：

- 选模型。
- 比较模型在 MMLU、GSM8K、IFEval 等任务上的表现。
- 做基础模型升级前的 benchmark。

不适合单独回答：

```text
我的 RAG 知识库有没有检索到正确 chunk？
我的 reranker 是否让上下文更干净？
我的答案是否忠实于 retrieved_contexts？
```

这些问题还是用 RAGAs/DeepEval 更直接。

## 十三、最终报告模板

```markdown
# RAG 评估报告：baseline_vector_top4 vs optimized_top10_rerank4_prompt_v2

## 实验结论

本轮优化主要针对检索排序和答案忠实性。optimized 版本在 Context Precision、Faithfulness 上提升明显，说明 reranker 和 grounded prompt 对减少噪声、降低幻觉有效。代价是平均延迟增加，需要后续评估是否满足线上 SLA。

## 实验配置

| 项目 | baseline | optimized |
|---|---|---|
| retriever | vector | vector |
| top_k | 4 | 10 |
| reranker | none | bge-reranker-large |
| rerank_top_n | none | 4 |
| prompt | v1 | v2_grounded |
| generator | gpt-4o-mini | gpt-4o-mini |
| eval set | rag_eval_v1 | rag_eval_v1 |

## 指标对比

| metric | baseline | optimized | delta |
|---|---:|---:|---:|
| context_precision | 0.62 | 0.78 | +0.16 |
| context_recall | 0.71 | 0.75 | +0.04 |
| faithfulness | 0.69 | 0.84 | +0.15 |
| answer_relevancy | 0.80 | 0.83 | +0.03 |
| noise_sensitivity | 0.28 | 0.16 | -0.12 |
| latency_ms | 1200 | 1850 | +650 |

## 失败样本归因

| root_cause | baseline_count | optimized_count | note |
|---|---:|---:|---|
| retrieval_miss | 12 | 8 | hybrid/query rewrite 仍有空间 |
| bad_ranking | 10 | 3 | reranker 有效 |
| hallucination | 8 | 3 | grounded prompt 有效 |
| no_answer_error | 4 | 5 | 优化后仍需加强拒答策略 |

## 下一步

1. 对 multi-hop 问题引入 query decomposition。
2. 为 no-answer 样本单独设计拒答评估指标。
3. 对 reranker 做延迟优化或缓存。
4. 将 20 条高价值失败样本加入回归集。
```

## 十四、学习检查清单

- [ ] 我能解释 Context Precision 和 Context Recall 的区别。
- [ ] 我能解释 Faithfulness 和 Answer Correctness 的区别。
- [ ] 我能构造包含 reference_contexts 的评估样本。
- [ ] 我能让 RAG pipeline 输出 retrieved_contexts。
- [ ] 我能用 RAGAs 跑 baseline。
- [ ] 我能按 root cause 分析失败样本。
- [ ] 我能只改变一个变量并跑 optimized。
- [ ] 我能输出指标对比和下一步优化建议。
