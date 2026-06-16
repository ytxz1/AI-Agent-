# 第 8 天：Query Transformation 查询改写策略

> 学习主题：Query Transformation  
> 核心目标：实现 HyDE、Multi-Query 等查询改写策略，并理解它们如何提升 RAG 检索质量。  
> 参考资料：<https://developers.llamaindex.ai/python/framework/module_guides/querying/query_transforms/root/>  
> 说明：该链接在 2026-06-15 访问时返回 404，可能是 LlamaIndex 文档路径已迁移。本文按 LlamaIndex Query Transform 的设计思想、HyDE 常用接口与生产级 RAG 实践进行整理。

---

## 1. 今天你要解决什么问题

普通 RAG 的检索入口通常只有一个原始用户问题：

```text
用户问题 -> embedding -> vector search -> rerank -> LLM answer
```

这个流程简单，但有几个常见问题：

1. 用户问题太短：例如“怎么优化召回？”缺少上下文。
2. 用户问题太口语化：例如“那个向量库慢咋办？”不容易命中文档里的专业表达。
3. 用户问题包含多个意图：例如“Milvus 和 Qdrant 的索引、过滤、扩展性分别怎么样？”
4. 用户问题和知识库措辞不一致：例如用户问“降噪”，文档写的是“query rewriting / retrieval enhancement”。
5. 用户问题本身没有足够语义信息：embedding 后的查询向量不够稳定。

Query Transformation 的目标就是：在真正检索前，对原始 query 做一次或多次结构化改写，让检索器看到更适合检索的查询表达。

改写后的流程变成：

```text
用户问题
  -> Query Transformation
  -> 一个或多个改写查询
  -> 检索
  -> 融合 / 去重 / 重排
  -> LLM answer
```

---

## 2. 今日学习产出

完成今天之后，你应该拥有下面这些能力：

1. 能解释 Query Transformation 在 RAG 中的位置和价值。
2. 能实现 HyDE 查询改写。
3. 能实现 Multi-Query 查询扩展。
4. 能实现 Query Decomposition 思路，用于复杂问题拆解。
5. 能将多个改写策略接入统一检索管线。
6. 能对比不同策略的召回质量、成本、延迟和适用场景。
7. 能写出一套可复用的 Query Transformer 抽象接口。
8. 能记录实验结果，为第 9 天混合检索与 Rerank 做准备。

---

## 3. 推荐目录结构

建议在本目录下逐步形成以下结构：

```text
第8天Query Transformation/
  README.md
  notes/
    01_query_transformation_concepts.md
    02_hyde.md
    03_multi_query.md
    04_query_decomposition.md
    05_evaluation_notes.md
  src/
    config.py
    data_loader.py
    index_builder.py
    query_transformers.py
    retriever.py
    pipeline.py
    run_hyde_demo.py
    run_multi_query_demo.py
    run_compare_demo.py
  data/
    raw/
    processed/
  outputs/
    transformed_queries.jsonl
    retrieval_results.jsonl
    compare_report.md
  tests/
    test_query_transformers.py
```

如果今天只做最小可运行版本，可以先保留：

```text
第8天Query Transformation/
  README.md
  query_transformers.py
  demo.py
  outputs/
```

---

## 4. 概念地图

### 4.1 Query Transformation 是什么

Query Transformation 是 RAG 中“检索前”的查询处理层。它不是回答问题，而是把用户问题改造成更适合检索的形式。

常见改写方式：

| 策略 | 输入 | 输出 | 适合场景 |
| --- | --- | --- | --- |
| Query Rewrite | 原始问题 | 一个更清晰的问题 | 口语化、含糊问题 |
| HyDE | 原始问题 | 一段假设性答案 / 文档 | 原问题太短、语义稀疏 |
| Multi-Query | 原始问题 | 多个不同角度的问题 | 用户问题可能有多种表达 |
| Query Decomposition | 复杂问题 | 多个子问题 | 多跳问题、比较问题、聚合问题 |
| Step-back Query | 具体问题 | 更抽象的上位问题 | 需要背景知识的问题 |
| Metadata-aware Rewrite | 原始问题 + schema | 带过滤条件的查询 | 文档带时间、类型、作者等元数据 |

### 4.2 它和 Rerank 的区别

Query Transformation 发生在检索前，决定“拿什么去搜”。

Rerank 发生在检索后，决定“搜到的结果怎么排序”。

```text
Query Transformation: 改变查询
Retriever: 找候选文档
Rerank: 重排候选文档
Synthesizer: 生成答案
```

### 4.3 它和 Prompt Engineering 的区别

Prompt Engineering 通常优化最终回答质量。

Query Transformation 优化检索输入，目标是提升召回和上下文质量。

如果检索到的上下文本身错了，最终回答 prompt 写得再好也很难稳定。

---

## 5. 核心策略一：HyDE

### 5.1 HyDE 是什么

HyDE 全称 Hypothetical Document Embeddings。

它的思路是：

1. 不直接拿用户问题做 embedding。
2. 先让 LLM 根据问题生成一段“假设性答案”或“假设性文档”。
3. 再对这段假设性文档做 embedding。
4. 用这个 embedding 去检索真实文档。

例子：

```text
原始问题：
Milvus 的 IVF_FLAT 和 HNSW 有什么区别？

HyDE 生成的假设性文档：
Milvus 中 IVF_FLAT 是基于倒排文件的近似最近邻索引，适合大规模向量检索，
需要设置 nlist 和 nprobe；HNSW 是基于分层导航小世界图的索引，通常召回率高、
查询速度快，但内存消耗更大...

检索器实际拿这段文本去做 embedding。
```

### 5.2 为什么 HyDE 有效

用户问题通常短，语义信息少。假设性答案更长，包含更多领域词、同义表达和上下文结构，因此 embedding 更容易靠近知识库里的相关文档。

它尤其适合：

1. 专业概念查询。
2. 简短问题。
3. 文档中术语表达固定，但用户提问比较随意的场景。
4. 需要召回解释型文档的场景。

不适合：

1. 事实非常敏感的问题，例如金额、政策日期、实时状态。
2. 用户问题已经很长且准确。
3. LLM 容易编造大量错误术语的领域。
4. 对延迟和成本极度敏感的检索链路。

### 5.3 HyDE 的风险

HyDE 的假设性文档可能包含幻觉。注意：HyDE 不是把假设性文档当事实，而是用它辅助检索。

风险控制方式：

1. Prompt 中明确要求“生成用于检索的假设性文档，不要编造具体数字、日期、人名”。
2. 只用 HyDE 文本做检索，不把 HyDE 文本直接放入最终回答上下文。
3. 对 HyDE 检索结果再做 rerank。
4. 保留原始 query 的检索结果，与 HyDE 结果合并。
5. 对事实敏感问题降低 HyDE 权重。

### 5.4 HyDE Prompt 模板

```text
你是一个 RAG 查询改写器。
请根据用户问题生成一段“可能出现在知识库中的说明性文档”。
这段文本只用于向量检索，不会直接作为最终答案。

要求：
1. 使用清晰、专业、可检索的表达。
2. 覆盖问题中的关键概念、同义词和相关术语。
3. 不要编造具体数字、日期、实验结果或不存在的专有名词。
4. 如果问题缺少上下文，生成通用但相关的说明。
5. 输出 1 段，长度控制在 120-250 中文字。

用户问题：
{query}
```

### 5.5 HyDE 实现伪代码

```python
class HyDETransformer:
    def __init__(self, llm):
        self.llm = llm

    def transform(self, query: str) -> list[str]:
        hypothetical_doc = self.llm.complete(
            prompt=HYDE_PROMPT.format(query=query)
        )
        return [hypothetical_doc]
```

### 5.6 HyDE 检索融合方式

建议不要只用 HyDE 替代原始查询，而是同时保留两路结果：

```text
原始 query -> 检索 top_k=5
HyDE doc  -> 检索 top_k=5
合并去重 -> rerank top_k=5 -> answer
```

简单合并规则：

1. 用 node id / doc id 去重。
2. 原始查询命中的文档加权更高。
3. HyDE 命中的文档提供补充召回。
4. 最终交给 reranker 排序。

---

## 6. 核心策略二：Multi-Query

### 6.1 Multi-Query 是什么

Multi-Query 是让 LLM 从多个角度重写用户问题，生成多个检索查询。

例子：

```text
原始问题：
怎么提升 RAG 的召回率？

Multi-Query 输出：
1. RAG 系统中提高向量检索召回率的方法有哪些？
2. 如何通过查询改写、混合检索和 rerank 改善 RAG 检索效果？
3. 向量数据库召回不足时应该如何调整 embedding、chunk 和 top_k？
4. RAG 中 query transformation 对召回率有什么帮助？
```

然后每个 query 分别检索，最后合并结果。

### 6.2 Multi-Query 为什么有效

同一个问题可以有不同表达方式。单一 query 的 embedding 可能错过部分相关文档，多 query 可以扩大语义覆盖。

它适合：

1. 知识库表达风格不统一。
2. 用户问题比较宽泛。
3. 问题包含多个角度。
4. 需要提高召回率，而可以接受额外检索成本。

不适合：

1. top_k 已经很大且召回足够。
2. 延迟要求极低。
3. 知识库噪声很大，多 query 会放大噪声。
4. 用户问题需要严格过滤条件，改写可能丢失约束。

### 6.3 Multi-Query Prompt 模板

```text
你是一个 RAG 查询改写器。
请把用户问题改写成 {num_queries} 个不同角度的检索查询。

要求：
1. 每个查询都必须保留用户原始意图。
2. 不要回答问题，只输出查询。
3. 查询之间要有明显差异，覆盖不同术语、同义表达或子角度。
4. 不要引入原问题没有的硬性事实条件。
5. 输出 JSON 数组，数组元素是字符串。

用户问题：
{query}
```

### 6.4 Multi-Query 实现伪代码

```python
import json

class MultiQueryTransformer:
    def __init__(self, llm, num_queries: int = 4):
        self.llm = llm
        self.num_queries = num_queries

    def transform(self, query: str) -> list[str]:
        response = self.llm.complete(
            prompt=MULTI_QUERY_PROMPT.format(
                query=query,
                num_queries=self.num_queries,
            )
        )
        queries = json.loads(response)
        return [query] + queries
```

生产实现要增加：

1. JSON 解析失败兜底。
2. 去除空字符串。
3. 去重。
4. 限制最大长度。
5. 保留原始 query。
6. 日志记录每次改写结果。

### 6.5 Multi-Query 结果融合

最简单方式：

```text
query_0 -> top_k=5
query_1 -> top_k=5
query_2 -> top_k=5
query_3 -> top_k=5
合并去重 -> rerank -> answer
```

更好的方式是 Reciprocal Rank Fusion，简称 RRF。

RRF 公式：

```text
score(doc) = sum(1 / (k + rank_i(doc)))
```

其中：

1. `rank_i(doc)` 是文档在第 i 个查询结果中的排名。
2. `k` 是平滑常数，常用 60。
3. 文档被多个 query 检索到，会得到更高分。

RRF 的好处是不用强依赖不同检索器的原始分数是否可比。

---

## 7. 核心策略三：Query Decomposition

### 7.1 Query Decomposition 是什么

Query Decomposition 是把复杂问题拆成多个子问题。

例子：

```text
原始问题：
Milvus 和 Qdrant 在索引类型、过滤能力、分布式扩展和 Python 生态上有什么区别？

拆解：
1. Milvus 支持哪些向量索引类型？
2. Qdrant 支持哪些向量索引类型？
3. Milvus 的 metadata filter 能力如何？
4. Qdrant 的 payload filter 能力如何？
5. Milvus 的分布式扩展方式是什么？
6. Qdrant 的集群和扩展能力是什么？
7. Milvus 和 Qdrant 的 Python SDK 生态分别如何？
```

### 7.2 它适合什么问题

适合：

1. 多跳问题。
2. 横向比较问题。
3. “分别说明”“从 A/B/C 角度分析”的问题。
4. 需要多个证据片段才能回答的问题。

不适合：

1. 简单事实查询。
2. 用户只需要一句话答案。
3. 子问题拆太多会导致成本过高的场景。

### 7.3 Decomposition Prompt 模板

```text
你是一个 RAG 查询拆解器。
请把用户问题拆解成若干个可以独立检索的子问题。

要求：
1. 子问题必须共同覆盖原始问题。
2. 每个子问题只问一个清晰的信息点。
3. 不要回答问题。
4. 如果原问题很简单，只返回原问题本身。
5. 输出 JSON 数组。
6. 子问题数量控制在 1-6 个。

用户问题：
{query}
```

---

## 8. 统一抽象设计

为了让不同改写策略可插拔，建议定义统一接口。

```python
from dataclasses import dataclass
from typing import Protocol

@dataclass
class TransformedQuery:
    text: str
    strategy: str
    weight: float = 1.0
    metadata: dict | None = None

class QueryTransformer(Protocol):
    def transform(self, query: str) -> list[TransformedQuery]:
        ...
```

不同策略返回同一种结构：

```python
class OriginalQueryTransformer:
    def transform(self, query: str) -> list[TransformedQuery]:
        return [
            TransformedQuery(
                text=query,
                strategy="original",
                weight=1.0,
            )
        ]
```

HyDE：

```python
class HyDETransformer:
    def __init__(self, llm):
        self.llm = llm

    def transform(self, query: str) -> list[TransformedQuery]:
        doc = self.llm.complete(HYDE_PROMPT.format(query=query))
        return [
            TransformedQuery(
                text=doc,
                strategy="hyde",
                weight=0.8,
                metadata={"source_query": query},
            )
        ]
```

Multi-Query：

```python
class MultiQueryTransformer:
    def __init__(self, llm, num_queries: int = 4):
        self.llm = llm
        self.num_queries = num_queries

    def transform(self, query: str) -> list[TransformedQuery]:
        queries = generate_queries_with_llm(
            llm=self.llm,
            query=query,
            num_queries=self.num_queries,
        )
        return [
            TransformedQuery(
                text=q,
                strategy="multi_query",
                weight=0.9,
                metadata={"source_query": query, "index": i},
            )
            for i, q in enumerate(queries)
        ]
```

组合器：

```python
class QueryTransformationPipeline:
    def __init__(self, transformers: list[QueryTransformer]):
        self.transformers = transformers

    def transform(self, query: str) -> list[TransformedQuery]:
        results = []
        for transformer in self.transformers:
            results.extend(transformer.transform(query))
        return deduplicate_transformed_queries(results)
```

---

## 9. 推荐实现路线

### 阶段一：基础环境

目标：跑通一个最小 RAG 检索流程。

任务：

1. 准备一批课程相关文档，建议使用前 1-7 天的学习资料或自己整理的 RAG 笔记。
2. 用 LlamaIndex 构建基础索引。
3. 实现原始 query 检索。
4. 输出 top_k 文档片段、score、metadata。

验收标准：

1. 输入一个问题，能返回 top_k 文档。
2. 能看到每个命中文档的文本片段。
3. 能记录检索结果到 `outputs/retrieval_results.jsonl`。

### 阶段二：实现 HyDE

目标：用假设性文档提升语义召回。

任务：

1. 编写 HyDE prompt。
2. 调用 LLM 生成 hypothetical document。
3. 使用 hypothetical document 检索。
4. 同时保留 original query 检索结果。
5. 合并、去重、输出对比。

验收标准：

1. 能输出 HyDE 生成文本。
2. 能分别看到 original 和 hyde 的 top_k。
3. 能比较两路结果是否召回不同文档。

### 阶段三：实现 Multi-Query

目标：生成多个角度的检索查询，扩大召回覆盖。

任务：

1. 编写 Multi-Query prompt。
2. 让 LLM 输出 JSON 数组。
3. 解析并清洗查询列表。
4. 每个 query 分别检索。
5. 用去重或 RRF 融合结果。

验收标准：

1. 每个输入问题能生成 3-5 个不同查询。
2. JSON 解析失败时能兜底。
3. 能输出每个 query 的检索结果。
4. 能生成融合后的最终候选文档列表。

### 阶段四：实现 Query Decomposition

目标：处理复杂比较、多跳和多约束问题。

任务：

1. 编写 Decomposition prompt。
2. 生成子问题列表。
3. 子问题分别检索。
4. 保留子问题与证据片段的映射。
5. 最终回答时按子问题组织上下文。

验收标准：

1. 复杂问题能拆成 2-6 个子问题。
2. 简单问题不会被过度拆解。
3. 输出中能追踪每个证据来自哪个子问题。

### 阶段五：统一 Pipeline

目标：把所有策略变成可配置能力。

任务：

1. 实现 `QueryTransformer` 抽象。
2. 实现 `HyDETransformer`。
3. 实现 `MultiQueryTransformer`。
4. 实现 `DecompositionTransformer`。
5. 实现 `QueryTransformationPipeline`。
6. 配置不同模式：
   - `original`
   - `hyde`
   - `multi_query`
   - `hyde_multi_query`
   - `decomposition`
7. 输出每个模式的耗时、token 成本和召回结果。

验收标准：

1. 一行配置切换策略。
2. 每个策略可单独测试。
3. 每次运行产生结构化日志。

### 阶段六：评估和复盘

目标：知道什么时候该用哪种策略。

任务：

1. 准备 10-20 个测试问题。
2. 人工标注每个问题的相关文档。
3. 分别跑：
   - original
   - hyde
   - multi_query
   - hyde + original
   - multi_query + rerank
4. 记录：
   - Recall@5
   - MRR
   - 命中文档数量
   - 平均延迟
   - LLM 调用次数
   - token 成本
5. 总结策略适用边界。

验收标准：

1. 有一份 `outputs/compare_report.md`。
2. 能说清楚哪类问题 HyDE 更好。
3. 能说清楚哪类问题 Multi-Query 更好。
4. 能说清楚哪类问题不该做复杂改写。

---

## 10. LlamaIndex 实现参考

下面是偏工程化的实现思路。不同版本的 LlamaIndex API 可能存在差异，建议以当前安装版本为准。

### 10.1 基础检索流程

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./data/raw").load_data()
index = VectorStoreIndex.from_documents(documents)
retriever = index.as_retriever(similarity_top_k=5)

nodes = retriever.retrieve("什么是 Query Transformation？")
for node in nodes:
    print(node.score, node.node.get_content()[:200])
```

### 10.2 使用 LlamaIndex HyDE Query Transform 的思路

在部分 LlamaIndex 版本中，HyDE 查询改写通常围绕 `HyDEQueryTransform` 和 query engine 包装器使用。

概念形态大致如下：

```python
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine

query_engine = index.as_query_engine()
hyde = HyDEQueryTransform(include_original=True)
hyde_query_engine = TransformQueryEngine(query_engine, hyde)

response = hyde_query_engine.query("Milvus 的 HNSW 索引适合什么场景？")
print(response)
```

如果当前版本的 API 不一致，可以自己实现 HyDETransformer，再把输出文本交给 retriever 检索。

### 10.3 手写 Multi-Query 更推荐

Multi-Query 在不同框架里的抽象差异较大，建议先手写，这样更利于理解和调试。

```python
def retrieve_with_multi_query(query: str, transformer, retriever, top_k: int = 5):
    transformed_queries = transformer.transform(query)
    all_results = []

    for transformed in transformed_queries:
        results = retriever.retrieve(transformed.text)
        for rank, node in enumerate(results, start=1):
            all_results.append({
                "query": transformed.text,
                "strategy": transformed.strategy,
                "rank": rank,
                "node": node,
                "score": node.score,
            })

    return fuse_results_with_rrf(all_results, top_k=top_k)
```

---

## 11. RRF 融合实现

```python
from collections import defaultdict

def fuse_results_with_rrf(results, top_k: int = 5, k: int = 60):
    scores = defaultdict(float)
    payload = {}

    for item in results:
        node = item["node"]
        node_id = node.node.node_id
        rank = item["rank"]

        scores[node_id] += 1.0 / (k + rank)
        payload[node_id] = node

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        {
            "node_id": node_id,
            "rrf_score": score,
            "node": payload[node_id],
        }
        for node_id, score in ranked[:top_k]
    ]
```

改进方向：

1. 加入策略权重。
2. 加入原始 similarity score。
3. 加入 reranker 分数。
4. 记录每个文档被哪些 query 命中。

---

## 12. 策略选择建议

| 问题类型 | 推荐策略 | 原因 |
| --- | --- | --- |
| 短问题、概念问题 | HyDE | 补充语义上下文 |
| 宽泛问题 | Multi-Query | 覆盖多个表达角度 |
| 比较问题 | Decomposition + Multi-Query | 分维度检索证据 |
| 精确事实问题 | Original + Metadata Filter | 避免 LLM 改写引入幻觉 |
| 多跳推理问题 | Decomposition | 拆成可检索子问题 |
| 召回不足 | Multi-Query + RRF | 扩大候选集合 |
| 噪声过多 | Original + Rerank | 控制召回扩张 |
| 延迟敏感 | Original 或轻量 Rewrite | 减少 LLM 调用 |

---

## 13. 实验问题集

建议用下面的问题做第一轮实验：

1. 什么是 Query Transformation？
2. HyDE 为什么能提升向量检索召回？
3. Multi-Query 和 HyDE 有什么区别？
4. RAG 中如何处理用户问题过短的问题？
5. 如何提升 RAG 的召回率？
6. Milvus 的 HNSW 和 IVF_FLAT 有什么区别？
7. 为什么混合检索通常比纯向量检索更稳？
8. Rerank 在 RAG 管线中解决什么问题？
9. 如果用户问题包含多个子问题，应该如何检索？
10. 如何评估 Query Transformation 是否真的有效？
11. Query Transformation 会不会带来幻觉？
12. 什么情况下不应该使用 HyDE？
13. 如何用 RRF 合并多路检索结果？
14. Chunk 大小会影响查询改写效果吗？
15. 生产环境中如何记录查询改写日志？

---

## 14. 评估指标

### 14.1 Recall@K

衡量 top_k 结果中是否包含相关文档。

```text
Recall@K = 命中的相关文档数 / 所有相关文档数
```

Query Transformation 最核心的目标通常是提升 Recall。

### 14.2 MRR

MRR 衡量第一个相关文档出现得有多靠前。

```text
MRR = 1 / 第一个相关文档的排名
```

如果改写后相关文档虽然出现了，但排得很靠后，MRR 不一定好。

### 14.3 Latency

记录每个策略的总耗时：

```text
总耗时 = 查询改写耗时 + 多路检索耗时 + 融合耗时 + rerank 耗时
```

### 14.4 Cost

记录：

1. LLM 调用次数。
2. 输入 token。
3. 输出 token。
4. embedding 次数。
5. 检索次数。

Multi-Query 的成本通常和生成查询数量线性相关。

### 14.5 Faithfulness

Query Transformation 不直接回答问题，但它会影响最终上下文。如果改写引入错误方向，最终回答也可能偏。

需要检查：

1. 最终答案是否由真实检索上下文支持。
2. HyDE 生成文本是否被误当作证据。
3. Multi-Query 是否偏离原始意图。

---

## 15. 日志格式建议

保存到 `outputs/transformed_queries.jsonl`：

```json
{
  "query_id": "q_001",
  "original_query": "HyDE 为什么能提升检索？",
  "strategy": "hyde",
  "transformed_query": "HyDE 是一种通过生成假设性文档来增强向量检索语义表示的方法...",
  "created_at": "2026-06-15T10:00:00+08:00",
  "model": "gpt-4.1-mini",
  "latency_ms": 850
}
```

保存到 `outputs/retrieval_results.jsonl`：

```json
{
  "query_id": "q_001",
  "strategy": "multi_query",
  "transformed_query_index": 2,
  "rank": 1,
  "node_id": "node_abc",
  "score": 0.83,
  "text_preview": "Query transformation improves retrieval by...",
  "metadata": {
    "source": "rag_notes.md"
  }
}
```

保存到 `outputs/compare_report.md`：

```markdown
| Query | Strategy | Recall@5 | MRR | Latency(ms) | Notes |
| --- | --- | --- | --- | --- | --- |
| HyDE 为什么有效？ | original | 0.50 | 0.50 | 120 | 漏掉理论说明 |
| HyDE 为什么有效？ | hyde | 1.00 | 1.00 | 980 | 召回更完整 |
```

---

## 16. 生产级注意事项

### 16.1 缓存

Query Transformation 通常会调用 LLM，必须考虑缓存。

缓存 key：

```text
hash(strategy + model + prompt_version + original_query)
```

缓存内容：

1. 改写后的 query。
2. LLM 原始响应。
3. prompt version。
4. latency。
5. token usage。

### 16.2 Prompt Versioning

不要直接覆盖 prompt。建议记录：

```python
PROMPT_VERSION = "hyde_v1_2026_06_15"
```

否则后续评估结果无法复现。

### 16.3 失败兜底

LLM 改写失败时：

1. 返回原始 query。
2. 记录错误日志。
3. 不阻断主检索流程。

### 16.4 防止意图漂移

Multi-Query 容易为了“多样性”偏离原问题。控制方法：

1. prompt 中强调保留原始意图。
2. 对改写结果做相似度过滤。
3. 查询数量不要过多。
4. 对每个改写 query 设置最大长度。

### 16.5 权限和安全

不要让 query 改写绕过权限过滤。

正确顺序：

```text
Query Transformation
  -> Retriever with user permission filters
  -> Rerank
  -> Answer
```

用户权限、租户 id、文档范围等 metadata filter 必须始终保留。

### 16.6 可观测性

必须能回答：

1. 用户原始问题是什么？
2. 系统改写成了什么？
3. 每个改写 query 命中了哪些文档？
4. 哪些文档进入最终上下文？
5. 最终答案引用了哪些证据？
6. 哪个策略带来了最多命中？

---

## 17. 今日详细时间安排

### 第 1 小时：理解概念

任务：

1. 阅读 Query Transformation 相关资料。
2. 梳理 HyDE、Multi-Query、Decomposition 的区别。
3. 画出 RAG 中 Query Transformation 的位置。

产出：

1. 一页概念笔记。
2. 一张流程图。

### 第 2 小时：跑通基础检索

任务：

1. 准备小型知识库。
2. 构建 LlamaIndex 索引。
3. 原始 query 检索 top_k。
4. 打印检索结果。

产出：

1. `index_builder.py`
2. `retriever.py`
3. 一组 baseline 检索结果。

### 第 3 小时：实现 HyDE

任务：

1. 编写 HyDE prompt。
2. 生成 hypothetical document。
3. 用 HyDE 文本检索。
4. 与原始 query 对比。

产出：

1. `HyDETransformer`
2. `run_hyde_demo.py`
3. HyDE 对比结果。

### 第 4 小时：实现 Multi-Query

任务：

1. 编写 Multi-Query prompt。
2. JSON 输出解析。
3. 多路检索。
4. 结果合并去重。

产出：

1. `MultiQueryTransformer`
2. `run_multi_query_demo.py`
3. 多路检索日志。

### 第 5 小时：实现 RRF 融合

任务：

1. 为每个 query 的结果记录 rank。
2. 实现 RRF。
3. 输出融合后 top_k。
4. 对比简单去重和 RRF 排序。

产出：

1. `fuse_results_with_rrf`
2. 融合结果日志。

### 第 6 小时：实现 Decomposition

任务：

1. 编写拆解 prompt。
2. 复杂问题拆成子问题。
3. 子问题分别检索。
4. 保存子问题和证据映射。

产出：

1. `DecompositionTransformer`
2. 复杂问题检索结果。

### 第 7 小时：统一 Pipeline

任务：

1. 抽象 `QueryTransformer`。
2. 实现 pipeline 配置。
3. 统一日志结构。
4. 命令行参数选择策略。

产出：

1. `pipeline.py`
2. `run_compare_demo.py`
3. 可切换策略的 Demo。

### 第 8 小时：评估与复盘

任务：

1. 准备 10-15 个测试问题。
2. 跑不同策略。
3. 记录 Recall@5、MRR、延迟。
4. 总结策略适用边界。

产出：

1. `outputs/compare_report.md`
2. 今日复盘。

---

## 18. 最小可运行 Demo 设计

### 18.1 输入

```text
python demo.py --strategy hyde --query "HyDE 为什么能提升检索召回？"
python demo.py --strategy multi_query --query "如何提升 RAG 的召回率？"
python demo.py --strategy compare --query "Milvus 和 Qdrant 的索引有什么区别？"
```

### 18.2 输出

```text
Original Query:
如何提升 RAG 的召回率？

Transformed Queries:
[original] 如何提升 RAG 的召回率？
[multi_query] RAG 系统中提高向量检索召回率的方法有哪些？
[multi_query] 如何通过查询改写、混合检索和 rerank 改善 RAG 检索效果？
[multi_query] 向量数据库召回不足时应该如何调整 embedding、chunk 和 top_k？

Retrieved Nodes:
1. score=0.87 source=rag_retrieval.md
2. score=0.82 source=query_transform.md
3. score=0.78 source=rerank.md
```

---

## 19. 今日复盘模板

```markdown
# 第 8 天复盘：Query Transformation

## 今天完成了什么

- [ ] 理解 Query Transformation 的作用
- [ ] 实现 HyDE
- [ ] 实现 Multi-Query
- [ ] 实现 Query Decomposition
- [ ] 实现 RRF 融合
- [ ] 完成策略对比实验

## 哪个策略效果最好

结论：

原因：

## 哪些问题 HyDE 更适合

例子：

## 哪些问题 Multi-Query 更适合

例子：

## 发现的问题

1.
2.
3.

## 明天和 Rerank 如何衔接

第 9 天可以重点验证：

1. Multi-Query 扩召回后，Rerank 是否能压噪声。
2. HyDE 召回的候选文档是否能被 Rerank 正确排序。
3. Hybrid Search + Query Transformation 是否比单一向量检索更稳。
```

---

## 20. 今日最终验收清单

- [ ] 能说清楚 Query Transformation 解决的核心问题。
- [ ] 能实现原始 query baseline。
- [ ] 能实现 HyDE 改写。
- [ ] 能实现 Multi-Query 改写。
- [ ] 能实现 Query Decomposition。
- [ ] 能合并多路检索结果。
- [ ] 能记录每个改写 query 与检索结果。
- [ ] 能对比 original、HyDE、Multi-Query 的效果。
- [ ] 能总结不同策略的适用场景。
- [ ] 能为第 9 天 Rerank 准备候选文档集。

---

## 21. 建议结论

今天最重要的不是“用了多少复杂策略”，而是建立一个清晰的判断框架：

1. Query 太短：优先 HyDE。
2. Query 太宽：优先 Multi-Query。
3. Query 太复杂：优先 Decomposition。
4. 召回变多但噪声增加：第 9 天用 Rerank 解决。
5. 成本太高：减少改写数量、做缓存、只对低置信查询触发改写。

Query Transformation 是高级 RAG 的入口层。它决定检索器看到什么问题，也决定后续 Rerank 和答案生成的上限。
