# 第 8 天 Query Transformation 实施计划

这份计划用于今天动手实现 HyDE、Multi-Query、Query Decomposition，并把它们接入 RAG 检索流程。

---

## 1. 今日目标

今天的目标不是只跑通一个 Demo，而是形成一个可复用的查询改写层。

最终你应该得到：

1. 一个 baseline 原始查询检索流程。
2. 一个 HyDE 查询改写器。
3. 一个 Multi-Query 查询改写器。
4. 一个 Query Decomposition 查询拆解器。
5. 一个多路检索结果融合函数。
6. 一套可对比不同策略效果的实验脚本。
7. 一份实验结果报告。

---

## 2. 推荐执行顺序

### Step 1：确认知识库与基础检索

目标：先有 baseline，否则无法判断 Query Transformation 是否真的提升效果。

要做：

1. 准备 `data/raw/` 文档。
2. 使用 LlamaIndex 加载文档。
3. 构建 VectorStoreIndex。
4. 使用原始 query 检索 top_k。
5. 打印 score、source、text preview。

输出：

```text
outputs/baseline_results.jsonl
```

验收：

```text
输入：什么是 Query Transformation？
输出：能返回 3-5 个相关片段。
```

---

### Step 2：定义通用数据结构

目标：让所有改写策略返回统一结构。

建议结构：

```python
from dataclasses import dataclass, field

@dataclass
class TransformedQuery:
    text: str
    strategy: str
    weight: float = 1.0
    metadata: dict = field(default_factory=dict)
```

为什么要这么做：

1. 后续可以区分 query 来自 HyDE 还是 Multi-Query。
2. 可以给不同策略设置权重。
3. 可以记录 source query、prompt version、model、latency。
4. 便于日志和评估。

---

### Step 3：实现 Original Transformer

目标：把原始 query 也当作一种 transformer，方便统一处理。

```python
class OriginalQueryTransformer:
    def transform(self, query: str) -> list[TransformedQuery]:
        return [
            TransformedQuery(
                text=query,
                strategy="original",
                weight=1.0,
                metadata={"source": "user"},
            )
        ]
```

验收：

```text
transform("什么是 HyDE？")
返回一个 strategy=original 的 TransformedQuery。
```

---

### Step 4：实现 HyDE Transformer

目标：生成 hypothetical document，用更丰富的语义文本做向量检索。

关键设计：

1. `include_original=True`：建议同时保留原始 query。
2. HyDE 文本只用于检索，不作为事实上下文。
3. prompt 限制不要编造具体数字、人名、日期。
4. 输出长度控制在 120-250 中文字。

实现要点：

```python
class HyDETransformer:
    def __init__(self, llm, include_original: bool = True):
        self.llm = llm
        self.include_original = include_original

    def transform(self, query: str) -> list[TransformedQuery]:
        results = []

        if self.include_original:
            results.append(
                TransformedQuery(
                    text=query,
                    strategy="original",
                    weight=1.0,
                )
            )

        hypothetical_doc = self.llm.complete(
            HYDE_PROMPT.format(query=query)
        )

        results.append(
            TransformedQuery(
                text=str(hypothetical_doc),
                strategy="hyde",
                weight=0.8,
                metadata={"source_query": query},
            )
        )

        return results
```

验收：

```text
输入：Milvus 的 HNSW 索引适合什么场景？
输出：
1. original query
2. 一段包含 HNSW、图索引、召回率、内存、近似最近邻等关键词的 hypothetical document
```

---

### Step 5：实现 Multi-Query Transformer

目标：从多个表达角度提升召回覆盖。

关键设计：

1. 默认生成 4 个改写 query。
2. 输出 JSON 数组。
3. 必须保留原始意图。
4. 解析失败时降级为原始 query。
5. 去重、去空、限制长度。

实现要点：

```python
class MultiQueryTransformer:
    def __init__(self, llm, num_queries: int = 4, include_original: bool = True):
        self.llm = llm
        self.num_queries = num_queries
        self.include_original = include_original

    def transform(self, query: str) -> list[TransformedQuery]:
        rewritten = self._generate_queries(query)
        cleaned = self._clean_queries([query] + rewritten if self.include_original else rewritten)

        return [
            TransformedQuery(
                text=item,
                strategy="original" if item == query else "multi_query",
                weight=1.0 if item == query else 0.9,
                metadata={"source_query": query},
            )
            for item in cleaned
        ]
```

验收：

```text
输入：如何提升 RAG 的召回率？
输出 4-5 个不同角度的查询，并且没有偏离“提升 RAG 召回率”这个核心意图。
```

---

### Step 6：实现 Query Decomposition Transformer

目标：把复杂问题拆解为多个可独立检索的子问题。

适合问题：

```text
Milvus 和 Qdrant 在索引类型、过滤能力、分布式扩展和 Python 生态上有什么区别？
```

期望输出：

```json
[
  "Milvus 支持哪些向量索引类型？",
  "Qdrant 支持哪些向量索引类型？",
  "Milvus 的 metadata filter 能力如何？",
  "Qdrant 的 payload filter 能力如何？",
  "Milvus 的分布式扩展能力如何？",
  "Qdrant 的 Python SDK 生态如何？"
]
```

验收：

1. 简单问题只返回 1 个子问题。
2. 复杂问题返回 2-6 个子问题。
3. 每个子问题只包含一个检索意图。

---

### Step 7：实现多路检索

目标：每个 transformed query 都能独立检索。

```python
def retrieve_transformed_queries(transformed_queries, retriever):
    all_results = []

    for transformed_query in transformed_queries:
        nodes = retriever.retrieve(transformed_query.text)

        for rank, node in enumerate(nodes, start=1):
            all_results.append({
                "transformed_query": transformed_query,
                "rank": rank,
                "node": node,
                "score": node.score,
            })

    return all_results
```

验收：

1. 能看到每个 query 的检索结果。
2. 能追踪每个 node 是被哪个 query 命中的。
3. 同一个 node 被多次命中时不会丢失来源信息。

---

### Step 8：实现融合与去重

目标：把多路结果合并成最终候选集。

最小版本：

1. 按 node id 去重。
2. 保留最高 score。
3. 记录命中过该 node 的所有 query。

推荐版本：

1. 使用 RRF。
2. 再接第 9 天的 reranker。

验收：

```text
输入：多个 query 的 top_k 检索结果
输出：去重后的 top_k 文档，并带 fusion_score
```

---

### Step 9：实现策略配置

目标：通过参数切换策略。

推荐模式：

```text
original
hyde
multi_query
decomposition
hyde_multi_query
compare
```

命令示例：

```text
python demo.py --strategy original --query "什么是 Query Transformation？"
python demo.py --strategy hyde --query "HyDE 为什么有效？"
python demo.py --strategy multi_query --query "如何提升 RAG 召回？"
python demo.py --strategy compare --query "Multi-Query 和 HyDE 有什么区别？"
```

验收：

1. 不改代码，只改参数即可切换策略。
2. compare 模式可以一次输出多种策略结果。

---

### Step 10：记录日志

目标：所有改写和检索都可复盘。

必须记录：

1. 原始 query。
2. 改写策略。
3. 改写后的 query。
4. LLM model。
5. prompt version。
6. LLM latency。
7. 检索 top_k。
8. 命中文档 id。
9. score。
10. fusion score。

日志文件：

```text
outputs/transformed_queries.jsonl
outputs/retrieval_results.jsonl
outputs/fused_results.jsonl
```

---

### Step 11：评估

目标：判断策略是否有效，而不是凭感觉。

准备问题集：

```text
eval/questions.jsonl
```

格式：

```json
{
  "query_id": "q001",
  "query": "HyDE 为什么能提升向量检索召回？",
  "relevant_doc_ids": ["doc_hyde_001", "doc_query_transform_002"]
}
```

跑实验：

```text
original
hyde
multi_query
decomposition
```

评估指标：

1. Recall@3
2. Recall@5
3. MRR
4. Average latency
5. LLM calls
6. Token cost

验收：

```text
outputs/compare_report.md 中有表格和结论。
```

---

## 3. 今日代码任务清单

- [ ] 创建 `src/query_transformers.py`
- [ ] 创建 `src/retriever.py`
- [ ] 创建 `src/fusion.py`
- [ ] 创建 `src/pipeline.py`
- [ ] 创建 `demo.py`
- [ ] 实现 `TransformedQuery`
- [ ] 实现 `OriginalQueryTransformer`
- [ ] 实现 `HyDETransformer`
- [ ] 实现 `MultiQueryTransformer`
- [ ] 实现 `DecompositionTransformer`
- [ ] 实现 JSON 解析兜底
- [ ] 实现 query 去重
- [ ] 实现多路检索
- [ ] 实现 RRF 融合
- [ ] 实现日志保存
- [ ] 实现 compare 模式
- [ ] 准备 10 个测试问题
- [ ] 生成对比报告

---

## 4. 常见错误与排查

### 错误一：HyDE 生成内容太像最终答案

表现：

```text
HyDE 文本被直接拿去生成答案。
```

修正：

1. HyDE 文本只用于 embedding 和 retrieval。
2. 最终回答上下文只能来自真实文档。

### 错误二：Multi-Query 偏离原始问题

表现：

```text
原问题问 RAG 召回，改写 query 跑去问 prompt engineering。
```

修正：

1. prompt 强调保留原始意图。
2. 对改写 query 和原始 query 做相似度过滤。
3. 减少 num_queries。

### 错误三：多路检索结果噪声太多

表现：

```text
召回数量变多，但最终上下文更乱。
```

修正：

1. 使用 RRF。
2. 第 9 天接入 reranker。
3. 限制每个 query 的 top_k。
4. 对 query 设置权重。

### 错误四：延迟过高

表现：

```text
每次查询都要等多次 LLM 和多次检索。
```

修正：

1. 缓存 query transformation 结果。
2. 并发检索多 query。
3. 减少生成 query 数量。
4. 只对低置信问题触发复杂改写。

---

## 5. 今日最终交付物

最低交付：

1. `README.md`
2. `implementation_plan.md`
3. HyDE 实现
4. Multi-Query 实现
5. 一个可运行 Demo

推荐交付：

1. 完整 `src/` 模块。
2. 多策略 compare 模式。
3. RRF 融合。
4. 日志文件。
5. `outputs/compare_report.md`

优秀交付：

1. 加入 rerank 前置接口，为第 9 天衔接。
2. 加入 prompt version。
3. 加入缓存。
4. 加入评估指标。
5. 加入 10-20 个测试问题的对比结论。

---

## 6. 学习重点排序

如果时间有限，按这个优先级做：

1. Original baseline。
2. HyDE。
3. Multi-Query。
4. RRF 融合。
5. Compare report。
6. Decomposition。
7. 缓存和日志。
8. 自动评估。

原因：

1. 没有 baseline 就无法比较。
2. HyDE 和 Multi-Query 是今天核心。
3. RRF 是多路检索结果融合的关键。
4. Decomposition 很重要，但可以在复杂问题场景中再深化。

---

## 7. 明天衔接第 9 天 Rerank

第 8 天会扩大召回，第 9 天要解决扩大召回后带来的噪声。

衔接方式：

```text
Original / HyDE / Multi-Query
  -> 多路检索
  -> RRF 初步融合
  -> Reranker 精排
  -> 最终上下文
  -> Answer
```

今天最好保留每个候选文档的来源 query，这样明天分析 rerank 效果时能知道：

1. 哪个改写策略贡献了最终高分文档。
2. 哪个策略引入了噪声。
3. Rerank 是否成功把噪声排下去。
