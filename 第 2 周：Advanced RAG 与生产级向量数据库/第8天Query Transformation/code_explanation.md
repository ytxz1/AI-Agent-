# Query Transformation 代码实现详解

本文专门解释本目录中的代码实现。目标是让你不仅能运行 Demo，还能真正理解 HyDE、Multi-Query、Query Decomposition、RRF 融合在代码里分别负责什么。

---

## 1. 代码目录

当前实现位于：

```text
第8天Query Transformation/
  demo.py
  src/
    __init__.py
    models.py
    toy_corpus.py
    simple_retriever.py
    mock_llm.py
    query_transformers.py
    fusion.py
    pipeline.py
    logger.py
  outputs/
    *.jsonl
```

每个文件的作用：

| 文件 | 作用 |
| --- | --- |
| `demo.py` | 命令行入口，负责选择策略、运行检索、打印结果、保存日志 |
| `src/models.py` | 定义核心数据结构，例如 Document、TransformedQuery、RetrievedNode、FusedResult |
| `src/toy_corpus.py` | 内置一个小型 RAG 知识库，方便无需外部数据也能运行 |
| `src/simple_retriever.py` | 用纯 Python 实现一个简化版 TF-IDF 检索器 |
| `src/mock_llm.py` | 本地规则版 LLM，用来模拟 HyDE、Multi-Query、Decomposition 的输出 |
| `src/query_transformers.py` | Query Transformation 核心实现 |
| `src/fusion.py` | RRF 多路检索结果融合 |
| `src/pipeline.py` | 把 retriever 和 fusion 串起来 |
| `src/logger.py` | 把改写 query、原始检索结果、融合结果保存成 JSONL |

---

## 2. 如何运行

进入第 8 天目录：

```powershell
cd "D:\vscode项目\AI Agent 开发工程师学习路线图（工程落地版）\第 2 周：Advanced RAG 与生产级向量数据库\第8天Query Transformation"
```

运行全部策略对比：

```powershell
python demo.py --strategy compare --query "如何提升 RAG 的召回率？"
```

只运行 HyDE：

```powershell
python demo.py --strategy hyde --query "如何提升 RAG 的召回率？"
```

只运行 Multi-Query：

```powershell
python demo.py --strategy multi_query --query "HyDE 为什么能提升检索召回？"
```

只运行查询拆解：

```powershell
python demo.py --strategy decomposition --query "Milvus 和 Qdrant 在索引类型、过滤能力、分布式扩展和 Python 生态上有什么区别？"
```

运行 HyDE + Multi-Query：

```powershell
python demo.py --strategy hyde_multi_query --query "如何提升 RAG 的召回率？"
```

运行后会生成：

```text
outputs/original_transformed_queries.jsonl
outputs/original_raw_results.jsonl
outputs/original_fused_results.jsonl
outputs/hyde_transformed_queries.jsonl
outputs/hyde_raw_results.jsonl
outputs/hyde_fused_results.jsonl
...
```

---

## 3. 整体流程

代码实现的是一个完整但轻量的 RAG 检索前处理流程：

```text
用户问题
  -> Query Transformer
  -> TransformedQuery 列表
  -> SimpleTfidfRetriever 多路检索
  -> RetrievedNode 列表
  -> RRF 融合
  -> FusedResult 列表
  -> 保存 JSONL 日志
```

以 HyDE 为例：

```text
用户问题：
如何提升 RAG 的召回率？

HyDE 生成：
提升 RAG 召回率通常需要从查询改写、向量表示、chunk 切分、top_k 设置、混合检索和 rerank 等方面优化...

检索器实际检索两路：
1. 原始问题
2. HyDE 假设性文档

最后用 RRF 合并两路结果。
```

以 Multi-Query 为例：

```text
用户问题：
如何提升 RAG 的召回率？

Multi-Query 生成：
1. RAG 系统中提高向量检索召回率的方法有哪些？
2. 如何通过 HyDE、Multi-Query 和查询改写改善 RAG 检索效果？
3. 向量数据库召回不足时应该如何调整 chunk、embedding、top_k 和 rerank？
4. Query Transformation 在提升 RAG 候选文档覆盖率方面有什么作用？

检索器会对 5 个查询分别检索：
1. 原始 query
2. 改写 query 1
3. 改写 query 2
4. 改写 query 3
5. 改写 query 4
```

---

## 4. `models.py`：核心数据结构

### 4.1 Document

```python
@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
```

`Document` 表示知识库中的一份文档。

字段解释：

| 字段 | 说明 |
| --- | --- |
| `doc_id` | 文档唯一 ID，用于去重和融合 |
| `title` | 文档标题 |
| `text` | 文档正文 |
| `metadata` | 元数据，例如 topic、source、tenant_id、created_at |

为什么需要 `doc_id`：

同一篇文档可能被多个改写 query 命中。如果没有稳定 ID，就无法判断“这是同一个文档还是不同文档”。RRF 融合必须依赖文档 ID 去重。

### 4.2 TransformedQuery

```python
@dataclass
class TransformedQuery:
    text: str
    strategy: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
```

`TransformedQuery` 表示改写后的查询。

字段解释：

| 字段 | 说明 |
| --- | --- |
| `text` | 真正送入 retriever 的文本 |
| `strategy` | 来源策略，例如 original、hyde、multi_query、decomposition |
| `weight` | 策略权重 |
| `metadata` | 额外信息，例如 source_query、prompt、index |

为什么不直接用字符串：

如果只用字符串，后续你就不知道这个查询来自 HyDE 还是 Multi-Query，也不知道它对应的原始 query 是什么。生产环境中，Query Transformation 必须可观测、可追踪、可评估。

### 4.3 RetrievedNode

```python
@dataclass
class RetrievedNode:
    doc: Document
    score: float
    rank: int
    query_text: str
    strategy: str
```

`RetrievedNode` 表示某个 query 检索出来的一条结果。

字段解释：

| 字段 | 说明 |
| --- | --- |
| `doc` | 命中的文档 |
| `score` | 检索器返回的相关性分数 |
| `rank` | 该文档在本次查询结果中的排名 |
| `query_text` | 是哪个改写 query 命中的 |
| `strategy` | 是哪个策略命中的 |

注意：同一个文档可能对应多个 `RetrievedNode`，因为它可能被多个 query 命中。

### 4.4 FusedResult

```python
@dataclass
class FusedResult:
    doc: Document
    fusion_score: float
    best_score: float
    strategies: list[str]
    matched_queries: list[str]
    ranks: list[int]
```

`FusedResult` 表示融合后的最终候选文档。

字段解释：

| 字段 | 说明 |
| --- | --- |
| `doc` | 去重后的文档 |
| `fusion_score` | RRF 融合分 |
| `best_score` | 该文档在所有检索结果中的最高原始分 |
| `strategies` | 哪些策略命中过它 |
| `matched_queries` | 哪些 query 命中过它 |
| `ranks` | 它在不同 query 结果中的排名列表 |

这个结构非常适合第 9 天接 Rerank，因为你可以看到 reranker 最终排序和 query transformation 的贡献关系。

---

## 5. `toy_corpus.py`：内置知识库

代码里没有依赖外部文档，而是内置了一个小型知识库：

```python
def load_toy_documents() -> list[Document]:
    return [
        Document(
            doc_id="doc_query_transform",
            title="Query Transformation 概览",
            text="Query Transformation 是 RAG 检索前的查询改写层...",
            metadata={"topic": "query_transformation"},
        ),
        ...
    ]
```

这批文档覆盖：

1. Query Transformation。
2. HyDE。
3. Multi-Query。
4. Query Decomposition。
5. RRF。
6. Rerank。
7. Milvus 索引。
8. Qdrant filter。

为什么使用 toy corpus：

1. 你可以立刻运行，不需要先准备数据。
2. 每篇文档主题明确，方便观察召回变化。
3. 适合作为单元测试和教学例子。
4. 后续可以替换成真实课程文档。

替换成真实文档时，只需要让你的 loader 返回 `list[Document]` 即可。

---

## 6. `simple_retriever.py`：简化检索器

这个文件实现了一个纯 Python 的 TF-IDF 风格检索器。

它不是生产级向量检索器，但非常适合教学，因为你可以看清楚“多个 query 进入 retriever，然后返回多个结果”的机制。

### 6.1 分词函数

```python
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
```

这个正则会提取：

1. 英文单词，例如 `RAG`、`HyDE`、`Multi`。
2. 数字和下划线，例如 `top_k`。
3. 单个中文字符。

接着生成 bigram：

```python
bigrams = [tokens[i] + tokens[i + 1] for i in range(len(tokens) - 1)]
return tokens + bigrams
```

为什么加 bigram：

中文如果只按单字匹配，噪声会比较大。例如“召回率”拆成“召”“回”“率”后，单字信息太弱。加入 bigram 后，会出现“召回”“回率”这种更有意义的特征。

### 6.2 初始化检索器

```python
class SimpleTfidfRetriever:
    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.doc_tokens = {doc.doc_id: tokenize(doc.title + " " + doc.text) for doc in documents}
        self.doc_vectors = {doc.doc_id: Counter(tokens) for doc, tokens in zip(documents, self.doc_tokens.values())}
        self.idf = self._build_idf()
```

初始化时做了三件事：

1. 对每篇文档分词。
2. 用 `Counter` 统计词频。
3. 计算 IDF。

IDF 的作用：

常见词权重低，稀有词权重高。例如 `RAG`、`HyDE`、`RRF` 这类词比“的”“和”“是”更有检索价值。

### 6.3 检索入口

```python
def retrieve(self, query: TransformedQuery, top_k: int = 5) -> list[RetrievedNode]:
    query_vector = Counter(tokenize(query.text))
    scored = []

    for doc in self.documents:
        doc_vector = self.doc_vectors[doc.doc_id]
        score = self._cosine_similarity(query_vector, doc_vector) * query.weight
        if score > 0:
            scored.append((doc, score))
```

这里的关键是入参不是普通字符串，而是 `TransformedQuery`。

这样检索器可以拿到：

1. `query.text`：实际检索文本。
2. `query.strategy`：这个 query 来自哪个策略。
3. `query.weight`：该策略权重。

### 6.4 为什么乘以 weight

```python
score = self._cosine_similarity(query_vector, doc_vector) * query.weight
```

不同策略可靠性不同：

1. `original` 通常最忠实，权重设为 `1.0`。
2. `hyde` 可能引入假设性内容，权重设为 `0.85`。
3. `multi_query` 可能扩展范围，权重设为 `0.9`。

这不是唯一做法，但它展示了一个重要思想：查询改写策略可以被加权。

---

## 7. `mock_llm.py`：本地规则版 LLM

这个文件的作用是模拟真实 LLM，让整个项目在没有 API Key、没有网络、没有额外依赖的情况下也能运行。

核心接口：

```python
class RuleBasedLLM:
    def complete(self, prompt: str) -> str:
        ...
```

这个接口刻意设计得很像真实 LLM：

```python
response = llm.complete(prompt)
```

未来你接 OpenAI、DeepSeek、Qwen、LlamaIndex LLM 时，只要实现同样的 `complete(prompt: str) -> str` 就行。

### 7.1 策略判断

```python
if "假设性文档" in prompt or "HyDE" in prompt:
    return self._hyde(query)

if "不同角度" in prompt or "JSON 数组" in prompt and "检索查询" in prompt:
    return json.dumps(self._multi_query(query), ensure_ascii=False)

if "拆解" in prompt or "子问题" in prompt:
    return json.dumps(self._decompose(query), ensure_ascii=False)
```

这里根据 prompt 判断要模拟哪类输出：

1. HyDE 返回一段文本。
2. Multi-Query 返回 JSON 数组。
3. Decomposition 返回 JSON 数组。

代码里曾经出现过一个真实排查点：HyDE prompt 最初没有显式包含 `HyDE` 或 `假设性文档` 关键词，导致 mock LLM 没有走 HyDE 分支。后来在 prompt 中加入：

```python
HYDE_PROMPT = """你是一个 RAG 查询改写器，当前策略是 HyDE。
...
"""
```

这说明生产系统中最好让 prompt、strategy、日志三者一致，否则排查会很痛苦。

### 7.2 HyDE 输出

```python
def _hyde(self, query: str) -> str:
    if "召回" in query or "RAG" in query:
        return (
            "提升 RAG 召回率通常需要从查询改写、向量表示、chunk 切分、top_k 设置、混合检索和 rerank 等方面优化。"
            "Query Transformation 可以把原始问题改写成更适合检索的表达，例如 HyDE 生成假设性文档，"
            "Multi-Query 生成多个角度的查询，以扩大候选文档覆盖范围。"
        )
```

注意：HyDE 输出不是最终答案，而是“用于检索的假设性文档”。

它会刻意包含：

1. 查询改写。
2. 向量表示。
3. chunk。
4. top_k。
5. 混合检索。
6. rerank。
7. HyDE。
8. Multi-Query。

这些词会帮助检索器命中更多相关文档。

### 7.3 Multi-Query 输出

```python
def _multi_query(self, query: str) -> list[str]:
    if "召回" in query or "RAG" in query:
        return [
            "RAG 系统中提高向量检索召回率的方法有哪些？",
            "如何通过 HyDE、Multi-Query 和查询改写改善 RAG 检索效果？",
            "向量数据库召回不足时应该如何调整 chunk、embedding、top_k 和 rerank？",
            "Query Transformation 在提升 RAG 候选文档覆盖率方面有什么作用？",
        ]
```

这体现了 Multi-Query 的核心：

同一个用户问题被改写成多个不同角度的检索问题。

### 7.4 Decomposition 输出

```python
def _decompose(self, query: str) -> list[str]:
    if "Milvus" in query and "Qdrant" in query:
        return [
            "Milvus 支持哪些向量索引类型？",
            "Qdrant 支持哪些向量索引和检索能力？",
            "Milvus 的 metadata filter 能力如何？",
            "Qdrant 的 payload filter 能力如何？",
            "Milvus 和 Qdrant 在分布式扩展方面有什么差异？",
            "Milvus 和 Qdrant 的 Python SDK 生态分别如何？",
        ]
```

Decomposition 的重点不是改写同义句，而是拆任务。

---

## 8. `query_transformers.py`：核心改写策略

这是今天最重要的文件。

### 8.1 LLM 协议

```python
class LLM(Protocol):
    def complete(self, prompt: str) -> str:
        ...
```

`Protocol` 的意思是：只要一个对象有 `complete(prompt: str) -> str` 方法，它就可以被当作 LLM 使用。

好处：

1. 当前可以使用 `RuleBasedLLM`。
2. 后续可以换成 OpenAI。
3. 也可以换成 LlamaIndex 的 LLM。
4. 业务代码不用改。

### 8.2 QueryTransformer 协议

```python
class QueryTransformer(Protocol):
    def transform(self, query: str) -> list[TransformedQuery]:
        ...
```

所有查询改写器都遵守同一个接口：

```python
输入：str
输出：list[TransformedQuery]
```

这让 pipeline 可以统一处理不同策略。

### 8.3 OriginalQueryTransformer

```python
class OriginalQueryTransformer:
    def transform(self, query: str) -> list[TransformedQuery]:
        return [TransformedQuery(text=query, strategy="original", weight=1.0)]
```

为什么原始 query 也要做成 transformer：

1. 统一接口。
2. 方便和 HyDE、Multi-Query 组合。
3. 保留 baseline。
4. 便于 compare 模式。

### 8.4 HyDETransformer

```python
class HyDETransformer:
    def __init__(self, llm: LLM, include_original: bool = True):
        self.llm = llm
        self.include_original = include_original
```

`include_original=True` 非常重要。

推荐不要只用 HyDE 替代原始 query，而是同时保留：

```text
原始 query
HyDE 假设性文档
```

因为：

1. 原始 query 最忠实。
2. HyDE 提供语义扩展。
3. 两路结果融合后更稳。

核心逻辑：

```python
hypothetical_doc = self.llm.complete(HYDE_PROMPT.format(query=query)).strip()
transformed.append(
    TransformedQuery(
        text=hypothetical_doc,
        strategy="hyde",
        weight=0.85,
        metadata={"source_query": query, "prompt": "HYDE_PROMPT"},
    )
)
```

这里生成的 `hypothetical_doc` 会被送入 retriever。

注意：它不会被当作真实文档。

### 8.5 MultiQueryTransformer

核心逻辑：

```python
prompt = MULTI_QUERY_PROMPT.format(query=query, num_queries=self.num_queries)
raw_response = self.llm.complete(prompt)
rewritten_queries = parse_json_string_list(raw_response)
```

Multi-Query 强制 LLM 输出 JSON 数组，原因是便于程序解析。

然后转成统一结构：

```python
TransformedQuery(
    text=item,
    strategy="multi_query",
    weight=0.9,
    metadata={"source_query": query, "index": index, "prompt": "MULTI_QUERY_PROMPT"},
)
```

这里的 `index` 表示这是第几个改写 query。

### 8.6 DecompositionTransformer

核心逻辑类似 Multi-Query，但语义不同：

```python
sub_questions = parse_json_string_list(raw_response)
```

Multi-Query 是“多个角度重写同一问题”。

Decomposition 是“把复杂问题拆成多个子问题”。

例如：

```text
Milvus 和 Qdrant 在索引类型、过滤能力、分布式扩展和 Python 生态上有什么区别？
```

会被拆成：

```text
Milvus 支持哪些向量索引类型？
Qdrant 支持哪些向量索引和检索能力？
Milvus 的 metadata filter 能力如何？
Qdrant 的 payload filter 能力如何？
...
```

### 8.7 QueryTransformationPipeline

```python
class QueryTransformationPipeline:
    def __init__(self, transformers: list[QueryTransformer]):
        self.transformers = transformers

    def transform(self, query: str) -> list[TransformedQuery]:
        transformed = []
        for transformer in self.transformers:
            transformed.extend(transformer.transform(query))
        return deduplicate_queries(transformed)
```

这个类负责组合多个 transformer。

例如 `hyde_multi_query`：

```python
QueryTransformationPipeline(
    [
        HyDETransformer(llm, include_original=True),
        MultiQueryTransformer(llm, num_queries=4, include_original=False),
    ]
)
```

它会返回：

```text
1. original query
2. hyde query
3. multi query 1
4. multi query 2
5. multi query 3
6. multi query 4
```

### 8.8 JSON 解析兜底

```python
def parse_json_string_list(raw_response: str) -> list[str]:
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        return fallback_parse_lines(raw_response)
```

真实 LLM 不一定每次都严格输出 JSON，可能输出：

```text
1. 查询 A
2. 查询 B
3. 查询 C
```

所以代码提供了 `fallback_parse_lines` 兜底。

这在生产环境中非常重要。LLM 输出解析失败时，不应该让整个检索流程崩溃。

### 8.9 去重

```python
def deduplicate_queries(queries: list[TransformedQuery]) -> list[TransformedQuery]:
    seen = set()
    result = []

    for query in queries:
        normalized = query.text.strip().lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(query)

    return result
```

为什么需要去重：

1. LLM 可能生成和原始 query 相同的改写。
2. 多个 transformer 可能生成重复 query。
3. 重复 query 会浪费检索成本。
4. 重复 query 会让融合分数失真。

---

## 9. `fusion.py`：RRF 融合

RRF 全称 Reciprocal Rank Fusion。

公式：

```text
score(doc) = sum(1 / (rrf_k + rank_i(doc)))
```

代码：

```python
fusion_scores[doc_id] += 1.0 / (rrf_k + item.rank)
```

假设 `rrf_k=60`：

| rank | 贡献分 |
| --- | --- |
| 1 | 1 / 61 = 0.01639 |
| 2 | 1 / 62 = 0.01613 |
| 3 | 1 / 63 = 0.01587 |
| 4 | 1 / 64 = 0.01563 |

如果同一文档被多个 query 命中，它的分数会累加。

例如：

```text
doc_query_transform:
原始 query rank=2 -> 1/62
HyDE query rank=1 -> 1/61
总分约 0.0325
```

这就是为什么被多路 query 同时命中的文档会排得更靠前。

### 9.1 为什么不用原始 score 直接相加

不同 query 的原始 score 不一定可比。

例如：

```text
query A 的最高 score = 0.80
query B 的最高 score = 0.35
```

这不一定说明 query A 的结果更相关，可能只是该 query 更容易和文档重合。

RRF 更关注排名，不强依赖 score 的绝对值。

### 9.2 保存可观测信息

```python
strategies[doc_id].add(item.strategy)
matched_queries[doc_id].append(item.query_text)
ranks[doc_id].append(item.rank)
```

这样融合后你仍然知道：

1. 哪些策略命中了这个文档。
2. 哪些改写 query 命中了这个文档。
3. 它在各路结果中的排名。

这对调试非常关键。

---

## 10. `pipeline.py`：检索管线

```python
class RetrievalPipeline:
    def __init__(self, retriever: SimpleTfidfRetriever, per_query_top_k: int = 4, final_top_k: int = 5):
        self.retriever = retriever
        self.per_query_top_k = per_query_top_k
        self.final_top_k = final_top_k
```

参数解释：

| 参数 | 说明 |
| --- | --- |
| `per_query_top_k` | 每个改写 query 检索多少条 |
| `final_top_k` | 融合后保留多少条 |

核心逻辑：

```python
def retrieve(self, transformed_queries: list[TransformedQuery]) -> tuple[list[RetrievedNode], list[FusedResult]]:
    all_results = []

    for query in transformed_queries:
        all_results.extend(self.retriever.retrieve(query, top_k=self.per_query_top_k))

    fused = fuse_with_rrf(all_results, top_k=self.final_top_k)
    return all_results, fused
```

这段代码体现了 Query Transformation 的本质：

```text
一个用户问题 -> 多个 transformed query -> 多次检索 -> 结果融合
```

---

## 11. `demo.py`：命令行入口

### 11.1 构建策略

```python
def build_transformer(strategy: str, llm: RuleBasedLLM):
    if strategy == "original":
        return QueryTransformationPipeline([OriginalQueryTransformer()])

    if strategy == "hyde":
        return QueryTransformationPipeline([HyDETransformer(llm, include_original=True)])
```

支持的策略：

| strategy | 说明 |
| --- | --- |
| `original` | 只用原始 query |
| `hyde` | 原始 query + HyDE 假设性文档 |
| `multi_query` | 原始 query + 多个改写 query |
| `decomposition` | 原始 query + 子问题 |
| `hyde_multi_query` | 原始 query + HyDE + Multi-Query |
| `compare` | 依次运行所有策略 |

### 11.2 单次运行

```python
def run_once(strategy: str, query: str) -> dict:
    documents = load_toy_documents()
    llm = RuleBasedLLM()
    transformer = build_transformer(strategy, llm)
    retriever = SimpleTfidfRetriever(documents)
    pipeline = RetrievalPipeline(retriever, per_query_top_k=4, final_top_k=5)
```

这里初始化：

1. 文档。
2. LLM。
3. Query Transformer。
4. Retriever。
5. Retrieval Pipeline。

然后：

```python
transformed_queries = transformer.transform(query)
raw_results, fused_results = pipeline.retrieve(transformed_queries)
```

这两行就是主流程。

### 11.3 输出结构

Demo 会打印三段核心信息：

1. `Transformed Queries`：看 query 被改写成什么。
2. `Raw Retrieval Results`：看每个 query 分别召回什么。
3. `Fused Results`：看融合后的最终候选文档。

这三个视角都重要。

只看最终答案是不够的，因为你不知道召回是否正常。

---

## 12. 运行 HyDE 的结果怎么读

命令：

```powershell
python demo.py --strategy hyde --query "如何提升 RAG 的召回率？"
```

你会看到：

```text
Transformed Queries
1. [original] 如何提升 RAG 的召回率？
2. [hyde] 提升 RAG 召回率通常需要从查询改写、向量表示、chunk 切分、top_k 设置、混合检索和 rerank 等方面优化...
```

这说明 HyDE 生成了一段更长、更丰富的假设性文档。

原始 query 可能召回：

```text
doc_milvus_index
doc_query_transform
doc_qdrant_filter
doc_rerank
```

HyDE query 可能召回：

```text
doc_query_transform
doc_multi_query
doc_hyde
doc_decomposition
```

观察点：

1. 原始 query 命中了 `doc_milvus_index`，因为里面有“召回率”等词，但主题不一定最贴近。
2. HyDE query 命中了 `doc_query_transform`、`doc_multi_query`、`doc_hyde`，这些更接近“如何提升 RAG 召回”。
3. 融合后 `doc_query_transform` 会排到前面，因为它被 original 和 HyDE 同时命中。

这就是 HyDE 的价值：用更丰富的语义文本改善召回方向。

---

## 13. 运行 Multi-Query 的结果怎么读

命令：

```powershell
python demo.py --strategy multi_query --query "如何提升 RAG 的召回率？"
```

你会看到 5 个 query：

```text
1. original
2. RAG 系统中提高向量检索召回率的方法有哪些？
3. 如何通过 HyDE、Multi-Query 和查询改写改善 RAG 检索效果？
4. 向量数据库召回不足时应该如何调整 chunk、embedding、top_k 和 rerank？
5. Query Transformation 在提升 RAG 候选文档覆盖率方面有什么作用？
```

这些 query 分别覆盖：

1. RAG 召回率。
2. HyDE / Multi-Query / 查询改写。
3. chunk / embedding / top_k / rerank。
4. Query Transformation / 候选文档覆盖。

Multi-Query 的优势是覆盖面变大。

风险是噪声也会变多，所以第 9 天要接 Rerank。

---

## 14. 运行 Decomposition 的结果怎么读

命令：

```powershell
python demo.py --strategy decomposition --query "Milvus 和 Qdrant 在索引类型、过滤能力、分布式扩展和 Python 生态上有什么区别？"
```

你会看到子问题：

```text
Milvus 支持哪些向量索引类型？
Qdrant 支持哪些向量索引和检索能力？
Milvus 的 metadata filter 能力如何？
Qdrant 的 payload filter 能力如何？
Milvus 和 Qdrant 在分布式扩展方面有什么差异？
Milvus 和 Qdrant 的 Python SDK 生态分别如何？
```

Decomposition 适合这种复合问题，因为原问题包含多个维度。

如果只用原始 query，检索器可能只命中其中一两个维度。

拆成子问题后，每个维度都能单独召回证据。

---

## 15. 日志文件怎么看

### 15.1 transformed_queries

文件：

```text
outputs/hyde_transformed_queries.jsonl
```

示例：

```json
{
  "original_query": "如何提升 RAG 的召回率？",
  "strategy": "hyde",
  "text": "提升 RAG 召回率通常需要从查询改写、向量表示...",
  "weight": 0.85,
  "metadata": {
    "source_query": "如何提升 RAG 的召回率？",
    "prompt": "HYDE_PROMPT"
  }
}
```

用途：

1. 检查 LLM 到底改写了什么。
2. 排查意图漂移。
3. 做 prompt 版本对比。
4. 统计改写成本。

### 15.2 raw_results

文件：

```text
outputs/hyde_raw_results.jsonl
```

示例字段：

```json
{
  "strategy": "hyde",
  "query_text": "提升 RAG 召回率通常需要...",
  "rank": 1,
  "score": 0.3702,
  "doc_id": "doc_query_transform",
  "title": "Query Transformation 概览"
}
```

用途：

1. 看每个 query 单独召回了什么。
2. 判断是哪个 query 贡献了好结果。
3. 找到引入噪声的 query。

### 15.3 fused_results

文件：

```text
outputs/hyde_fused_results.jsonl
```

示例字段：

```json
{
  "rank": 1,
  "doc_id": "doc_query_transform",
  "fusion_score": 0.0325,
  "best_score": 0.3702,
  "strategies": ["hyde", "original"],
  "matched_queries": ["如何提升 RAG 的召回率？", "提升 RAG 召回率通常需要..."],
  "ranks": [2, 1]
}
```

用途：

1. 看融合后的最终排序。
2. 看文档是否被多路 query 命中。
3. 为第 9 天 Rerank 提供候选集。

---

## 16. 如何接入真实 LLM

当前代码使用：

```python
llm = RuleBasedLLM()
```

你可以替换为任意实现了 `complete(prompt: str) -> str` 的类。

示例结构：

```python
class RealLLM:
    def complete(self, prompt: str) -> str:
        response = your_llm_client.generate(prompt)
        return response.text
```

然后在 `demo.py` 中替换：

```python
llm = RealLLM()
```

不要改 `HyDETransformer`、`MultiQueryTransformer`、`DecompositionTransformer`。

这就是接口抽象的价值。

---

## 17. 如何替换成 LlamaIndex Retriever

当前使用：

```python
retriever = SimpleTfidfRetriever(documents)
```

如果你有 LlamaIndex 索引，可以写一个适配器：

```python
class LlamaIndexRetrieverAdapter:
    def __init__(self, llama_retriever):
        self.llama_retriever = llama_retriever

    def retrieve(self, query: TransformedQuery, top_k: int = 5) -> list[RetrievedNode]:
        nodes = self.llama_retriever.retrieve(query.text)
        results = []

        for rank, node in enumerate(nodes[:top_k], start=1):
            doc = Document(
                doc_id=node.node.node_id,
                title=node.node.metadata.get("title", ""),
                text=node.node.get_content(),
                metadata=node.node.metadata,
            )
            results.append(
                RetrievedNode(
                    doc=doc,
                    score=node.score or 0.0,
                    rank=rank,
                    query_text=query.text,
                    strategy=query.strategy,
                )
            )

        return results
```

然后：

```python
retriever = LlamaIndexRetrieverAdapter(index.as_retriever(similarity_top_k=5))
pipeline = RetrievalPipeline(retriever)
```

这说明我们当前代码不是和 toy retriever 绑定死的。

---

## 18. 如何扩展新策略：Step-back Query

如果你想新增 Step-back Query，只需要新增一个类：

```python
STEP_BACK_PROMPT = """你是一个 RAG 查询改写器。
请把用户问题改写成一个更抽象、更上位的背景问题。

用户问题：
{query}
"""


class StepBackTransformer:
    def __init__(self, llm: LLM, include_original: bool = True):
        self.llm = llm
        self.include_original = include_original

    def transform(self, query: str) -> list[TransformedQuery]:
        transformed = []

        if self.include_original:
            transformed.append(TransformedQuery(text=query, strategy="original", weight=1.0))

        step_back_query = self.llm.complete(STEP_BACK_PROMPT.format(query=query)).strip()
        transformed.append(
            TransformedQuery(
                text=step_back_query,
                strategy="step_back",
                weight=0.8,
                metadata={"source_query": query, "prompt": "STEP_BACK_PROMPT"},
            )
        )

        return deduplicate_queries(transformed)
```

再在 `demo.py` 里加一个策略分支即可。

---

## 19. 生产环境建议

### 19.1 加缓存

真实 Query Transformation 会调用 LLM，必须缓存。

缓存 key 建议包含：

```text
strategy
prompt_version
model
original_query
```

伪代码：

```python
cache_key = sha256(f"{strategy}:{prompt_version}:{model}:{query}".encode()).hexdigest()
```

### 19.2 加 prompt version

当前 metadata 里记录了 prompt 名称：

```python
metadata={"prompt": "HYDE_PROMPT"}
```

生产环境建议升级为：

```python
metadata={"prompt_version": "HYDE_PROMPT_V1_2026_06_16"}
```

这样后续评估结果可复现。

### 19.3 加失败兜底

真实 LLM 可能超时、限流、返回非法 JSON。

建议：

1. Multi-Query JSON 解析失败时使用 `fallback_parse_lines`。
2. LLM 调用失败时返回 original query。
3. 记录错误日志。
4. 不阻断主检索流程。

### 19.4 加并发

Multi-Query 会产生多路检索。

当前代码是串行：

```python
for query in transformed_queries:
    all_results.extend(self.retriever.retrieve(query, top_k=self.per_query_top_k))
```

生产环境可以并发：

```python
with ThreadPoolExecutor() as pool:
    results = pool.map(retrieve_one_query, transformed_queries)
```

### 19.5 加 Rerank

第 8 天先做 Query Transformation + RRF。

第 9 天建议变成：

```text
Query Transformation
  -> 多路检索
  -> RRF 初步融合
  -> Reranker 精排
  -> Final Context
```

这样 Multi-Query 扩大召回后产生的噪声，可以由 Rerank 压下去。

---

## 20. 你应该重点观察什么

运行 Demo 时，不要只看最后排名，重点看三件事：

### 20.1 改写是否合理

看 `Transformed Queries`：

```text
HyDE 是否补充了有用术语？
Multi-Query 是否覆盖了不同角度？
Decomposition 是否拆得清楚？
```

如果改写偏了，后面检索一定会偏。

### 20.2 每一路 query 召回了什么

看 `Raw Retrieval Results`：

```text
哪个 query 命中了好文档？
哪个 query 引入了噪声？
原始 query 是否仍然有价值？
```

### 20.3 融合是否把多路共识排上来了

看 `Fused Results`：

```text
被多路 query 命中的文档是否排得更靠前？
fusion_score 是否符合预期？
strategies 是否记录完整？
```

---

## 21. 代码学习路线

建议按这个顺序阅读：

1. `demo.py`
2. `src/query_transformers.py`
3. `src/mock_llm.py`
4. `src/simple_retriever.py`
5. `src/fusion.py`
6. `src/models.py`
7. `src/logger.py`

原因：

1. 先看入口，知道整体怎么跑。
2. 再看核心策略。
3. 再看模拟 LLM 如何生成内容。
4. 再看检索器。
5. 最后看融合和日志。

---

## 22. 今日你可以继续做的练习

### 练习一：修改 HyDE prompt

目标：让 HyDE 输出更短、更保守。

观察：

1. 召回是否变少。
2. 噪声是否降低。
3. `doc_hyde` 是否仍然能被召回。

### 练习二：把 `num_queries` 从 4 改成 2

位置：

```python
MultiQueryTransformer(llm, num_queries=4, include_original=True)
```

观察：

1. raw result 数量是否减少。
2. fused result 是否更集中。
3. 是否漏掉 `doc_multi_query` 或 `doc_rerank`。

### 练习三：修改 `rrf_k`

位置：

```python
fuse_with_rrf(all_results, top_k=self.final_top_k)
```

可以改成：

```python
fuse_with_rrf(all_results, top_k=self.final_top_k, rrf_k=10)
```

观察：

1. rank 靠前的结果权重是否更强。
2. 多路命中的文档是否更容易排第一。

### 练习四：新增真实课程文档

在 `toy_corpus.py` 里新增一篇文档：

```python
Document(
    doc_id="doc_my_note",
    title="我的 RAG 笔记",
    text="这里写你自己的学习资料...",
    metadata={"topic": "my_note"},
)
```

然后重新运行 Demo。

### 练习五：新增 Step-back Query

按照第 18 节添加 `StepBackTransformer`，再观察它适合哪些问题。

---

## 23. 最重要的工程结论

Query Transformation 的代码实现可以拆成四层：

```text
1. Strategy Layer
   HyDE / Multi-Query / Decomposition

2. Retrieval Layer
   每个 transformed query 分别检索

3. Fusion Layer
   对多路检索结果去重、融合、排序

4. Observability Layer
   记录 query、结果、策略、排名、分数
```

你今天写的代码虽然是教学版，但结构已经接近真实生产系统。

后续要做生产化，主要是替换三件东西：

1. `RuleBasedLLM` 替换成真实 LLM。
2. `SimpleTfidfRetriever` 替换成 LlamaIndex / Milvus / Qdrant 检索器。
3. `RRF` 后面接入第 9 天的 Rerank。

其余抽象可以保留。
