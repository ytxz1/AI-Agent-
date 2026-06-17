# 04 代码实战：第一周 RAG 系统升级为 Milvus + 混合检索 + Reranker

> 本文专门解释第 14 天新增代码。目标不是只给你一个脚本，而是把每一段代码背后的 RAG 工程意义讲清楚，让你知道如何从第一周基础 RAG 升级到 Advanced RAG v2。

---

## 1. 本次新增了哪些文件

```text
第14天周度总结与系统升级/
  requirements.txt
  configs/
    retrieval.yaml
  data/
    raw/
      01_rag_v1_baseline.md
      02_milvus_vector_database.md
      03_hybrid_retrieval.md
      04_reranker_second_stage.md
      05_query_transformation.md
      06_rag_evaluation_trace.md
    eval/
      eval_queries.jsonl
  src/
    rag_v2_upgrade_demo.py
    evaluate_rag_v2.py
  outputs/
    rag_v2_retrieval_trace.jsonl
    rag_v2_evaluation_report.md
```

其中最重要的是两个 Python 文件：

| 文件 | 作用 |
|---|---|
| `src/rag_v2_upgrade_demo.py` | 完整 RAG v2 查询链路：加载文档、切分、BM25、dense retrieval、Milvus 可选、RRF、Rerank、Context、Trace |
| `src/evaluate_rag_v2.py` | 评估脚本：对比 BM25、Dense、Hybrid、Hybrid+Rerank、QueryTransform+Hybrid+Rerank |

---

## 2. 如何运行

进入第 14 天目录：

```powershell
cd "D:\vscode项目\AI Agent 开发工程师学习路线图（工程落地版）\第 2 周：Advanced RAG 与生产级向量数据库\第14天周度总结与系统升级"
```

### 2.1 离线运行完整 RAG v2 demo

这个版本不需要安装任何第三方库：

```powershell
python src/rag_v2_upgrade_demo.py
```

指定一个问题：

```powershell
python src/rag_v2_upgrade_demo.py --query "为什么生产级 RAG 需要 BM25 和 Embedding 混合检索？"
```

你会看到四组结果：

```text
=== BM25 sparse retrieval ===
...

=== Dense retrieval ===
...

=== Hybrid RRF fusion ===
...

=== Reranked final contexts ===
...
```

最后会生成：

```text
outputs/rag_v2_retrieval_trace.jsonl
```

### 2.2 运行评估

```powershell
python src/evaluate_rag_v2.py
```

输出报告：

```text
outputs/rag_v2_evaluation_report.md
```

### 2.3 可选：使用真实 Milvus 后端

先确保 Milvus 已经在本机运行：

```text
http://localhost:19530
```

安装依赖：

```powershell
pip install pymilvus
```

运行：

```powershell
python src/rag_v2_upgrade_demo.py --vector-backend milvus --rebuild-milvus
```

说明：

- `--vector-backend local` 是默认值，不依赖 Milvus。
- `--vector-backend milvus` 会把 chunk embedding 写入 Milvus。
- `--rebuild-milvus` 会重建 demo collection。

---

## 3. 整体代码架构

`rag_v2_upgrade_demo.py` 的主流程在 `run_pipeline()` 中：

```python
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
```

这就是一个完整 RAG v2 查询管线：

```text
load_documents
  -> split_into_chunks
  -> build BM25 retriever
  -> build dense retriever
  -> query transformation
  -> BM25 retrieve
  -> dense retrieve
  -> RRF fusion
  -> lightweight rerank
  -> build context
  -> compose answer
  -> save retrieval trace
```

和第一周基础 RAG 相比，它多了：

1. Query Transformation。
2. BM25。
3. Dense Retriever 抽象。
4. Milvus 可选后端。
5. RRF 融合。
6. Reranker。
7. Context Builder。
8. Retrieval Trace。
9. Evaluation。

---

## 4. 数据结构解释

### 4.1 Document

```python
@dataclass(frozen=True)
class Document:
    doc_id: str
    source: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)
```

它表示一篇原始文档。

字段说明：

| 字段 | 含义 |
|---|---|
| `doc_id` | 文档 ID，这里用文件名 stem |
| `source` | 文档来源路径 |
| `text` | 文档正文 |
| `metadata` | 文件名、语料版本等元数据 |

### 4.2 Chunk

```python
@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)
```

Chunk 是 RAG 检索的基本单元。

生产系统里，chunk 必须有稳定 ID，因为后续要：

1. 写入 Milvus。
2. 建 BM25 索引。
3. 做 evaluation gold 标注。
4. 输出 citation。
5. 做增量更新和删除。

本代码中的 chunk_id 形式：

```text
03_hybrid_retrieval::chunk_000
```

### 4.3 TransformedQuery

```python
@dataclass
class TransformedQuery:
    text: str
    strategy: str
    weight: float = 1.0
    metadata: dict[str, str] = field(default_factory=dict)
```

它表示改写后的 query。

比如原问题：

```text
为什么生产级 RAG 需要混合检索？
```

可能生成：

```text
original: 为什么生产级 RAG 需要混合检索？
hyde: 为什么生产级 RAG 需要混合检索？这通常涉及 BM25、Milvus、RRF、Reranker...
multi_query: 为什么生产级 RAG 需要混合检索？ 的工程实现步骤是什么？
```

### 4.4 SearchResult

```python
@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    rank: int = 0
    method: str = ""
    debug: dict[str, object] = field(default_factory=dict)
```

所有检索器都返回统一的 SearchResult。

好处：

1. BM25、Dense、Milvus、Fusion、Rerank 都能共用同一种结果结构。
2. 后面做 RRF 时不用关心结果来自哪个检索器。
3. debug 字段能记录 matched_terms、query_strategy、fusion_trace 等调试信息。

---

## 5. 文档加载与切分

### 5.1 加载文档

```python
def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
```

它读取：

```text
data/raw/*.md
data/raw/*.txt
```

每个文件变成一个 Document。

关键代码：

```python
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
```

注意：`source` 使用相对路径，是为了让 evaluation 中的 `relevant_sources` 可以稳定匹配。

### 5.2 切分 Chunk

```python
def split_into_chunks(
    documents: Iterable[Document],
    chunk_size: int = 720,
    chunk_overlap: int = 120,
) -> list[Chunk]:
```

这不是最复杂的生产级 chunker，但比简单固定长度切分更清晰：

1. 先按空行切成段落。
2. 尽量把段落合并到 `chunk_size` 以内。
3. 超长时保留 `chunk_overlap`。
4. 每个 chunk 继承文档 metadata。
5. 每个 chunk 提取标题作为 `section_path`。

生产系统中你可以继续升级：

1. Markdown 按标题切分。
2. PDF 按 block 切分。
3. 表格单独生成 table chunk。
4. 图片用 caption + OCR 生成 image chunk。

---

## 6. Embedding：为什么用 HashingEmbedder

代码里没有直接调用 OpenAI 或 BGE，而是写了：

```python
class HashingEmbedder:
```

原因是教学 demo 要保证：

1. 不需要 API key。
2. 不需要下载模型。
3. 不需要网络。
4. 结果可复现。
5. 仍然保持真实工程中的接口形状。

真实 embedding 的接口通常是：

```python
embed_query(text) -> vector
embed_documents(texts) -> vectors
```

本 demo 也保持这个接口：

```python
def embed_query(self, text: str) -> list[float]:
    return self.embed_text(text)

def embed_documents(self, texts: list[str]) -> list[list[float]]:
    return [self.embed_text(text) for text in texts]
```

所以你后续可以很容易替换成真实模型：

```python
class OpenAIEmbedder:
    def embed_query(self, text):
        ...

    def embed_documents(self, texts):
        ...
```

---

## 7. BM25 Retriever

BM25 代码在：

```python
class BM25Retriever:
```

它做了三件事：

1. 对每个 chunk 分词。
2. 统计 term frequency 和 document frequency。
3. 根据 BM25 公式计算 query 和 chunk 的匹配分数。

核心公式代码：

```python
numerator = tf * (self.k1 + 1)
denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
score += self.idf(term) * numerator / denominator
```

BM25 的价值：

1. 精确术语强。
2. 参数名、函数名、文件名强。
3. 不依赖 embedding 模型。
4. 可解释性好。

在输出中你会看到：

```python
debug={"matched_terms": matched[:20]}
```

这能告诉你 BM25 到底匹配了哪些词。

---

## 8. Dense Retriever

本项目有两个 dense retriever：

```python
class LocalDenseRetriever:
class MilvusDenseRetriever:
```

### 8.1 LocalDenseRetriever

默认运行的是本地 dense retriever：

```python
dense = LocalDenseRetriever(chunks, embedder)
```

它会：

1. 把所有 chunk 转成向量。
2. 把 query 转成向量。
3. 用 dot product 计算相似度。
4. 返回 top_k。

这对应第一周向量检索的基本形态。

### 8.2 MilvusDenseRetriever

如果你指定：

```powershell
python src/rag_v2_upgrade_demo.py --vector-backend milvus --rebuild-milvus
```

代码会走：

```python
class MilvusDenseRetriever:
```

它会：

1. 连接 Milvus。
2. 创建 collection。
3. 插入 chunk vectors。
4. 调用 Milvus search。
5. 把 Milvus hits 转成 SearchResult。

核心 schema：

```python
schema.add_field("id", self.DataType.INT64, is_primary=True)
schema.add_field("vector", self.DataType.FLOAT_VECTOR, dim=self.embedder.dimension)
schema.add_field("chunk_id", self.DataType.VARCHAR, max_length=256)
schema.add_field("doc_id", self.DataType.VARCHAR, max_length=256)
schema.add_field("text", self.DataType.VARCHAR, max_length=8192)
schema.add_field("source", self.DataType.VARCHAR, max_length=1024)
schema.add_field("title", self.DataType.VARCHAR, max_length=512)
```

这就是 RAG chunk collection 的最小可用版本。

生产系统还可以增加：

1. `tenant_id`
2. `category`
3. `page_start`
4. `page_end`
5. `corpus_version`
6. `metadata`

---

## 9. Query Transformation

代码中使用：

```python
class RuleBasedQueryTransformer:
```

它是离线规则版，用来模拟 LLM Query Transformation。

支持策略：

```text
original
hyde
multi_query
decomposition
```

默认运行：

```text
original,hyde,multi_query
```

你可以指定：

```powershell
python src/rag_v2_upgrade_demo.py --strategies original
```

或：

```powershell
python src/rag_v2_upgrade_demo.py --strategies original,hyde,multi_query,decomposition
```

### 9.1 original

保留原始 query：

```python
TransformedQuery(text=query, strategy="original", weight=1.0)
```

生产系统中必须始终保留 original query。

### 9.2 HyDE

代码里模拟 HyDE：

```python
text=(
    f"{query}。这通常涉及 RAG 系统中的文档切分、Embedding、BM25、"
    "Milvus 向量检索、混合检索、RRF 融合、Reranker 精排、评估指标和 retrieval trace。"
)
```

真实 HyDE 会调用 LLM 生成假设性文档。

注意：

1. HyDE 文本只用于检索。
2. 不应该把 HyDE 文本当事实证据。
3. Reranker 仍然使用 original query。

### 9.3 Multi-Query

代码里生成两个角度：

```python
f"{query} 的工程实现步骤是什么？"
f"{query} 在生产级 RAG 系统中的作用和风险是什么？"
```

真实系统中可以让 LLM 输出 JSON 数组。

---

## 10. RRF 融合

函数：

```python
def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    top_k: int = 8,
    rrf_k: int = 60,
) -> list[SearchResult]:
```

它把多路结果融合：

```text
BM25 original
BM25 hyde
BM25 multi_query
Dense original
Dense hyde
Dense multi_query
```

核心代码：

```python
scores[chunk_id] += 1.0 / (rrf_k + result.rank)
```

RRF 的优点：

1. 不需要比较 BM25 分数和 dense 分数。
2. 一个 chunk 被多路召回会加分。
3. 实现简单。
4. 很适合做初始 hybrid fusion。

debug 中会记录：

```python
"matched_by": [
  {"method": "bm25", "rank": 1, "query_strategy": "original"},
  {"method": "local_dense", "rank": 2, "query_strategy": "hyde"}
]
```

这能解释一个 chunk 为什么排到前面。

---

## 11. Reranker

代码中使用：

```python
class LightweightReranker:
```

它不是神经网络 reranker，而是为了离线教学写的特征打分器。

它使用这些特征：

| 特征 | 含义 |
|---|---|
| `term_coverage` | query token 被 chunk 覆盖的比例 |
| `keyword_coverage` | 关键术语覆盖比例 |
| `bigram_coverage` | 中文 bigram 覆盖比例 |
| `first_stage_score` | fusion 阶段分数 |
| `title_bonus` | query 是否命中文档标题 |

打分代码：

```python
rerank_score = (
    0.34 * term_coverage
    + 0.24 * keyword_coverage
    + 0.18 * bigram_coverage
    + 0.16 * first_stage
    + 0.08 * title_bonus
)
```

真实系统可以替换成：

1. `BAAI/bge-reranker-base`
2. `BAAI/bge-reranker-large`
3. `Cohere Rerank`
4. `cross-encoder/ms-marco-MiniLM-L-6-v2`

替换时只要保持接口：

```python
rerank(original_query, candidates, top_n)
```

---

## 12. Context Builder

函数：

```python
def build_context(results: list[SearchResult], max_chars: int = 2600) -> str:
```

它把 rerank 后的 chunk 转成可以交给 LLM 的上下文：

```text
[Source 1]
source: data/raw/03_hybrid_retrieval.md
chunk_id: 03_hybrid_retrieval::chunk_000
title: BM25 + Embedding 混合检索
content: ...
```

这样做的好处：

1. LLM 可以引用 Source 编号。
2. 用户可以看到答案来源。
3. trace 可以定位到 chunk。
4. 后续可以实现 citations。

生产系统中建议用 token 数控制，而不是字符数控制。

---

## 13. Retrieval Trace

每次运行都会保存：

```text
outputs/rag_v2_retrieval_trace.jsonl
```

一条 trace 包含：

```json
{
  "trace_id": "...",
  "query": "...",
  "vector_backend": "local",
  "num_documents": 6,
  "num_chunks": 6,
  "transformed_queries": [],
  "bm25_results": [],
  "dense_results": [],
  "fused_results": [],
  "reranked_results": [],
  "context": "...",
  "answer": "...",
  "latency_ms": {},
  "config": {}
}
```

这非常重要。

当答案错了，你可以按顺序排查：

1. transformed query 是否偏离意图？
2. BM25 是否召回相关文档？
3. Dense 是否召回相关文档？
4. Fusion 是否把正确文档排到前面？
5. Reranker 是否把正确文档排到前面？
6. Context 是否包含答案？
7. 生成阶段是否幻觉？

---

## 14. 评估脚本解释

评估数据在：

```text
data/eval/eval_queries.jsonl
```

示例：

```json
{"query_id":"q003","query":"为什么生产级 RAG 需要 BM25 和 Embedding 混合检索？","relevant_sources":["data/raw/03_hybrid_retrieval.md"],"query_type":"hybrid"}
```

每条 query 标注：

1. `query_id`
2. `query`
3. `relevant_sources`
4. `query_type`

评估脚本对比五种方法：

| 方法 | 含义 |
|---|---|
| `bm25_only` | 只用 BM25 |
| `dense_only` | 只用 dense retrieval |
| `hybrid_rrf` | BM25 + Dense + RRF |
| `hybrid_rerank` | Hybrid 后再 rerank |
| `transform_hybrid_rerank` | Query Transformation + Hybrid + Rerank |

指标包括：

| 指标 | 含义 |
|---|---|
| Hit@1 | 第一名是否命中相关 source |
| Hit@3 | 前 3 是否命中相关 source |
| Hit@5 | 前 5 是否命中相关 source |
| Recall@5 | 前 5 覆盖了多少相关 source |
| MRR | 第一个相关结果排得多靠前 |

---

## 15. 你应该如何改造成自己的第一周项目

### 第一步：替换数据源

把你第一周的文档放到：

```text
data/raw/
```

如果是 PDF，建议先用第 13 天高级数据处理方式解析成 Markdown 或 normalized blocks。

### 第二步：确认 chunk_id

你的 chunk 必须稳定：

```text
doc_id::chunk_000
doc_id::chunk_001
```

不要每次运行都随机生成 ID，否则无法评估和增量更新。

### 第三步：替换 Embedding

把 `HashingEmbedder` 替换成真实模型：

```python
class RealEmbedder:
    def embed_query(self, text: str) -> list[float]:
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...
```

可以用：

1. OpenAI `text-embedding-3-small`
2. BGE 中文模型
3. GTE
4. E5

### 第四步：接入 Milvus

先用：

```powershell
python src/rag_v2_upgrade_demo.py --vector-backend milvus --rebuild-milvus
```

跑通后，再把 Milvus 相关代码拆到你自己的：

```text
vectorstores/milvus_store.py
retrievers/milvus_retriever.py
```

### 第五步：替换 Reranker

把 `LightweightReranker` 换成本地 BGE：

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("BAAI/bge-reranker-base")
scores = model.predict([(query, candidate.text) for candidate in candidates])
```

### 第六步：保留 trace

无论你怎么改，都要保留：

```text
retrieval_trace.jsonl
```

这是定位 RAG 问题最重要的工具。

---

## 16. 常见实验命令

只跑原始 query：

```powershell
python src/rag_v2_upgrade_demo.py --strategies original
```

开启 HyDE 和 Multi-Query：

```powershell
python src/rag_v2_upgrade_demo.py --strategies original,hyde,multi_query
```

增加候选数：

```powershell
python src/rag_v2_upgrade_demo.py --bm25-top-k 10 --dense-top-k 10 --fusion-top-k 15 --rerank-top-n 5
```

查询 Milvus：

```powershell
python src/rag_v2_upgrade_demo.py --vector-backend milvus --query "Milvus 在 RAG 中负责什么？"
```

重建 Milvus collection：

```powershell
python src/rag_v2_upgrade_demo.py --vector-backend milvus --rebuild-milvus
```

运行评估：

```powershell
python src/evaluate_rag_v2.py
```

---

## 17. 这份代码和生产系统的差距

这份代码是教学版，不是最终生产版。

需要继续升级的地方：

1. HashingEmbedder 替换成真实 embedding 模型。
2. LightweightReranker 替换成 BGE / Cohere / CrossEncoder。
3. BM25 中文分词换成更专业方案。
4. Chunker 支持 PDF block、table、image。
5. Milvus schema 增加 tenant_id、page、metadata。
6. Config 真正从 YAML 读取。
7. Trace 接入日志系统。
8. Evaluation 增加 NDCG 和 answer faithfulness。
9. Generator 接入真实 LLM。
10. 增加缓存、错误重试和并发。

但它已经把生产级 RAG 的关键骨架搭出来了：

```text
Query Transformation
  -> BM25 + Dense/Milvus
  -> RRF Fusion
  -> Rerank
  -> Context Builder
  -> Answer with Sources
  -> Trace
  -> Evaluation
```

---

## 18. 最终理解

第一周 RAG 是一条直线：

```text
query -> vector search -> answer
```

第 14 天升级后的 RAG 是一条可调试的工程链路：

```text
query
  -> query transformation
  -> sparse retrieval
  -> dense retrieval / Milvus
  -> fusion
  -> rerank
  -> context
  -> answer
  -> trace
  -> evaluation
```

真正的提升不只是多用了 Milvus 或 Reranker，而是你现在能回答：

1. 这个答案来自哪些 chunk？
2. 这些 chunk 是哪一路检索召回的？
3. fusion 为什么把它排到前面？
4. reranker 为什么重排？
5. 升级后 Hit@K、Recall@K、MRR 有没有变化？
6. 如果答案错了，下一步应该查哪里？

这就是从 RAG demo 走向 RAG 工程的关键一步。

