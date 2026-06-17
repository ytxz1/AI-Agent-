# 代码详解：BM25 + Embedding 混合检索与 Rerank 实现

这份文档专门解释第 9 天目录中的代码，不讲空泛概念，而是围绕“怎么跑、每段代码为什么这么写、输出怎么看、如何改造成生产版本”展开。

## 1. 本目录现在包含什么

```text
第9天混合检索与重排 (Rerank)/
  README.md
  代码详解：BM25+Embedding混合检索与Rerank实现.md
  requirements.txt
  .env.example
  eval_queries.jsonl
  configs/
    retrieval.yaml
  data/
    raw/
      01_bm25_sparse_retrieval.md
      02_embedding_dense_retrieval.md
      03_hybrid_retrieval_rrf.md
      04_rerank_second_stage.md
      05_llamaindex_implementation_notes.md
  src/
    hybrid_rerank_from_scratch.py
    evaluate_from_scratch.py
    hybrid_rerank_demo.py
```

其中最重要的是三个 Python 文件：

| 文件 | 作用 | 是否需要 API Key | 适合阶段 |
|---|---|---:|---|
| `src/hybrid_rerank_from_scratch.py` | 从零实现 BM25、Embedding-like 检索、RRF、Rerank | 否 | 学原理、调试链路 |
| `src/evaluate_from_scratch.py` | 批量评估不同检索方法的 Hit@K 和 MRR | 否 | 做实验对比 |
| `src/hybrid_rerank_demo.py` | LlamaIndex + BM25 + VectorStoreIndex + QueryFusionRetriever + CohereRerank | 是 | 接近真实工程 |

建议学习顺序：

1. 先跑 `hybrid_rerank_from_scratch.py`，理解每一步的输入输出。
2. 再跑 `evaluate_from_scratch.py`，理解为什么要量化评估。
3. 最后看 `hybrid_rerank_demo.py`，把手写逻辑映射到 LlamaIndex 组件。

## 2. 运行方式

### 2.1 运行从零实现版

进入第 9 天目录：

```bash
cd "第9天混合检索与重排 (Rerank)"
python src/hybrid_rerank_from_scratch.py
```

你也可以传入自己的问题：

```bash
python src/hybrid_rerank_from_scratch.py --query "RRF 为什么不用直接相加 BM25 和 Embedding 的分数？"
```

调整召回数量：

```bash
python src/hybrid_rerank_from_scratch.py \
  --query "LlamaIndex 如何接入 CohereRerank？" \
  --top-k 5 \
  --fusion-top-k 8 \
  --rerank-top-n 4
```

脚本会输出四段结果：

```text
=== BM25 sparse retrieval ===
=== Embedding-like TF-IDF retrieval ===
=== Hybrid retrieval with RRF ===
=== Reranked final contexts ===
```

同时会写入调试文件：

```text
outputs/from_scratch_retrieval_debug.jsonl
```

### 2.2 运行评估脚本

```bash
python src/evaluate_from_scratch.py
```

它会读取：

```text
eval_queries.jsonl
```

然后输出评估报告：

```text
outputs/from_scratch_evaluation_report.md
```

报告里会有：

```text
BM25
Embedding-like
Hybrid RRF
Hybrid RRF + Rerank
```

四种方法的 Hit@1、Hit@3、Hit@5 和 MRR。

### 2.3 运行 LlamaIndex 版

先安装依赖：

```bash
pip install -r requirements.txt
```

复制环境变量文件：

```bash
copy .env.example .env
```

在 `.env` 中填入：

```bash
OPENAI_API_KEY=your_openai_api_key
COHERE_API_KEY=your_cohere_api_key
```

运行：

```bash
python src/hybrid_rerank_demo.py
```

注意：这个版本会调用 OpenAI embedding 和 Cohere Rerank，所以需要网络和有效 API Key。

## 3. 示例语料如何设计

`data/raw/` 里有 5 份小文档：

| 文件 | 设计目的 |
|---|---|
| `01_bm25_sparse_retrieval.md` | 包含 `BM25Retriever.from_defaults`、`similarity_top_k` 等精确关键词 |
| `02_embedding_dense_retrieval.md` | 描述语义检索、自然语言改写、向量相似度 |
| `03_hybrid_retrieval_rrf.md` | 专门解释 Hybrid Retrieval 和 RRF 公式 |
| `04_rerank_second_stage.md` | 专门解释 Rerank 二阶段排序 |
| `05_llamaindex_implementation_notes.md` | 放 LlamaIndex 代码片段和 API 名称 |

这些语料不是随便写的，而是为了让你观察三类现象：

1. 精确术语查询时，BM25 往往更稳。
2. 语义描述查询时，Embedding-like 检索更容易补充相关片段。
3. Hybrid + Rerank 会让最终 top contexts 更适合送给 LLM。

## 4. 从零实现版整体流程

`src/hybrid_rerank_from_scratch.py` 的核心流程在 `run_pipeline()`：

```python
def run_pipeline(query: str, top_k: int, fusion_top_k: int, rerank_top_n: int) -> dict:
    documents = load_documents(DATA_DIR)
    chunks = split_into_chunks(documents)

    bm25 = BM25Retriever(chunks)
    embedding_like = TfidfEmbeddingLikeRetriever(chunks)
    reranker = LightweightReranker()

    bm25_results = bm25.retrieve(query, top_k=top_k)
    embedding_results = embedding_like.retrieve(query, top_k=top_k)
    fused_results = reciprocal_rank_fusion(
        [bm25_results, embedding_results],
        top_k=fusion_top_k,
        rrf_k=60,
    )
    reranked_results = reranker.rerank(query, fused_results, top_n=rerank_top_n)
    answer = answer_from_context(query, reranked_results)
```

这段就是今天主题的最小闭环：

```text
query
  -> BM25Retriever
  -> TfidfEmbeddingLikeRetriever
  -> reciprocal_rank_fusion
  -> LightweightReranker
  -> answer_from_context
```

真实生产系统中，`TfidfEmbeddingLikeRetriever` 会替换成真正的 embedding 模型，`LightweightReranker` 会替换成 Cohere、BGE reranker 或 cross-encoder。

## 5. 数据结构解释

### 5.1 Document

```python
@dataclass(frozen=True)
class Document:
    doc_id: str
    source: str
    text: str
```

`Document` 表示原始文件。

字段解释：

| 字段 | 示例 | 含义 |
|---|---|---|
| `doc_id` | `01_bm25_sparse_retrieval` | 文档 ID |
| `source` | `data/raw/01_bm25_sparse_retrieval.md` | 来源路径 |
| `text` | 文件全文 | 原始文本 |

真实工程中，`Document` 还可以包含：

- `tenant_id`
- `user_group`
- `created_at`
- `doc_type`
- `permission_level`
- `version`

### 5.2 Chunk

```python
@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)
```

`Chunk` 是检索的基本单位。RAG 很少直接检索整篇文档，因为整篇文档太长、噪声太多、无法精准引用。

`chunk_id` 示例：

```text
03_hybrid_retrieval_rrf::chunk_000
```

它表示来自 `03_hybrid_retrieval_rrf.md` 的第 0 个切片。

### 5.3 SearchResult

```python
@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    rank: int = 0
    method: str = ""
    debug: dict[str, float | int | str] = field(default_factory=dict)
```

`SearchResult` 表示一次检索返回的候选。

字段解释：

| 字段 | 作用 |
|---|---|
| `chunk` | 命中的文本片段 |
| `score` | 当前方法给出的分数 |
| `rank` | 当前方法内的排名 |
| `method` | `bm25`、`embedding_like`、`rrf_fusion`、`lightweight_rerank` |
| `debug` | 保存可解释信息，例如匹配词数量、融合来源、覆盖率 |

这个结构非常重要。生产 RAG 出问题时，如果没有 `debug` 信息，你很难判断是召回问题、融合问题还是 rerank 问题。

## 6. 文档加载与切分

### 6.1 加载文档

```python
def load_documents(data_dir: Path) -> list[Document]:
    files = sorted(
        path for path in data_dir.rglob("*") if path.suffix.lower() in {".md", ".txt"}
    )
```

这里读取 `data/raw/` 下所有 `.md` 和 `.txt` 文件。

代码中跳过了 `README.md`：

```python
if path.name.lower() == "readme.md":
    continue
```

原因是 `README.md` 是目录说明，不是知识库正文。如果把说明文件也放进检索库，会污染结果。

### 6.2 切分 chunk

```python
def split_into_chunks(
    documents: Iterable[Document],
    chunk_size: int = 700,
    chunk_overlap: int = 120,
) -> list[Chunk]:
```

参数解释：

| 参数 | 当前值 | 作用 |
|---|---:|---|
| `chunk_size` | 700 | 单个 chunk 最大字符数 |
| `chunk_overlap` | 120 | 相邻 chunk 重叠字符数 |

为什么需要 overlap？

因为答案可能跨段落边界。如果没有 overlap，重要上下文可能被切断。overlap 可以提高召回稳定性，但也会带来重复 chunk。

真实工程建议：

| 文档类型 | chunk_size | overlap |
|---|---:|---:|
| FAQ | 300-600 | 50-100 |
| 技术文档 | 600-1000 | 80-150 |
| 法律/合同 | 800-1500 | 150-300 |
| 代码说明 | 400-800 | 80-160 |

## 7. Tokenizer 解释

```python
def tokenize(text: str) -> list[str]:
    text = text.lower()
    english_terms = re.findall(r"[a-z][a-z0-9_\-\.]*|\d+(?:\.\d+)?", text)
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    return english_terms + chinese_chars
```

这个 tokenizer 同时处理：

- 英文单词：`retriever`
- 参数名：`similarity_top_k`
- 模型名：`rerank-english-v2.0`
- 数字：`60`
- 中文字符：`检`、`索`、`重`、`排`

为什么中文按单字切？

因为这是零依赖教学版。生产环境里更建议：

- `jieba`
- HanLP
- Elasticsearch IK 分词
- OpenSearch 中文 analyzer
- 专门的中文 sparse embedding 模型

### 7.1 Chinese bigram

```python
def make_char_ngrams(text: str, n: int = 2) -> list[str]:
    chars = re.findall(r"[\u4e00-\u9fff]", text)
    return ["".join(chars[i : i + n]) for i in range(max(0, len(chars) - n + 1))]
```

这个函数把中文字符变成二元组：

```text
混合检索 -> 混合、合检、检索
```

它用于 `TfidfEmbeddingLikeRetriever`，让“语义检索”的教学版本比单字匹配更稳定一点。

## 8. BM25Retriever 详解

### 8.1 初始化

```python
class BM25Retriever:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.term_freqs = [Counter(tokenize(chunk.text)) for chunk in chunks]
        self.doc_lengths = [sum(freqs.values()) for freqs in self.term_freqs]
        self.avg_doc_length = sum(self.doc_lengths) / max(1, len(self.doc_lengths))
        self.doc_freqs = self._build_doc_freqs()
        self.total_docs = len(chunks)
```

BM25 需要提前统计：

| 变量 | 含义 |
|---|---|
| `term_freqs` | 每个 chunk 内各词出现次数 |
| `doc_lengths` | 每个 chunk 的 token 数 |
| `avg_doc_length` | 平均 chunk 长度 |
| `doc_freqs` | 每个词出现在多少个 chunk 中 |
| `total_docs` | chunk 总数 |

### 8.2 IDF

```python
def idf(self, term: str) -> float:
    doc_freq = self.doc_freqs.get(term, 0)
    return math.log(1 + (self.total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
```

IDF 的直觉：

- 如果一个词在很多 chunk 中都出现，它区分度低。
- 如果一个词只在少数 chunk 中出现，它区分度高。

例如：

```text
RAG
检索
的
```

这些词可能到处都有，IDF 不高。

而：

```text
QueryFusionRetriever
rerank-english-v2.0
similarity_top_k
```

这些词更稀有，更能定位具体文档。

### 8.3 BM25 打分

```python
def score_chunk(self, query_terms: list[str], chunk_index: int) -> float:
    freqs = self.term_freqs[chunk_index]
    doc_length = self.doc_lengths[chunk_index]
    score = 0.0

    for term in query_terms:
        term_frequency = freqs.get(term, 0)
        if term_frequency == 0:
            continue

        numerator = term_frequency * (self.k1 + 1)
        denominator = term_frequency + self.k1 * (
            1 - self.b + self.b * doc_length / self.avg_doc_length
        )
        score += self.idf(term) * numerator / denominator
```

这段就是 BM25 的核心。

`k1` 控制词频饱和：

- 词出现 1 次和 2 次差异明显。
- 出现 20 次和 21 次差异不应该太大。

`b` 控制长度归一化：

- `b = 0`：不惩罚长文档。
- `b = 1`：强烈考虑文档长度。
- 常用值：`0.75`。

### 8.4 BM25 retrieve

```python
def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
    query_terms = tokenize(query)
    results = []

    for index, chunk in enumerate(self.chunks):
        score = self.score_chunk(query_terms, index)
        if score > 0:
            results.append(SearchResult(...))

    return rank_results(results, top_k=top_k)
```

注意：只有 `score > 0` 的 chunk 才会返回。如果 query 和 chunk 没有任何词项重合，BM25 不会召回。

这正是 BM25 的优点，也是缺点。

## 9. Embedding-like Retriever 详解

文件里叫：

```python
class TfidfEmbeddingLikeRetriever:
```

它不是真正的 neural embedding，而是 TF-IDF cosine 检索。为什么这么设计？

因为你今天的重点是理解 RAG 检索链路，不是先卡在模型下载、API Key 或网络环境上。这个类保持了“向量化 -> 余弦相似度 -> top_k”的结构，因此可以替代真实 embedding retriever 做教学。

### 9.1 初始化

```python
self.term_freqs = [Counter(analysis_terms(chunk.text)) for chunk in chunks]
self.doc_freqs = self._build_doc_freqs()
self.total_docs = len(chunks)
self.vectors = [self._to_tfidf_vector(freqs) for freqs in self.term_freqs]
self.norms = [vector_norm(vector) for vector in self.vectors]
```

它会把每个 chunk 转成 TF-IDF 向量：

```text
chunk text -> terms -> term frequency -> TF-IDF vector
```

### 9.2 为什么叫 Embedding-like

真实 embedding 检索是：

```text
query -> neural embedding vector
chunk -> neural embedding vector
cosine(query_vector, chunk_vector)
```

这里的教学版是：

```text
query -> TF-IDF vector
chunk -> TF-IDF vector
cosine(query_vector, chunk_vector)
```

它们都具有“向量相似度检索”的外形，所以非常适合讲清楚系统结构。

### 9.3 替换成真实 embedding 的位置

未来你只需要替换这个类：

```python
embedding_like = TfidfEmbeddingLikeRetriever(chunks)
```

替换成：

```python
embedding_retriever = OpenAIEmbeddingRetriever(chunks)
```

或：

```python
embedding_retriever = BGEEmbeddingRetriever(chunks)
```

后面的 RRF 和 Rerank 不需要变。

这就是模块化 RAG 的好处。

## 10. RRF 融合详解

核心函数：

```python
def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    top_k: int = 10,
    rrf_k: int = 60,
) -> list[SearchResult]:
```

调用方式：

```python
fused_results = reciprocal_rank_fusion(
    [bm25_results, embedding_results],
    top_k=fusion_top_k,
    rrf_k=60,
)
```

### 10.1 为什么不用原始分数相加

BM25 分数可能长这样：

```text
7.84
5.21
3.11
```

Embedding cosine 分数可能长这样：

```text
0.82
0.78
0.61
```

它们不是同一个尺度，直接相加会让 BM25 分数主导结果。

RRF 不看原始分数，只看排名：

```python
fused_scores[chunk_id] += 1.0 / (rrf_k + result.rank)
```

如果某个 chunk 在 BM25 第 1 名，在 Embedding 第 3 名：

```text
1 / (60 + 1) + 1 / (60 + 3)
```

如果另一个 chunk 只在 BM25 第 1 名出现：

```text
1 / (60 + 1)
```

前者会更高，因为它被多个检索器共同认可。

### 10.2 debug 信息

```python
debug={"sources": ",".join(sources_by_id[chunk_id])}
```

输出可能是：

```text
debug={'sources': 'bm25@1,embedding_like@2'}
```

这表示该 chunk 被两路检索器都召回了。

如果输出：

```text
debug={'sources': 'embedding_like@1'}
```

说明它只被 embedding-like 检索召回。

## 11. LightweightReranker 详解

真实 reranker 通常是 cross-encoder：

```text
input:  query + chunk
output: relevance score
```

本项目里的教学版是：

```python
class LightweightReranker:
```

它用一组可解释特征模拟 reranker。

### 11.1 特征 1：term coverage

```python
term_coverage = safe_divide(len(query_terms & chunk_terms), len(query_terms))
```

含义：

```text
query 中有多少词能在 chunk 中找到
```

如果 query 是：

```text
Reranker 在 RAG 链路中位于什么位置？
```

chunk 里也有：

```text
Rerank 是 RAG 检索链路中的第二阶段排序
```

那么 term coverage 会比较高。

### 11.2 特征 2：bigram coverage

```python
bigram_coverage = safe_divide(len(query_bigrams & chunk_bigrams), len(query_bigrams))
```

它对中文短语更友好。

比如：

```text
混合检索
```

会拆成：

```text
混合、合检、检索
```

比单字匹配更有辨识度。

### 11.3 特征 3：keyword coverage

```python
query_keyword_terms = {
    term for term in query_terms if re.search(r"[a-z0-9_\-\.]", term) and len(term) >= 3
}
```

这个特征专门照顾技术文档里的关键术语，例如：

```text
RRF
BM25
Embedding
QueryFusionRetriever
similarity_top_k
rerank-english-v2.0
```

为什么需要它？

因为中文 query 中会包含很多常见字，例如“为、什、么、检、索”。如果只看普通 `term_coverage`，一些包含大量常见字但没有核心术语的 chunk 也可能得高分。

`keyword_coverage` 会检查 query 中的英文、数字、下划线、连字符、点号术语有多少被 chunk 覆盖：

```python
keyword_coverage = safe_divide(
    len(query_keyword_terms & chunk_terms),
    len(query_keyword_terms),
)
```

对于技术 RAG，这个特征非常关键。

### 11.4 特征 4：first-stage score

```python
normalize_first_stage_score(candidate.score)
```

Reranker 不应该完全忽略前面的融合分数。RRF 已经表达了“多个 retriever 是否共同认可这个 chunk”，所以 reranker 可以把它作为一个弱特征。

### 11.5 特征 5：exact bonus

```python
exact_bonus = 1.0 if query.lower() in text else 0.0
```

如果 query 完整出现在 chunk 里，说明很可能高度相关。

真实 reranker 不会这么简单，但它会学习类似信号。

### 11.6 特征 6：title bonus

```python
title_bonus = safe_divide(
    sum(1 for term in query_keyword_terms if term in candidate.chunk.source.lower()),
    len(query_keyword_terms),
)
```

如果 query 里的专有术语出现在文件名中，给一点加分。例如查询 `RRF`，来源文件 `03_hybrid_retrieval_rrf.md` 本身就很有提示价值。

### 11.7 总分公式

```python
rerank_score = (
    0.35 * term_coverage
    + 0.25 * keyword_coverage
    + 0.15 * bigram_coverage
    + 0.15 * normalize_first_stage_score(candidate.score)
    + 0.07 * exact_bonus
    + 0.03 * title_bonus
)
```

权重解释：

| 特征 | 权重 | 原因 |
|---|---:|---|
| `term_coverage` | 0.35 | query-chunk 字面覆盖最直接 |
| `keyword_coverage` | 0.25 | 技术术语、参数名、模型名必须被重视 |
| `bigram_coverage` | 0.15 | 增强中文短语匹配 |
| `first_stage_score` | 0.15 | 保留 BM25 + Embedding 融合信号 |
| `exact_bonus` | 0.07 | 完整 query 匹配很少见，但很强 |
| `title_bonus` | 0.03 | 文件名相关性只能轻微加分 |

真实生产中，这些权重不是手写的，而是模型学出来的。

## 12. 输出怎么看

假设你运行：

```bash
python src/hybrid_rerank_from_scratch.py --query "RRF 为什么不直接相加 BM25 和 Embedding 的原始分数？"
```

你应该重点看四组结果。

### 12.1 BM25 sparse retrieval

如果 top1 是：

```text
data/raw/03_hybrid_retrieval_rrf.md
```

说明 BM25 命中了 `RRF`、`BM25`、`Embedding`、`分数` 等词。

### 12.2 Embedding-like TF-IDF retrieval

如果 top1 也是：

```text
data/raw/03_hybrid_retrieval_rrf.md
```

说明这个问题同时适合词法召回和向量相似度召回。

### 12.3 Hybrid retrieval with RRF

看 debug：

```text
debug={'sources': 'bm25@1,embedding_like@1'}
```

这表示两个检索器都把它排在第一。RRF 会强烈保留这个 chunk。

### 12.4 Reranked final contexts

Rerank 后 top1 应该更接近最终要给 LLM 的上下文。

如果 Rerank 后结果变差，检查：

1. query 是否太短。
2. chunk 是否太短或太长。
3. fusion_top_k 是否太小。
4. reranker 的语言特征是否适合中文。
5. 正确文档是否在 fusion 阶段已经出现。

## 13. 调试文件 JSONL 解释

运行后会生成：

```text
outputs/from_scratch_retrieval_debug.jsonl
```

每一行是一条 query 的完整检索 trace。

结构如下：

```json
{
  "query": "...",
  "num_documents": 5,
  "num_chunks": 5,
  "bm25": [],
  "embedding_like": [],
  "hybrid_rrf": [],
  "reranked": [],
  "answer": "..."
}
```

为什么用 JSONL？

- 每次 query 追加一行。
- 方便后续用 pandas 读取。
- 适合生产日志。
- 不需要一次性把整个 JSON 数组加载进内存。

生产中你应该额外记录：

- `request_id`
- `user_id`
- `corpus_version`
- `retriever_config_version`
- `latency_ms`
- `prompt_tokens`
- `completion_tokens`
- `user_feedback`

## 14. 评估脚本详解

`src/evaluate_from_scratch.py` 做了四件事：

1. 加载 `eval_queries.jsonl`。
2. 对每个 query 跑四种方法。
3. 计算 Hit@1、Hit@3、Hit@5、MRR。
4. 写出 Markdown 报告。

### 14.1 eval_queries.jsonl 格式

```json
{
  "query": "RRF 为什么不直接相加 BM25 和 Embedding 的原始分数？",
  "relevant_sources": ["data/raw/03_hybrid_retrieval_rrf.md"]
}
```

`relevant_sources` 是人工标注的正确来源。

真实项目中可以标注到更细粒度：

```json
{
  "query": "...",
  "relevant_chunk_ids": ["03_hybrid_retrieval_rrf::chunk_000"],
  "relevance_grade": 3
}
```

### 14.2 Hit@K

```python
def hit_at_k(results, relevant_sources: set[str], k: int) -> int:
    for result in results[:k]:
        if result.chunk.source in relevant_sources:
            return 1
    return 0
```

含义：

```text
top K 里只要出现一个相关文档，就算命中。
```

Hit@K 适合回答：

```text
检索系统有没有把正确资料捞上来？
```

### 14.3 MRR

```python
def first_relevant_rank(results, relevant_sources: set[str]) -> int | None:
    for rank, result in enumerate(results, start=1):
        if result.chunk.source in relevant_sources:
            return rank
    return None
```

MRR 计算：

```python
"mrr": 1 / rank if rank else 0.0
```

如果第一个相关结果排第 1：

```text
MRR = 1.0
```

如果第一个相关结果排第 3：

```text
MRR = 0.333
```

MRR 适合回答：

```text
正确资料是不是排得足够靠前？
```

这正是 rerank 关心的问题。

## 15. LlamaIndex 版如何对应手写版

手写版：

```python
bm25 = BM25Retriever(chunks)
embedding_like = TfidfEmbeddingLikeRetriever(chunks)
fused_results = reciprocal_rank_fusion([bm25_results, embedding_results])
reranked_results = reranker.rerank(query, fused_results)
```

LlamaIndex 版：

```python
bm25_retriever = BM25Retriever.from_defaults(...)
vector_retriever = index.as_retriever(...)
hybrid_retriever = QueryFusionRetriever([vector_retriever, bm25_retriever])
reranker = CohereRerank(...)
```

映射关系：

| 手写版 | LlamaIndex 版 |
|---|---|
| `Document` | `Document` |
| `Chunk` | `Node` |
| `BM25Retriever` | `llama_index.retrievers.bm25.BM25Retriever` |
| `TfidfEmbeddingLikeRetriever` | `VectorStoreIndex.as_retriever()` |
| `reciprocal_rank_fusion()` | `QueryFusionRetriever` |
| `LightweightReranker` | `CohereRerank` 或 `SentenceTransformerRerank` |
| `answer_from_context()` | `RetrieverQueryEngine` + LLM |

## 16. 生产化改造路线

### 16.1 替换 Embedding-like 检索

当前：

```python
embedding_like = TfidfEmbeddingLikeRetriever(chunks)
```

生产：

```python
embedding_model = OpenAIEmbedding(...)
vector_store = MilvusVectorStore(...)
vector_retriever = VectorIndexRetriever(...)
```

或：

```python
embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-zh-v1.5")
```

### 16.2 替换 LightweightReranker

当前：

```python
reranker = LightweightReranker()
```

Cohere：

```python
reranker = CohereRerank(top_n=5, model="rerank-english-v2.0")
```

本地 BGE：

```python
reranker = SentenceTransformerRerank(
    model="BAAI/bge-reranker-base",
    top_n=5,
)
```

### 16.3 增加 metadata filter

生产中必须支持：

```text
tenant_id = 当前租户
permission_group in 当前用户权限
doc_type in 用户选择范围
created_at >= 时间范围
```

注意：BM25 和 vector retriever 必须应用同一套权限过滤。不能只在 vector 检索上过滤，BM25 不过滤，否则 fusion 后可能泄露无权限文档。

### 16.4 增加缓存

可以缓存：

- query rewrite 结果。
- BM25 top_k。
- dense top_k。
- rerank 结果。
- 最终回答。

缓存 key 建议包含：

```text
query_hash
corpus_version
retriever_config_version
permission_scope
```

## 17. 常见实验问题

### 17.1 为什么我的 BM25 没结果？

可能是 query 和文档没有任何共同 token。

例如 query：

```text
怎样提升答案准确性？
```

文档写的是：

```text
reranker improves relevance ranking
```

BM25 不理解这两个表达相关。

解决：

- 加 query rewrite。
- 加同义词扩展。
- 使用 Hybrid Retrieval。

### 17.2 为什么 Rerank 没有提升 Hit@5？

因为 Hit@5 更看召回。Reranker 只负责候选内部排序，不能凭空召回新文档。

Rerank 更可能提升：

- Hit@1
- MRR
- NDCG@K
- 最终回答质量

### 17.3 为什么 Hybrid 有时不如单路？

可能原因：

- fusion_top_k 太小。
- 某一路检索器质量太差，引入噪声。
- query 本身非常适合单路检索。
- 测试集太小，波动明显。

解决：

- 增加 query 数量。
- 分 query 类型统计指标。
- 为不同 query 类型设置不同权重。
- 使用 rerank 压制融合噪声。

## 18. 今天你应该重点掌握的代码路径

最重要的 5 个函数或类：

```text
BM25Retriever.retrieve()
TfidfEmbeddingLikeRetriever.retrieve()
reciprocal_rank_fusion()
LightweightReranker.rerank()
evaluate_from_scratch.py 的 hit_at_k() 和 MRR 计算
```

把这五个点吃透，你就能说清楚生产级 RAG 检索链路的核心逻辑。

## 19. 你可以继续做的练习

1. 把 `chunk_size` 从 700 改成 300，观察结果变化。
2. 把 `fusion_top_k` 从 8 改成 3，观察 rerank 是否变差。
3. 在 `eval_queries.jsonl` 里新增 10 个问题。
4. 给 `LightweightReranker` 加一个 `section_title_bonus`。
5. 把 `TfidfEmbeddingLikeRetriever` 替换成真实 OpenAI embedding。
6. 把 `LightweightReranker` 替换成 BGE reranker。
7. 给每条输出增加 `latency_ms`。
8. 写一个 Streamlit 页面展示四阶段检索结果。

## 20. 一句话总结

今天这套代码的重点不是追求模型效果，而是把生产 RAG 的检索链路拆开，让你能清楚看到：

```text
BM25 负责精确词召回
Embedding 负责语义召回
RRF 负责多路候选融合
Reranker 负责二阶段精排
Evaluation 负责证明效果变化
```

这就是 BM25 + Embedding 混合检索与 Rerank 的工程主线。
