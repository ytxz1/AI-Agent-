# 第 9 天：混合检索与重排（Hybrid Retrieval + Rerank）

> 今日主题：实现 BM25 + Embedding 混合检索，并集成 Reranker，理解为什么生产级 RAG 通常不是“只做向量检索”，而是把多个检索信号组合起来，再通过重排器提升最终上下文质量。

## 1. 今日学习目标

完成今天后，你应该能够：

1. 解释 BM25、Embedding 检索、Hybrid Retrieval、Rerank 在 RAG 链路中的位置。
2. 实现一个最小可运行的混合检索流程：文档加载、切分、索引、BM25 召回、向量召回、候选融合、重排、生成。
3. 使用 LlamaIndex 的 `BM25Retriever`、`VectorStoreIndex`、`QueryFusionRetriever` 和 `node_postprocessors` 搭建完整链路。
4. 理解 RRF（Reciprocal Rank Fusion）和分数归一化融合的差异。
5. 集成 Cohere Rerank 或本地 cross-encoder reranker，并知道何时选择在线 API、何时选择本地模型。
6. 设计一个可复现实验，对比：
   - 仅 BM25
   - 仅 Embedding
   - BM25 + Embedding
   - BM25 + Embedding + Rerank
7. 为后续第 10-11 天 RAG 评估体系准备可量化指标：Recall@K、MRR、NDCG、命中率、回答引用覆盖率。

## 2. 参考资料

### 2.1 LlamaIndex Cohere Rerank 示例

参考地址：

https://developers.llamaindex.ai/python/examples/node_postprocessor/CohereRerank/

官方文档中的核心思想是：先由 retriever 取回一批候选 nodes，再使用 `CohereRerank` 作为 node postprocessor 对候选节点重新排序，只保留 `top_n` 个最相关节点进入后续生成阶段。

关键接口形态：

```python
from llama_index.postprocessor.cohere_rerank import CohereRerank

reranker = CohereRerank(
    top_n=3,
    model="rerank-english-v2.0",
    api_key="YOUR_COHERE_API_KEY",
)

reranked_nodes = reranker.postprocess_nodes(nodes)
```

在 query engine 中的典型使用方式：

```python
query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[reranker],
)
```

这说明 reranker 不是替代 retriever，而是位于 retriever 之后的第二阶段排序模块。

### 2.2 Modular RAG 论文

参考地址：

https://arxiv.org/pdf/2407.21059

论文主题是 Modular RAG：将复杂 RAG 系统拆解为可组合、可替换、可调度的模块。今天的混合检索与重排，正好可以映射为一个模块化 RAG 子图：

```text
Query
  -> Query Preprocess
  -> Sparse Retriever: BM25
  -> Dense Retriever: Embedding
  -> Fusion Operator
  -> Rerank Operator
  -> Context Builder
  -> Generator
  -> Evaluator
```

这比传统“retrieve then generate”更接近生产系统，因为每个模块都可以单独评估、替换和调参。

### 2.3 LlamaIndex BM25 + Hybrid Retrieval 文档

参考地址：

https://developers.llamaindex.ai/python/framework/integrations/retrievers/bm25_retriever/

官方文档给出了三个关键点：

1. `BM25Retriever.from_defaults()` 可以直接基于 nodes 或 docstore 创建 BM25 检索器。
2. `BM25Retriever` 可以持久化到磁盘。
3. BM25 与向量检索可以通过 `QueryFusionRetriever` 组合，形成 sparse + dense hybrid retrieval。

## 3. 核心概念地图

### 3.1 BM25 检索

BM25 是经典稀疏检索算法，依赖词项匹配。它关注：

- 查询词是否在文档中出现。
- 查询词在文档中出现的频率。
- 查询词在整个语料中是否稀有。
- 文档长度是否过长。

BM25 的优势：

- 对专有名词、编号、函数名、错误码、表字段、产品型号非常敏感。
- 不依赖 embedding 模型。
- 成本低、可解释性强。
- 对精确关键词查询很强。

BM25 的不足：

- 不能很好理解同义表达。
- 不能很好处理语义相近但字面不相同的问题。
- 对跨语言、改写、抽象概念类查询较弱。

典型适用问题：

```text
"Milvus IVF_FLAT 参数怎么设置？"
"报错 Connection refused 是哪里来的？"
"文档里有没有提到 rerank-english-v2.0？"
"metadata filter 的参数名是什么？"
```

### 3.2 Embedding 检索

Embedding 检索是稠密语义检索。它会把 query 和文档 chunk 编码为向量，然后根据余弦相似度、点积或 L2 距离查找语义接近的内容。

Embedding 检索的优势：

- 能理解语义相似。
- 对改写、概念查询、自然语言问题更友好。
- 能找出没有共享关键词但语义相关的片段。

Embedding 检索的不足：

- 对精确术语、数字、代码符号、错误码可能不稳定。
- 对 chunk 边界和 embedding 模型质量敏感。
- 召回结果的解释性弱于 BM25。
- 存在“语义相似但事实不相关”的风险。

典型适用问题：

```text
"为什么只用向量检索会漏掉精确配置项？"
"如何提升 RAG 的上下文相关性？"
"检索阶段和重排阶段分别解决什么问题？"
```

### 3.3 Hybrid Retrieval

Hybrid Retrieval 是把多种召回信号组合起来。今天重点是：

```text
BM25 sparse retrieval + Embedding dense retrieval
```

目标不是证明谁更强，而是利用二者互补：

| 检索方式 | 擅长 | 容易失败 |
|---|---|---|
| BM25 | 精确词、术语、代码、数字、实体 | 同义词、改写、抽象语义 |
| Embedding | 语义相似、自然语言、概念召回 | 精确符号、低频专名、数值约束 |
| Hybrid | 同时覆盖词法与语义 | 需要融合、去重、调参 |

### 3.4 Rerank

Rerank 是第二阶段排序：

```text
第一阶段：Retriever 负责“广召回”
第二阶段：Reranker 负责“精排序”
```

为什么不直接让 reranker 查全库？

- Reranker 通常是 cross-encoder 或外部 API，计算成本高。
- 它需要 query 和每个 candidate 成对输入。
- 所以它适合处理 top 20、top 50、top 100 候选，而不是百万级文档。

Reranker 的价值：

- 重新判断 query 与 chunk 的细粒度相关性。
- 修正 BM25 或 embedding 的粗排错误。
- 减少进入 LLM 上下文的噪声。
- 在固定上下文窗口下提升回答质量。

## 4. 今日最终系统架构

### 4.1 总体链路

```text
用户问题
  |
  v
查询预处理
  |
  +--------------------+
  |                    |
  v                    v
BM25 Retriever     Vector Retriever
  |                    |
  v                    v
BM25 Top-K         Dense Top-K
  |                    |
  +---------+----------+
            |
            v
候选去重与融合
            |
            v
Reranker Top-N
            |
            v
上下文构造
            |
            v
LLM 生成回答
            |
            v
答案 + 引用 + 检索调试信息
```

### 4.2 模块边界

建议将系统拆成这些模块：

| 模块 | 输入 | 输出 | 责任 |
|---|---|---|---|
| Loader | 文件路径 | Documents | 加载原始资料 |
| Splitter | Documents | Nodes | 切 chunk，保留 metadata |
| Sparse Index | Nodes | BM25Retriever | 构建 BM25 检索器 |
| Dense Index | Nodes | VectorStoreIndex | 构建向量索引 |
| Hybrid Retriever | Query | Candidate Nodes | 多路召回与融合 |
| Reranker | Query + Candidates | Reranked Nodes | 第二阶段排序 |
| Generator | Query + Context | Answer | 基于上下文生成回答 |
| Evaluator | Query + Gold / Answer | Metrics | 计算检索与回答指标 |

## 5. 推荐目录结构

第 9 天目录建议这样组织：

```text
第9天混合检索与重排 (Rerank)/
  README.md
  requirements.txt
  .env.example
  data/
    raw/
      README.md
  storage/
    chroma/
    bm25/
    docstore.json
  src/
    01_prepare_data.py
    02_bm25_retrieval.py
    03_embedding_retrieval.py
    04_hybrid_retrieval.py
    05_rerank.py
    06_query_engine.py
    07_evaluate_retrieval.py
  notebooks/
    hybrid_retrieval_rerank.ipynb
  configs/
    retrieval.yaml
  outputs/
    retrieval_debug.jsonl
    evaluation_report.md
```

如果你今天只想先做文档和最小实验，最低限度可以保留：

```text
第9天混合检索与重排 (Rerank)/
  README.md
  requirements.txt
  .env.example
  data/raw/
  src/hybrid_rerank_demo.py
```

## 6. 依赖安装

### 6.1 在线 API 版本

适合你要体验 Cohere Rerank 或 OpenAI Embedding 的情况。

```txt
llama-index
llama-index-retrievers-bm25
llama-index-vector-stores-chroma
llama-index-postprocessor-cohere-rerank
llama-index-embeddings-openai
llama-index-llms-openai
chromadb
cohere
pystemmer
python-dotenv
rich
nest-asyncio
```

`.env.example`：

```bash
OPENAI_API_KEY=your_openai_api_key
COHERE_API_KEY=your_cohere_api_key
```

### 6.2 本地模型版本

适合你想降低 API 成本，或希望离线跑 rerank。

```txt
llama-index
llama-index-retrievers-bm25
llama-index-vector-stores-chroma
llama-index-embeddings-huggingface
chromadb
sentence-transformers
pystemmer
python-dotenv
rich
nest-asyncio
```

本地 embedding 可选：

```text
BAAI/bge-small-zh-v1.5
BAAI/bge-base-zh-v1.5
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

本地 reranker 可选：

```text
BAAI/bge-reranker-base
BAAI/bge-reranker-large
cross-encoder/ms-marco-MiniLM-L-6-v2
```

中文资料建议优先考虑 BGE 系列。

## 7. 详细实现计划

### 阶段 0：明确实验问题

目标：不要一上来就写 RAG，而是先定义今天要证明什么。

今天建议验证这四个问题：

1. 当 query 包含专有名词、参数名、代码名时，BM25 是否比 embedding 更稳？
2. 当 query 是抽象语义问题时，embedding 是否能召回 BM25 漏掉的 chunk？
3. Hybrid 是否能提高 Recall@K？
4. Rerank 是否能把真正相关的 chunk 推到 top 3？

建议准备 8-15 个测试问题，每个问题手动标注 1-3 个相关文档片段或相关文件。

### 阶段 1：准备语料

输入资料可以来自：

- 你前几天的 Advanced RAG 学习文档。
- LlamaIndex 官方文档片段。
- 自己整理的 Markdown 笔记。
- 一组模拟企业知识库文档。

语料建议满足：

1. 有精确关键词，例如 `BM25Retriever`、`QueryFusionRetriever`、`CohereRerank`。
2. 有语义表达，例如“为什么需要二阶段排序”。
3. 有容易混淆的内容，例如“rerank”和“reorder”、“fusion”和“merge”。
4. 有中英文混合内容，贴近真实工程资料。

### 阶段 2：文档切分

推荐初始参数：

```python
chunk_size = 512
chunk_overlap = 80
```

调参原则：

| 参数 | 太小的问题 | 太大的问题 | 建议 |
|---|---|---|---|
| chunk_size | 上下文不完整 | 噪声多，rerank 难 | 512-1024 |
| overlap | 跨段信息丢失 | 重复候选多 | 10%-20% chunk_size |
| metadata | 无法溯源 | 无法过滤 | 保留 source、file_name、page、section |

### 阶段 3：构建 BM25 Retriever

核心代码：

```python
from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer

bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=10,
    stemmer=Stemmer.Stemmer("english"),
    language="english",
)
```

如果中文语料较多，要注意：

- 英文 stemming 对中文没有帮助。
- 中文需要更好的分词策略。
- 可以先用英文/中英混合语料完成主流程，再单独优化中文 BM25 tokenization。
- 生产中中文 sparse retrieval 更常用 Elasticsearch/OpenSearch + IK 分词，或支持中文 tokenization 的稀疏检索方案。

### 阶段 4：构建 Embedding Retriever

核心代码：

```python
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex(nodes)

vector_retriever = index.as_retriever(
    similarity_top_k=10,
)
```

如果使用 Chroma 持久化：

```python
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

db = chromadb.PersistentClient(path="./storage/chroma")
collection = db.get_or_create_collection("dense_vectors")
vector_store = ChromaVectorStore(chroma_collection=collection)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
)
```

### 阶段 5：实现 Hybrid Fusion

#### 方案 A：使用 LlamaIndex QueryFusionRetriever

推荐先使用官方组件跑通：

```python
from llama_index.core.retrievers import QueryFusionRetriever

hybrid_retriever = QueryFusionRetriever(
    [
        vector_retriever,
        bm25_retriever,
    ],
    similarity_top_k=20,
    num_queries=1,
    use_async=True,
)

candidate_nodes = hybrid_retriever.retrieve(query)
```

优点：

- 快速跑通。
- 接近官方推荐路径。
- 后续容易加入 query rewrite、多查询融合。

注意：

- `num_queries=1` 表示不做 LLM query generation，只融合原始 query。
- 如果开启多 query，需要额外考虑成本、延迟和 query 改写质量。

#### 方案 B：手写 RRF 融合

为了理解原理，建议自己实现一次 RRF：

```python
def reciprocal_rank_fusion(result_lists, k=60):
    fused = {}

    for results in result_lists:
        for rank, node_with_score in enumerate(results, start=1):
            node_id = node_with_score.node.node_id
            if node_id not in fused:
                fused[node_id] = {
                    "node": node_with_score.node,
                    "score": 0.0,
                    "sources": [],
                }

            fused[node_id]["score"] += 1.0 / (k + rank)
            fused[node_id]["sources"].append(
                {
                    "rank": rank,
                    "original_score": node_with_score.score,
                }
            )

    ranked = sorted(
        fused.values(),
        key=lambda item: item["score"],
        reverse=True,
    )

    return ranked
```

RRF 的直觉：

- 不直接比较 BM25 分数和 embedding 分数，因为它们不是同一尺度。
- 只使用排名位置。
- 一个 chunk 如果在多个检索器里都排得靠前，就会获得更高融合分。

RRF 常用公式：

```text
RRF(d) = sum(1 / (k + rank_i(d)))
```

其中：

- `d` 是文档或 chunk。
- `rank_i(d)` 是 d 在第 i 个检索器中的排名。
- `k` 是平滑常数，常见取值为 60。

### 阶段 6：接入 Reranker

#### 方案 A：Cohere Rerank

适合英文或多语言效果稳定、可接受外部 API 成本的场景。

```python
from llama_index.postprocessor.cohere_rerank import CohereRerank

reranker = CohereRerank(
    top_n=5,
    model="rerank-english-v2.0",
    api_key=os.getenv("COHERE_API_KEY"),
)

reranked_nodes = reranker.postprocess_nodes(
    candidate_nodes,
    query_str=query,
)
```

如果使用 query engine：

```python
from llama_index.core.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine.from_args(
    retriever=hybrid_retriever,
    node_postprocessors=[reranker],
)

response = query_engine.query(query)
```

#### 方案 B：本地 SentenceTransformerRerank

适合本地实验、中文模型、隐私要求更高的场景。

```python
from llama_index.core.postprocessor import SentenceTransformerRerank

reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n=5,
)
```

中文 reranker 可以考虑 BGE：

```python
from llama_index.core.postprocessor import SentenceTransformerRerank

reranker = SentenceTransformerRerank(
    model="BAAI/bge-reranker-base",
    top_n=5,
)
```

### 阶段 7：生成答案

Rerank 后只把 top N 节点交给 LLM。

推荐初始参数：

```text
BM25 top_k: 10
Embedding top_k: 10
Fusion top_k: 20
Rerank top_n: 5
LLM context nodes: 3-5
```

回答时要求模型：

1. 只基于检索上下文回答。
2. 不确定就说不知道。
3. 输出引用来源。
4. 对关键结论标明来自哪个 chunk 或文件。

示例 prompt：

```text
你是一个严谨的 RAG 学习助手。
请只根据给定上下文回答问题。
如果上下文不足，请明确说明“当前资料不足以回答”。
回答后列出使用到的来源文件和 chunk_id。
```

### 阶段 8：输出调试信息

每次 query 都建议保存调试信息：

```json
{
  "query": "为什么要在混合检索后加 rerank？",
  "bm25_top_k": [
    {"node_id": "n1", "rank": 1, "score": 3.42, "source": "rag.md"}
  ],
  "embedding_top_k": [
    {"node_id": "n8", "rank": 1, "score": 0.82, "source": "rerank.md"}
  ],
  "fused_top_k": [
    {"node_id": "n8", "rank": 1, "score": 0.031, "source": "rerank.md"}
  ],
  "reranked_top_n": [
    {"node_id": "n8", "rank": 1, "score": 0.97, "source": "rerank.md"}
  ]
}
```

调试信息的价值：

- 能看到 BM25 和 embedding 各自召回了什么。
- 能发现 fusion 是否把重复 chunk 合并了。
- 能判断 reranker 是否真的把正确内容推到了前面。
- 能为第 10-11 天评估体系提供数据。

## 8. 最小可运行示例

下面是一个单文件 demo 的建议写法。

```python
import os
from dotenv import load_dotenv

import Stemmer
import nest_asyncio

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.postprocessor.cohere_rerank import CohereRerank


def build_nodes(data_dir: str):
    documents = SimpleDirectoryReader(data_dir).load_data()
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=80)
    return splitter.get_nodes_from_documents(documents)


def build_hybrid_retriever(nodes, top_k: int = 10):
    index = VectorStoreIndex(nodes)
    vector_retriever = index.as_retriever(similarity_top_k=top_k)

    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=top_k,
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )

    hybrid_retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=top_k * 2,
        num_queries=1,
        use_async=True,
    )

    return hybrid_retriever


def build_query_engine(retriever, rerank_top_n: int = 5):
    reranker = CohereRerank(
        top_n=rerank_top_n,
        model="rerank-english-v2.0",
        api_key=os.getenv("COHERE_API_KEY"),
    )

    return RetrieverQueryEngine.from_args(
        retriever=retriever,
        node_postprocessors=[reranker],
    )


def main():
    nest_asyncio.apply()
    load_dotenv()

    nodes = build_nodes("./data/raw")
    retriever = build_hybrid_retriever(nodes, top_k=10)
    query_engine = build_query_engine(retriever, rerank_top_n=5)

    query = "为什么生产级 RAG 需要 BM25 和 Embedding 混合检索？"
    response = query_engine.query(query)

    print(response)
    print("\nSources:")
    for source in response.source_nodes:
        print(source.node.node_id, source.score, source.node.metadata)


if __name__ == "__main__":
    main()
```

## 9. 手写混合检索版本

如果想更深入理解，不依赖 `QueryFusionRetriever`，可以手动完成：

```python
def retrieve_with_debug(query, bm25_retriever, vector_retriever, reranker):
    bm25_nodes = bm25_retriever.retrieve(query)
    dense_nodes = vector_retriever.retrieve(query)

    fused_nodes = reciprocal_rank_fusion_nodes(
        [bm25_nodes, dense_nodes],
        rrf_k=60,
        top_k=20,
    )

    reranked_nodes = reranker.postprocess_nodes(
        fused_nodes,
        query_str=query,
    )

    return {
        "bm25": bm25_nodes,
        "dense": dense_nodes,
        "fused": fused_nodes,
        "reranked": reranked_nodes,
    }
```

手写版的收益：

- 更容易保存中间结果。
- 更容易观察每个模块的贡献。
- 更容易实现自定义权重。
- 更容易为第 10-11 天写评估脚本。

## 10. 融合策略详解

### 10.1 直接拼接

做法：

```text
候选 = BM25 top_k + Embedding top_k
去重后保留前 N
```

优点：

- 简单。
- 适合最初调试。

缺点：

- 没有真正排序。
- 谁在前面取决于拼接顺序。
- 很容易偏向某一个 retriever。

### 10.2 分数归一化后加权

做法：

```text
final_score = alpha * normalized_bm25_score + beta * normalized_dense_score
```

优点：

- 可调权重。
- 适合需要业务控制的场景。

缺点：

- BM25 分数和 embedding 分数分布不同。
- 不同 query 的分数范围可能变化很大。
- 归一化策略会影响结果。

推荐初始权重：

```text
alpha = 0.5
beta = 0.5
```

如果你的语料里有大量代码、参数、术语：

```text
alpha = 0.6
beta = 0.4
```

如果用户问题更偏自然语言语义：

```text
alpha = 0.4
beta = 0.6
```

### 10.3 RRF

推荐作为今天的主融合策略。

优点：

- 不需要比较不同检索器的原始分数。
- 对多路召回非常友好。
- 实现简单，鲁棒性好。

缺点：

- 丢失了原始分数强弱信息。
- 对 rank 非常敏感。
- 当某一路 retriever 质量很差时，仍可能引入噪声。

### 10.4 Rerank 前候选数量

建议：

| 语料规模 | BM25 top_k | Dense top_k | Fusion top_k | Rerank top_n |
|---|---:|---:|---:|---:|
| 小实验 | 5 | 5 | 10 | 3 |
| 日常开发 | 10 | 10 | 20 | 5 |
| 生产初版 | 30 | 30 | 50 | 5-10 |
| 高召回场景 | 50 | 50 | 100 | 10 |

原则：

- Rerank 前要给足候选，否则再强的 reranker 也救不回没召回的答案。
- Rerank 后要控制数量，否则 LLM 上下文会被噪声污染。
- 候选越多，rerank 成本和延迟越高。

## 11. Reranker 选择指南

| Reranker | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| Cohere Rerank | API 易用，效果稳定 | 外部调用成本，隐私依赖 | 英文、多语言、快速上线 |
| SentenceTransformer Cross-Encoder | 本地可控，成本低 | 需要模型下载和推理资源 | 本地实验、私有化 |
| BGE Reranker | 中文/中英场景常用 | 需要本地部署 | 中文知识库 |
| LLM Rerank | 可解释性强，可加入复杂规则 | 成本高，延迟高，格式不稳定 | 小候选集、复杂判断 |
| ColBERT | 细粒度 token 级匹配 | 系统复杂度更高 | 高质量检索系统 |

建议今天优先顺序：

1. 先用 `SentenceTransformerRerank` 本地跑通，不依赖 API。
2. 再接入 `CohereRerank`，对比线上 rerank 效果。
3. 最后抽象成可切换配置。

## 12. 推荐配置文件

`configs/retrieval.yaml`：

```yaml
data:
  raw_dir: ./data/raw
  persist_dir: ./storage

chunking:
  chunk_size: 512
  chunk_overlap: 80

retrieval:
  bm25_top_k: 10
  dense_top_k: 10
  fusion_top_k: 20
  fusion_method: rrf
  rrf_k: 60

rerank:
  enabled: true
  provider: cohere
  model: rerank-english-v2.0
  top_n: 5

generation:
  response_mode: compact
  cite_sources: true

debug:
  save_retrieval_trace: true
  output_path: ./outputs/retrieval_debug.jsonl
```

## 13. 评估设计

今天不用做到完整 RAGAS 或 DeepEval，但至少要为下一阶段打基础。

### 13.1 测试集格式

建议创建 `eval_queries.jsonl`：

```json
{"query": "BM25Retriever 如何持久化？", "relevant_sources": ["bm25_retriever.md"], "relevant_keywords": ["persist", "from_persist_dir"]}
{"query": "为什么混合检索比单独向量检索更适合生产 RAG？", "relevant_sources": ["hybrid_retrieval.md"], "relevant_keywords": ["BM25", "Embedding", "fusion"]}
{"query": "CohereRerank 在 LlamaIndex 中属于哪个模块？", "relevant_sources": ["rerank.md"], "relevant_keywords": ["node_postprocessors", "top_n"]}
```

### 13.2 检索指标

#### Hit@K

如果 top K 里出现任意一个相关文档，记为命中。

```text
Hit@K = 命中的 query 数 / 总 query 数
```

#### Recall@K

如果一个 query 有多个相关文档，Recall@K 衡量 top K 覆盖了多少相关文档。

```text
Recall@K = top K 中相关文档数量 / 全部相关文档数量
```

#### MRR

关注第一个相关结果出现得有多靠前。

```text
MRR = mean(1 / 第一个相关结果排名)
```

#### NDCG@K

关注排序质量。越相关的文档越靠前，分数越高。

适合有 0、1、2、3 多级相关性标注时使用。

### 13.3 今日对比表

建议最终输出：

| 方法 | Hit@3 | Hit@5 | Recall@5 | MRR | 平均延迟 | 备注 |
|---|---:|---:|---:|---:|---:|---|
| BM25 |  |  |  |  |  | 精确词强 |
| Embedding |  |  |  |  |  | 语义强 |
| Hybrid |  |  |  |  |  | 召回更稳 |
| Hybrid + Rerank |  |  |  |  |  | 排序最好 |

### 13.4 预期现象

通常你会看到：

1. BM25 在包含函数名、参数名、英文术语的问题上表现好。
2. Embedding 在自然语言改写问题上表现好。
3. Hybrid 的 Recall@K 通常高于单一路径。
4. Rerank 对 Hit@K 的提升不一定明显，因为 Hit@K 更看召回；但对 MRR、NDCG、top 3 质量通常有帮助。
5. 如果 Rerank 后效果变差，常见原因是候选召回太少、chunk 太碎、reranker 模型语言不匹配、query 太模糊。

## 14. 实验记录模板

每次实验记录这些参数：

```markdown
## 实验编号

- 日期：
- 语料：
- chunk_size：
- chunk_overlap：
- embedding_model：
- bm25_top_k：
- dense_top_k：
- fusion_method：
- fusion_top_k：
- reranker：
- rerank_top_n：

## 结果

| 方法 | Hit@3 | Hit@5 | Recall@5 | MRR | 平均延迟 |
|---|---:|---:|---:|---:|---:|

## 观察

1.
2.
3.

## 下一步

1.
2.
```

## 15. 常见问题与排查

### 15.1 BM25 结果很差

可能原因：

- 中文没有正确分词。
- chunk 太长，关键词密度被稀释。
- query 和文档用词不一致。
- 停用词或 stemming 配置不适合语料。

解决：

- 先用英文或中英混合技术文档跑通。
- 缩小 chunk_size。
- 对中文接入更合适的 tokenizer。
- 使用 Elasticsearch/OpenSearch 验证中文 sparse retrieval。

### 15.2 Embedding 结果很差

可能原因：

- embedding 模型不适合中文或技术领域。
- chunk 太短，语义不完整。
- query 太依赖精确符号。
- 文档 metadata 丢失导致结果看似相关但不可溯源。

解决：

- 更换中文或多语言 embedding 模型。
- 增大 chunk_size。
- 对 query 做改写，补充关键词。
- 保留标题、章节名、文件名到 chunk text 或 metadata。

### 15.3 Hybrid 没有提升

可能原因：

- 两路 retriever 召回结果高度重复。
- fusion_top_k 太小。
- 语料太小，单路已经足够。
- query 集太少，无法观察差异。

解决：

- 扩大测试集。
- 增加包含精确术语和语义改写的 query。
- 分别打印 BM25 与 embedding top_k。
- 使用 RRF 而不是直接拼接。

### 15.4 Rerank 后效果下降

可能原因：

- reranker 模型语言不匹配。
- reranker top_n 太小。
- 候选中本来没有正确 chunk。
- chunk 缺少标题上下文，reranker 无法判断。
- query 太长或包含多个意图。

解决：

- 中文语料用 BGE reranker。
- `fusion_top_k` 增大到 50。
- `rerank_top_n` 从 3 调到 5 或 10。
- 把标题、文件名、章节名拼入 node text。
- 对复杂 query 先拆解或改写。

## 16. 生产级思考

### 16.1 延迟预算

一次 Hybrid + Rerank 请求的延迟来自：

```text
T_total =
  T_query_preprocess
  + max(T_bm25, T_dense)
  + T_fusion
  + T_rerank
  + T_generation
```

优化方向：

- BM25 和 dense 并行执行。
- 索引持久化，避免每次重建。
- Rerank 只处理融合后的 top 20-50。
- 对高频 query 做缓存。
- 对长文档使用 metadata filter 缩小搜索范围。

### 16.2 成本控制

成本主要来自：

- Embedding 构建。
- Rerank API。
- LLM 生成。

控制方法：

- 文档增量更新，不重复 embedding。
- 对 rerank 输入做截断。
- 对相同 query + corpus_version 缓存 rerank 结果。
- 根据 query 类型决定是否启用 rerank。

### 16.3 可观测性

生产 RAG 必须记录：

- query。
- retriever 配置版本。
- 检索 top_k。
- rerank top_n。
- source node ids。
- source scores。
- prompt token 数。
- answer。
- 用户反馈。
- 延迟和错误。

否则无法判断问题来自：

- 没召回。
- 召回了但排序差。
- 排序对了但上下文构造差。
- 上下文对了但生成失败。

### 16.4 安全与权限

Hybrid Retrieval 也要遵守权限：

- BM25 和 vector retriever 都要应用同样的 metadata filter。
- Fusion 后不能混入无权限文档。
- Rerank 前后都不能泄露无权限 chunk。
- Debug trace 中不要记录敏感原文，至少在生产中要脱敏。

## 17. 今日任务清单

### 必做

- [ ] 准备 10 篇左右 Markdown 或文本资料。
- [ ] 使用 LlamaIndex 切分为 nodes。
- [ ] 实现 BM25 检索。
- [ ] 实现 embedding 检索。
- [ ] 使用 `QueryFusionRetriever` 实现 hybrid retrieval。
- [ ] 接入 `CohereRerank` 或 `SentenceTransformerRerank`。
- [ ] 打印每一步 top_k 结果。
- [ ] 对 8-15 个 query 做人工观察。

### 进阶

- [ ] 手写 RRF。
- [ ] 输出 `retrieval_debug.jsonl`。
- [ ] 实现 Hit@K、Recall@K、MRR。
- [ ] 对比不同 `chunk_size`。
- [ ] 对比不同 reranker。
- [ ] 加入 metadata filter。

### 挑战

- [ ] 根据 query 类型动态选择 retriever 权重。
- [ ] 对中文 BM25 接入更好的分词。
- [ ] 将 reranker 抽象为 provider，可切换 Cohere、本地 BGE、LLM rerank。
- [ ] 做一个小型 Gradio 或 Streamlit 检索调试界面。

## 18. 推荐今日时间安排

如果你有 2 小时：

| 时间 | 内容 |
|---|---|
| 0-20 分钟 | 阅读概念与架构 |
| 20-45 分钟 | 准备数据与切分 |
| 45-75 分钟 | 跑通 BM25 + Embedding |
| 75-100 分钟 | 接入 Hybrid Fusion |
| 100-120 分钟 | 接入 Rerank，人工观察结果 |

如果你有 4 小时：

| 时间 | 内容 |
|---|---|
| 0-30 分钟 | 阅读参考资料 |
| 30-60 分钟 | 准备语料与测试问题 |
| 60-100 分钟 | 实现两路 retriever |
| 100-140 分钟 | 实现 QueryFusionRetriever 与手写 RRF |
| 140-180 分钟 | 接入 Cohere 或本地 reranker |
| 180-220 分钟 | 输出 debug trace |
| 220-240 分钟 | 写实验记录和下一步计划 |

## 19. 面试与工程表达

如果面试官问：“为什么 RAG 不直接用向量检索？”

可以这样答：

```text
向量检索擅长语义召回，但对精确符号、专有名词、错误码、字段名不一定稳定。
BM25 这类 sparse retrieval 对词法匹配更强。
生产 RAG 通常会把 BM25 和 dense embedding 做 hybrid retrieval，先扩大候选召回，再用 reranker 做第二阶段精排。
这样可以同时提升召回率和 top-k 排序质量。
```

如果面试官问：“Rerank 解决什么问题？”

可以这样答：

```text
Retriever 的任务是从大规模语料里快速取回候选，它通常是粗排。
Reranker 的任务是在较小候选集上做更精细的 query-document 相关性判断。
它不能弥补完全没召回的问题，但能显著减少 top-k 噪声，把更相关的 chunk 放到 LLM 上下文前面。
```

如果面试官问：“Hybrid Retrieval 怎么融合分数？”

可以这样答：

```text
常见做法有三类：直接拼接去重、分数归一化加权、RRF。
BM25 和 embedding 的原始分数不在同一尺度，直接相加风险较高。
RRF 基于排名而不是原始分数，简单且鲁棒，适合作为初始方案。
生产中可以再根据 query 类型、业务反馈和评估指标调整权重。
```

## 20. 今日产出标准

今天完成后，建议你至少拥有：

1. 一个可运行的 hybrid retrieval demo。
2. 一份包含 8-15 个问题的测试集。
3. 一份检索调试输出。
4. 一张方法对比表。
5. 一段能讲清楚 BM25、Embedding、Fusion、Rerank 关系的总结。

最终你要能画出这条线：

```text
Query
 -> BM25 + Embedding 双路召回
 -> Fusion 合并与去重
 -> Rerank 精排
 -> Top-N Context
 -> LLM Answer with Sources
```

这就是今天的主线。
