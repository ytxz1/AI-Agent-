# LlamaIndex Implementation Notes

在 LlamaIndex 中，可以使用 BM25Retriever 构建 sparse retrieval：

```python
from llama_index.retrievers.bm25 import BM25Retriever

bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=10,
)
```

向量检索可以使用 VectorStoreIndex：

```python
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex(nodes)
vector_retriever = index.as_retriever(similarity_top_k=10)
```

混合检索可以使用 QueryFusionRetriever：

```python
from llama_index.core.retrievers import QueryFusionRetriever

hybrid_retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=20,
    num_queries=1,
)
```

CohereRerank 可以作为 node_postprocessor 接入 query engine：

```python
from llama_index.postprocessor.cohere_rerank import CohereRerank

reranker = CohereRerank(
    top_n=5,
    model="rerank-english-v2.0",
    api_key=os.getenv("COHERE_API_KEY"),
)
```

完整 query engine：

```python
from llama_index.core.query_engine import RetrieverQueryEngine

query_engine = RetrieverQueryEngine.from_args(
    retriever=hybrid_retriever,
    node_postprocessors=[reranker],
)
```

调试时应该分别打印 bm25_results、dense_results、hybrid_results 和 reranked_results。
这样可以判断问题出在召回、融合、重排还是生成。
