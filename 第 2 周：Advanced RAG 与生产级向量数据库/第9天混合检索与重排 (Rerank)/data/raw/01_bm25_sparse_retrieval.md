# BM25 Sparse Retrieval

BM25 是一种经典的 sparse retrieval 方法。它依赖 query terms 和 document terms 的词项匹配，
特别擅长处理专有名词、函数名、参数名、错误码、产品型号、字段名和 API 名称。

在 RAG 系统中，BM25Retriever 常用于弥补 dense embedding retrieval 对精确符号不稳定的问题。
例如用户查询 `BM25Retriever.from_defaults`、`similarity_top_k`、`Connection refused`、
`rerank-english-v2.0` 这类字符串时，BM25 往往可以直接命中文档。

BM25 的核心分数由三部分影响：

1. term frequency：词项在当前 chunk 中出现得越多，分数通常越高。
2. inverse document frequency：越稀有的词项越重要。
3. document length normalization：过长文档会被适度惩罚，避免长文档因为词多而天然占优。

但是 BM25 不理解深层语义。如果用户问“如何减少 RAG 上下文噪声”，而文档只写了
“reranker improves relevance ranking”，BM25 可能无法很好召回。
