# Embedding Dense Retrieval

Embedding retrieval 会把 query 和 chunk 编码成向量，然后用 cosine similarity、dot product
或 L2 distance 找到语义上接近的内容。它不要求 query 和文档共享完全相同的关键词。

Dense retrieval 的优势是理解语义相似、自然语言改写和抽象问题。例如：

- “怎样提高 RAG 的上下文相关性？”
- “为什么只靠关键词检索不够？”
- “检索阶段和排序阶段分别解决什么问题？”

这些问题不一定包含文档里的原始术语，但 embedding 模型仍然可能找到相关 chunk。

Dense retrieval 的弱点是对精确实体、数字、字段名、函数名不一定稳定。
如果用户查询 `QueryFusionRetriever` 或 `rerank_top_n`，embedding 可能认为另一个语义相近
但不包含该参数的段落也很相关。

生产系统通常不会只使用 dense retrieval，而是把 dense retriever 和 sparse retriever 组合起来。
