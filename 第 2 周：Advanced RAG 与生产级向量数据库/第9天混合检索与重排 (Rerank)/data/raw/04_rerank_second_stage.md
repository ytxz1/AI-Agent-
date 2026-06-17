# Rerank as Second Stage Ranking

Rerank 是 RAG 检索链路中的第二阶段排序。第一阶段 retriever 负责从大规模知识库中快速召回候选，
第二阶段 reranker 负责对 query 和 candidate chunk 做更精细的相关性判断。

Reranker 通常使用 cross-encoder 模型或外部 API，例如 CohereRerank、bge-reranker-base、
cross-encoder/ms-marco-MiniLM-L-6-v2。与 embedding retriever 不同，cross-encoder 会同时读取
query 和 chunk，因此能判断更细粒度的语义关系。

Rerank 的位置通常是：

Query -> BM25 Retriever + Dense Retriever -> Fusion -> Reranker -> Top-N Context -> LLM

Reranker 不能解决“正确答案完全没有被召回”的问题。它只能在候选集合里重新排序。
所以生产级系统常常先扩大 top_k，例如 BM25 top 30、Dense top 30、Fusion top 50，
再由 reranker 选出 top 5 或 top 10。

Rerank 可以提升 MRR、NDCG 和最终上下文质量，但会增加延迟和成本。
因此需要对 rerank_top_n、fusion_top_k、候选文本长度和缓存策略做权衡。
