# Reranker 二阶段精排

Reranker 是 RAG 检索链路中的二阶段排序模块。Retriever 负责从大规模语料中快速召回候选，Reranker 负责在较小候选集上判断 query 和 chunk 的细粒度相关性。

Reranker 通常处理 fusion 后的 top 20、top 50 或 top 100 候选，然后输出 top 3、top 5 或 top 10 作为最终上下文。

Reranker 不能弥补完全没有召回的问题。如果正确 chunk 没有进入候选集，再强的 reranker 也无法把它排到前面。它主要提升 MRR、NDCG 和最终进入 LLM 上下文的 top-k 质量。

常见 reranker 包括 BGE Reranker、Cohere Rerank、Cross Encoder 和 LLM Rerank。中文或中英混合知识库常优先考虑 BGE 系列 reranker。

如果 rerank 后效果变差，常见原因包括候选召回太少、chunk 缺少标题上下文、reranker 模型语言不匹配、query 太复杂没有拆解、rerank_top_n 设置过小。

