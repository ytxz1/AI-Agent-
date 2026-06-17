# BM25 + Embedding 混合检索

生产级 RAG 通常不会只使用向量检索，而是结合 BM25 稀疏检索和 Embedding 稠密检索。

BM25 擅长精确词匹配，例如函数名、参数名、文件名、版本号、错误码、产品名和技术术语。Embedding 检索擅长语义相似、自然语言表达、概念解释和同义改写。

Hybrid Retrieval 的基本流程是：同一个 query 同时进入 BM25 Retriever 和 Dense Retriever，两路分别返回 top_k 候选 chunk，然后使用 Reciprocal Rank Fusion，也就是 RRF，根据排名位置融合结果。

RRF 的公式是：score(doc) = sum(1 / (k + rank_i(doc)))。它不直接比较 BM25 分数和向量相似度分数，因为两者不在同一个尺度上。

混合检索的核心收益是提高 Recall@K。BM25 找到精确匹配，Embedding 找到语义相关，两者融合后候选集通常比单一路径更稳定。

