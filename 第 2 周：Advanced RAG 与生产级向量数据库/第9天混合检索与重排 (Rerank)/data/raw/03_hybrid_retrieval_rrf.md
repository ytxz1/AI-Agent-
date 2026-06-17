# Hybrid Retrieval and RRF

Hybrid retrieval 指同时使用多种检索器召回候选文档。最常见的组合是：

BM25 sparse retriever + Embedding dense retriever

这两个检索器具有互补性。BM25 强在精确匹配，Embedding 强在语义匹配。
混合检索的目标不是让某一个检索器永远胜出，而是提高候选集合的 recall。

Reciprocal Rank Fusion，简称 RRF，是一种常用的融合策略。它不直接比较 BM25 分数和
Embedding 分数，因为两者分数尺度不同。RRF 只关心候选在每个检索器中的排名。

RRF 公式：

RRF(d) = sum(1 / (k + rank_i(d)))

如果一个 chunk 同时在 BM25 和 dense retrieval 中排名靠前，它的融合分数会更高。
如果某个 chunk 只被一路检索器召回，它也能进入候选集，但分数通常低一些。

在实际系统中，常见参数是：

- bm25_top_k = 10
- dense_top_k = 10
- fusion_top_k = 20
- rrf_k = 60

fusion_top_k 应该大于最终给 LLM 的 top_n，因为后面还要交给 reranker 精排。
