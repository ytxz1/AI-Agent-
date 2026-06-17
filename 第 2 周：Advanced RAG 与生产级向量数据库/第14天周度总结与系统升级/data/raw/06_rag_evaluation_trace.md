# RAG 评估与 Retrieval Trace

RAG 系统升级不能只凭感觉，需要使用评估集和指标证明改动是否有效。

检索评估常用指标包括 Hit@K、Recall@K、MRR 和 NDCG。Hit@K 判断 top K 中是否出现至少一个相关 chunk；Recall@K 衡量相关 chunk 覆盖率；MRR 关注第一个相关结果出现的位置；NDCG 关注排序质量。

除了指标，还需要 retrieval trace。Trace 应记录 original query、transformed queries、BM25 results、dense results、fusion results、reranked results、context chunks、answer sources 和每个阶段的 latency。

当答案错误时，trace 可以帮助定位问题来自 query 改写、BM25 召回、Milvus 召回、fusion 排序、reranker 精排、context builder 截断还是 LLM 生成。

生产级 RAG 的关键不是一次性调好，而是建立可观测、可评估、可复盘的优化闭环。

