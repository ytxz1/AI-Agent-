# RAG v2 Upgrade Evaluation Report

## Summary

| Method | Hit@1 | Hit@3 | Hit@5 | Recall@5 | MRR |
|---|---:|---:|---:|---:|---:|
| bm25_only | 0.875 | 1.000 | 1.000 | 1.000 | 0.938 |
| dense_only | 0.875 | 1.000 | 1.000 | 1.000 | 0.917 |
| hybrid_rrf | 0.875 | 1.000 | 1.000 | 1.000 | 0.917 |
| hybrid_rerank | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| transform_hybrid_rerank | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## Per Query Top-1

| Query ID | Query | Relevant | BM25 | Dense | Hybrid | Rerank | Transform+Hybrid+Rerank |
|---|---|---|---|---|---|---|---|
| q001 | 第一周基础 RAG 系统有哪些主要短板？ | data/raw/01_rag_v1_baseline.md | data/raw/01_rag_v1_baseline.md | data/raw/01_rag_v1_baseline.md | data/raw/01_rag_v1_baseline.md | data/raw/01_rag_v1_baseline.md | data/raw/01_rag_v1_baseline.md |
| q002 | Milvus 在生产级 RAG 中负责什么？ | data/raw/02_milvus_vector_database.md | data/raw/02_milvus_vector_database.md | data/raw/02_milvus_vector_database.md | data/raw/02_milvus_vector_database.md | data/raw/02_milvus_vector_database.md | data/raw/02_milvus_vector_database.md |
| q003 | 为什么生产级 RAG 需要 BM25 和 Embedding 混合检索？ | data/raw/03_hybrid_retrieval.md | data/raw/03_hybrid_retrieval.md | data/raw/03_hybrid_retrieval.md | data/raw/03_hybrid_retrieval.md | data/raw/03_hybrid_retrieval.md | data/raw/03_hybrid_retrieval.md |
| q004 | RRF 如何融合 BM25 和向量检索结果？ | data/raw/03_hybrid_retrieval.md | data/raw/03_hybrid_retrieval.md | data/raw/03_hybrid_retrieval.md | data/raw/03_hybrid_retrieval.md | data/raw/03_hybrid_retrieval.md | data/raw/03_hybrid_retrieval.md |
| q005 | Reranker 能解决什么问题，不能解决什么问题？ | data/raw/04_reranker_second_stage.md | data/raw/02_milvus_vector_database.md | data/raw/05_query_transformation.md | data/raw/02_milvus_vector_database.md | data/raw/04_reranker_second_stage.md | data/raw/04_reranker_second_stage.md |
| q006 | HyDE 和 Multi-Query 属于 RAG 链路中的哪个阶段？ | data/raw/05_query_transformation.md | data/raw/05_query_transformation.md | data/raw/05_query_transformation.md | data/raw/05_query_transformation.md | data/raw/05_query_transformation.md | data/raw/05_query_transformation.md |
| q007 | 如何证明 RAG 系统升级后真的更好？ | data/raw/06_rag_evaluation_trace.md | data/raw/06_rag_evaluation_trace.md | data/raw/06_rag_evaluation_trace.md | data/raw/06_rag_evaluation_trace.md | data/raw/06_rag_evaluation_trace.md | data/raw/06_rag_evaluation_trace.md |
| q008 | 如果答案错了，retrieval trace 应该帮助我们排查哪些环节？ | data/raw/06_rag_evaluation_trace.md | data/raw/06_rag_evaluation_trace.md | data/raw/06_rag_evaluation_trace.md | data/raw/06_rag_evaluation_trace.md | data/raw/06_rag_evaluation_trace.md | data/raw/06_rag_evaluation_trace.md |

## How To Read This Report

- `bm25_only` tests exact lexical retrieval.
- `dense_only` tests local dense-vector-shaped retrieval.
- `hybrid_rrf` tests sparse + dense fusion.
- `hybrid_rerank` tests second-stage ranking after fusion.
- `transform_hybrid_rerank` tests query transformation before hybrid retrieval.
