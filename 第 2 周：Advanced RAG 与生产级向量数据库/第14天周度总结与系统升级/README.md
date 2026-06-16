# 第14天：周度总结与系统升级

> 今日主题：周度总结与系统升级  
> 核心目标：将第一周的基础 RAG 系统升级为生产级 Advanced RAG 雏形，集成 Query Transformation、混合检索、Reranker、Milvus，并用评估体系验证升级是否真的有效。  
> 建议时长：4-8 小时  
> 最终交付物：一套可执行升级计划、一份系统架构设计、一份验收与复盘文档。

---

## 0. 今天要完成什么

今天不是重新写一个 RAG，而是把第一周已有的基础 RAG 系统做一次系统化升级。

第一周的系统大概率是这样的：

```text
文档加载
  -> 文本切分
  -> Embedding
  -> 简单向量库
  -> TopK 检索
  -> 拼接上下文
  -> LLM 生成答案
```

升级后的目标系统应该变成：

```text
文档解析与清洗
  -> 结构化 chunk
  -> Embedding
  -> Milvus 向量库
  -> BM25 稀疏检索
  -> Milvus 稠密检索
  -> Hybrid Fusion
  -> Reranker 精排
  -> Context Builder
  -> LLM 生成答案
  -> 引用、日志、评估、复盘
```

如果再结合第 8 天的 Query Transformation，完整查询链路可以进一步升级为：

```text
用户问题
  -> Query Analysis
  -> Query Rewrite / HyDE / Multi-Query / Decomposition
  -> BM25 + Milvus 双路召回
  -> RRF 融合
  -> Reranker
  -> Top-N Context
  -> LLM Answer with Sources
  -> Retrieval Trace + Evaluation
```

---

## 1. 今日文档导航

建议按这个顺序阅读和执行：

1. [01_周度总结与系统升级详细计划.md](./01_周度总结与系统升级详细计划.md)  
   用来安排今天的学习节奏、复盘本周知识点、明确升级范围和优先级。

2. [02_第一周RAG系统升级技术方案.md](./02_第一周RAG系统升级技术方案.md)  
   这是今天最核心的技术文档，包含架构、模块拆分、Milvus schema、混合检索、Reranker、评估与上线注意事项。

3. [03_实施检查清单与复盘模板.md](./03_实施检查清单与复盘模板.md)  
   用来边做边打勾，最后形成今天的总结和下一周改进计划。

---

## 2. 今日核心学习目标

完成今天之后，你应该能够：

1. 说清楚第一周基础 RAG 的主要短板。
2. 设计一个生产级 RAG 的模块化架构。
3. 将简单向量检索替换为 Milvus 向量数据库。
4. 在 Milvus 检索之外增加 BM25 稀疏检索。
5. 使用 RRF 或加权融合实现 Hybrid Retrieval。
6. 接入本地或 API Reranker，对候选 chunk 进行二阶段排序。
7. 将 Query Transformation 放到检索前，提高复杂问题的召回能力。
8. 设计统一的 retrieval trace 日志，能解释每个答案的来源。
9. 用 Recall@K、MRR、NDCG、Faithfulness 等指标评估升级收益。
10. 形成一份系统升级复盘，明确下一周要继续做什么。

---

## 3. 今日最终交付物

最低限度交付：

- [ ] 一份升级计划。
- [ ] 一张新旧架构对比图。
- [ ] 一个 Milvus collection schema 设计。
- [ ] 一个 Hybrid Retrieval 链路设计。
- [ ] 一个 Reranker 接入方案。
- [ ] 一个 10-20 条 query 的评估集设计。
- [ ] 一份升级验收清单。

进阶交付：

- [ ] 可运行的 `ingest` 脚本。
- [ ] 可运行的 `retrieve` 脚本。
- [ ] 可运行的 `query` 脚本。
- [ ] `retrieval_trace.jsonl` 调试输出。
- [ ] `evaluation_report.md` 对比报告。
- [ ] 支持 original / hybrid / hybrid_rerank / hyde_hybrid_rerank 多策略切换。

---

## 4. 推荐最终项目结构

如果你准备把第一周系统升级成一个更完整的工程，建议采用下面的目录结构：

```text
rag_system_v2/
  README.md
  .env.example
  requirements.txt
  configs/
    app.yaml
    retrieval.yaml
    milvus.yaml
    prompts.yaml
  data/
    raw/
    processed/
    eval/
      eval_queries.jsonl
  storage/
    bm25/
    logs/
  src/
    rag_v2/
      __init__.py
      config.py
      schema.py
      loaders/
      chunkers/
      embeddings/
      vectorstores/
        milvus_store.py
      retrievers/
        bm25_retriever.py
        milvus_retriever.py
        hybrid_retriever.py
      rerankers/
        base.py
        bge_reranker.py
        cohere_reranker.py
      query_transformers/
        base.py
        hyde.py
        multi_query.py
        decomposition.py
      generation/
        context_builder.py
        answer_generator.py
      evaluation/
        retrieval_metrics.py
        answer_metrics.py
      observability/
        trace_logger.py
  scripts/
    ingest.py
    query.py
    evaluate.py
    rebuild_index.py
  outputs/
    retrieval_trace.jsonl
    evaluation_report.md
    weekly_summary.md
```

---

## 5. 一句话总结

今天的重点不是“多接几个工具”，而是把第一周的线性 RAG demo 升级成一个可观测、可评估、可替换模块、可逐步生产化的 RAG 系统。Milvus 负责稳定存储和检索向量，BM25 补足精确词匹配，Reranker 提升 top-k 质量，评估体系告诉你这些升级到底有没有产生真实收益。
