# Milvus 生产级向量数据库

Milvus 是生产级向量数据库，适合在 RAG 系统中存储 chunk embedding，并提供大规模向量相似度搜索能力。

RAG 中常见的 Milvus collection schema 包含 id、chunk_id、doc_id、vector、text、source、page_start、page_end、section_path、tenant_id、corpus_version 和 metadata 等字段。

Milvus 解决的是向量持久化、向量索引、TopK 检索、metadata filter、collection 管理和后续扩展问题。它不能单独解决 chunk 质量、query 改写、rerank 排序、答案忠实性和评估问题。

查询链路中，用户问题会先经过 embedding 模型生成 query vector，然后调用 Milvus search，在指定 collection 中搜索相似 chunk。生产环境中通常会使用 tenant_id、doc_id、category、corpus_version 等 metadata filter 控制检索范围。

常见参数包括 metric_type、index_type、limit、output_fields 和 filter。文本类 RAG 常用 COSINE 或 IP 作为相似度度量，具体选择应和 embedding 模型保持一致。

