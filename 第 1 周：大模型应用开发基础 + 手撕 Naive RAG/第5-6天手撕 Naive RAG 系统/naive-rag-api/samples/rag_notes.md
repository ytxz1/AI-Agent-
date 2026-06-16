# RAG 学习笔记

RAG 是 Retrieval Augmented Generation 的缩写，中文通常叫检索增强生成。

一个基础 RAG 系统通常包含两个阶段：索引阶段和查询阶段。

索引阶段包括文档加载、文本切分、Embedding 编码和向量存储。

查询阶段包括用户问题向量化、相似度检索、上下文组装和大模型生成。

chunk size 会影响检索效果。chunk 太小可能导致上下文不完整，chunk 太大可能导致语义被稀释。

metadata 用于记录文档来源、页码、文件名、chunk 编号等信息，方便答案溯源。

