# 第3天：RAG Part 1 加载与分割

今天的学习主题是：RAG Part 1: 加载与分割。

本目录围绕 RAG 数据准备阶段展开：如何把 PDF、Markdown 等不同格式文档加载成统一的 `Document`，如何保留对检索有价值的元数据，如何根据文档结构、token 限制和问答场景选择合适的文本分块策略。

建议阅读顺序：

1. [01-今日学习计划.md](01-今日学习计划.md)
2. [02-RAG加载与文本分块详解.md](02-RAG加载与文本分块详解.md)
3. [03-代码实现详解.md](03-代码实现详解.md)

代码入口：

1. [scripts/rag_preprocess.py](scripts/rag_preprocess.py)
2. [examples/sample.md](examples/sample.md)
3. [requirements.txt](requirements.txt)

参考资料：

1. [opendatalab/MinerU](https://github.com/opendatalab/MinerU)
2. [docling-project/docling](https://github.com/docling-project/docling)

核心关键词：

1. RAG 数据摄取
2. PDF 解析
3. Markdown 加载
4. OCR
5. Layout Analysis
6. Document
7. Metadata
8. Chunk
9. Chunk Size
10. Chunk Overlap
11. Recursive Splitter
12. Hierarchical Chunking
13. Parent-Child Chunking
14. RAG-Ready Markdown
