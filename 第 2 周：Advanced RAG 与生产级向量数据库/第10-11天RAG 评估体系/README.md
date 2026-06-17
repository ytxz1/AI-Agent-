# 第10-11天：RAG 评估体系

本目录用于学习 RAG 评估体系：先理解 RAG 的核心评估指标，再使用 RAGAs 对优化前后的系统性能做可重复、可解释、可对比的评估。

## 学习目标

1. 建立 RAG 评估的完整心智模型：检索质量、上下文质量、答案质量、端到端业务质量、成本与延迟。
2. 掌握 RAG 常用核心指标：Context Precision、Context Recall、Context Relevancy、Faithfulness、Answer Relevancy、Answer Correctness、Noise Sensitivity、Hit Rate、MRR、nDCG。
3. 能够用 RAGAs 构造评估数据集、运行 baseline 评估、分析失败样本、做针对性优化、再次评估并输出对比报告。
4. 理解 RAGAs、FlashRAG、DeepEval、LightEval 的定位差异，知道在工程项目、研究复现、LLM 基准测试中分别怎么选。

## 推荐阅读顺序

1. [01_两天学习计划.md](./01_两天学习计划.md)
2. [02_RAG核心评估指标详解.md](./02_RAG核心评估指标详解.md)
3. [03_RAGAs优化前后评估实战.md](./03_RAGAs优化前后评估实战.md)
4. [04_评估数据集与报告模板.md](./04_评估数据集与报告模板.md)
5. [05_代码实战详解：用RAGAs评估优化前后的RAG系统.md](./05_代码实战详解：用RAGAs评估优化前后的RAG系统.md)

## 今日产出物

- 一份 RAG 评估指标速查表。
- 一个可运行的 RAGAs 评估脚本方案。
- 一份 baseline vs optimized 的实验记录模板。
- 一套失败样本归因方法：判断问题来自检索、重排、上下文压缩、生成提示词、模型能力，还是评估集本身。

## 参考资料

- RAGAs 官方文档：https://docs.ragas.io/en/latest/index.html
- RAGAs Evaluate and Improve a RAG App：https://docs.ragas.io/en/latest/howtos/applications/evaluate-and-improve-rag/
- FlashRAG：https://github.com/RUC-NLPIR/FlashRAG
- DeepEval：https://github.com/confident-ai/deepeval
- LightEval：https://github.com/huggingface/lighteval
