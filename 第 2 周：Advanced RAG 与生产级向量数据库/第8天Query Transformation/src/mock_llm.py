from __future__ import annotations

import json


class RuleBasedLLM:
    """A local stand-in for an LLM so the examples run without network access."""

    def complete(self, prompt: str) -> str:
        query = self._extract_query(prompt)

        if "假设性文档" in prompt or "HyDE" in prompt:
            return self._hyde(query)

        if "不同角度" in prompt or "JSON 数组" in prompt and "检索查询" in prompt:
            return json.dumps(self._multi_query(query), ensure_ascii=False)

        if "拆解" in prompt or "子问题" in prompt:
            return json.dumps(self._decompose(query), ensure_ascii=False)

        return query

    def _extract_query(self, prompt: str) -> str:
        markers = ["用户问题：", "用户问题:", "query:"]
        for marker in markers:
            if marker in prompt:
                return prompt.split(marker)[-1].strip()
        return prompt.strip()

    def _hyde(self, query: str) -> str:
        if "HyDE" in query or "hyde" in query.lower():
            return (
                "HyDE 是 RAG 中的查询改写方法，全称 Hypothetical Document Embeddings。"
                "它会根据用户问题生成一段假设性文档，补充专业术语、背景概念和语义上下文，"
                "再将该文本向量化用于检索真实知识库，从而改善短查询和概念查询的召回效果。"
            )

        if "召回" in query or "RAG" in query:
            return (
                "提升 RAG 召回率通常需要从查询改写、向量表示、chunk 切分、top_k 设置、混合检索和 rerank 等方面优化。"
                "Query Transformation 可以把原始问题改写成更适合检索的表达，例如 HyDE 生成假设性文档，"
                "Multi-Query 生成多个角度的查询，以扩大候选文档覆盖范围。"
            )

        if "Milvus" in query or "Qdrant" in query:
            return (
                "Milvus 和 Qdrant 都是常见向量数据库。比较它们时通常关注索引类型、过滤能力、分布式扩展、"
                "Python SDK、元数据过滤、召回率、延迟和生产部署成本。Milvus 支持 IVF、HNSW 等索引，"
                "Qdrant 以 payload filter 和易用的向量检索接口著称。"
            )

        return (
            f"{query} 这个问题可以从 RAG 检索流程、查询改写、语义召回、候选文档融合和最终重排等角度理解。"
            "查询改写的目标是把用户原始表达转换成更适合向量检索和关键词检索的文本。"
        )

    def _multi_query(self, query: str) -> list[str]:
        if "召回" in query or "RAG" in query:
            return [
                "RAG 系统中提高向量检索召回率的方法有哪些？",
                "如何通过 HyDE、Multi-Query 和查询改写改善 RAG 检索效果？",
                "向量数据库召回不足时应该如何调整 chunk、embedding、top_k 和 rerank？",
                "Query Transformation 在提升 RAG 候选文档覆盖率方面有什么作用？",
            ]

        if "HyDE" in query or "hyde" in query.lower():
            return [
                "HyDE 的 Hypothetical Document Embeddings 原理是什么？",
                "为什么生成假设性文档可以改善短查询的向量检索效果？",
                "HyDE 在 RAG 查询改写中有哪些优势和风险？",
                "HyDE 生成的文本应该如何和原始 query 的检索结果融合？",
            ]

        if "Milvus" in query or "Qdrant" in query:
            return [
                "Milvus 和 Qdrant 支持哪些向量索引类型？",
                "Milvus 与 Qdrant 的元数据过滤能力有什么区别？",
                "Milvus 和 Qdrant 在分布式扩展方面如何对比？",
                "Milvus 与 Qdrant 的 Python SDK 和 RAG 生态分别怎么样？",
            ]

        return [
            f"{query} 的核心概念是什么？",
            f"{query} 在 RAG 系统中如何落地？",
            f"{query} 有哪些常见方法和适用场景？",
            f"{query} 如何评估效果和风险？",
        ]

    def _decompose(self, query: str) -> list[str]:
        if "Milvus" in query and "Qdrant" in query:
            return [
                "Milvus 支持哪些向量索引类型？",
                "Qdrant 支持哪些向量索引和检索能力？",
                "Milvus 的 metadata filter 能力如何？",
                "Qdrant 的 payload filter 能力如何？",
                "Milvus 和 Qdrant 在分布式扩展方面有什么差异？",
                "Milvus 和 Qdrant 的 Python SDK 生态分别如何？",
            ]

        if "区别" in query or "对比" in query or "分别" in query:
            return [
                f"{query} 中第一个核心对象的特点是什么？",
                f"{query} 中第二个核心对象的特点是什么？",
                f"{query} 涉及哪些关键比较维度？",
            ]

        return [query]

