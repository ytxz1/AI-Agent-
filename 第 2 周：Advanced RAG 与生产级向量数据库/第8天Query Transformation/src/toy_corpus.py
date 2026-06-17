from __future__ import annotations

from .models import Document


def load_toy_documents() -> list[Document]:
    return [
        Document(
            doc_id="doc_query_transform",
            title="Query Transformation 概览",
            text=(
                "Query Transformation 是 RAG 检索前的查询改写层。它会把用户原始问题改写成更适合检索的表达，"
                "常见策略包括 query rewrite、HyDE、Multi-Query、Query Decomposition 和 Step-back Query。"
                "它的目标是提升召回率，让 retriever 更容易命中相关文档。"
            ),
            metadata={"topic": "query_transformation"},
        ),
        Document(
            doc_id="doc_hyde",
            title="HyDE 假设性文档检索",
            text=(
                "HyDE 的全称是 Hypothetical Document Embeddings。它先让大语言模型根据用户问题生成一段假设性答案或假设性文档，"
                "再对这段文本进行 embedding，并用该向量去检索真实知识库。HyDE 适合短查询、专业概念查询和语义信息不足的问题，"
                "但假设性文档不能直接当作事实证据。"
            ),
            metadata={"topic": "hyde"},
        ),
        Document(
            doc_id="doc_multi_query",
            title="Multi-Query 多查询扩展",
            text=(
                "Multi-Query 会从多个角度重写用户问题，生成多个检索查询。每个查询分别进入 retriever，得到候选文档后再进行去重、"
                "RRF 融合或 rerank。它适合宽泛问题、表达方式不统一的知识库，以及需要扩大召回覆盖面的场景。"
            ),
            metadata={"topic": "multi_query"},
        ),
        Document(
            doc_id="doc_decomposition",
            title="Query Decomposition 查询拆解",
            text=(
                "Query Decomposition 会把复杂问题拆成多个可以独立检索的子问题。它适合比较问题、多跳问题、包含多个维度的问题。"
                "例如比较 Milvus 和 Qdrant 时，可以拆成索引类型、过滤能力、分布式扩展、Python SDK 等子问题。"
            ),
            metadata={"topic": "decomposition"},
        ),
        Document(
            doc_id="doc_rrf",
            title="RRF 多路检索结果融合",
            text=(
                "Reciprocal Rank Fusion 简称 RRF，是一种常见的多路检索结果融合方法。它不依赖不同检索器的原始分数是否可比，"
                "而是根据文档在各路结果中的排名计算融合分数。文档被多个查询命中时，通常会获得更高的最终排序。"
            ),
            metadata={"topic": "fusion"},
        ),
        Document(
            doc_id="doc_rerank",
            title="Rerank 在 RAG 中的作用",
            text=(
                "Rerank 发生在初步检索之后，用更强的相关性模型对候选文档重新排序。Query Transformation 负责扩大召回，"
                "Rerank 负责压低噪声并把最相关的文档排在前面。生产级 RAG 经常把 Multi-Query、Hybrid Search 和 Rerank 组合使用。"
            ),
            metadata={"topic": "rerank"},
        ),
        Document(
            doc_id="doc_milvus_index",
            title="Milvus 索引类型",
            text=(
                "Milvus 支持多种向量索引，例如 IVF_FLAT、IVF_SQ8、HNSW、DiskANN 等。IVF 类索引通过聚类和倒排列表提升检索速度，"
                "HNSW 基于图结构，通常召回率高、查询速度快，但内存占用较高。索引选择需要权衡召回率、延迟、内存和构建成本。"
            ),
            metadata={"topic": "milvus"},
        ),
        Document(
            doc_id="doc_qdrant_filter",
            title="Qdrant Payload Filter",
            text=(
                "Qdrant 使用 payload 存储结构化元数据，并支持丰富的过滤条件。向量检索可以和 payload filter 结合，"
                "用于租户隔离、时间范围过滤、标签过滤和业务字段约束。过滤能力对生产级 RAG 的权限控制和精确召回非常重要。"
            ),
            metadata={"topic": "qdrant"},
        ),
    ]

