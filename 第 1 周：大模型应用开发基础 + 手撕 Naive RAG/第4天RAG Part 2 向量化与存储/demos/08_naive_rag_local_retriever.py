from dataclasses import dataclass
from typing import Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


DOCS = [
    {
        "text": "Embedding 是把文本映射成固定维度向量的表示方法，语义相近的文本在向量空间中距离更近。",
        "metadata": {"source": "day4", "topic": "embedding"},
    },
    {
        "text": "FAISS 是 Facebook AI Research 开源的相似度搜索库，常用于密集向量检索。",
        "metadata": {"source": "day4", "topic": "faiss"},
    },
    {
        "text": "Chroma 是一个面向 AI 应用的开源向量数据库，可以管理 documents、metadata、ids 和本地持久化。",
        "metadata": {"source": "day4", "topic": "chroma"},
    },
    {
        "text": "RAG 会先根据用户问题检索相关上下文，再把上下文和问题一起放入 Prompt，让大模型生成答案。",
        "metadata": {"source": "day4", "topic": "rag"},
    },
    {
        "text": "chunk 太大会导致一个向量混合多个主题，chunk 太小会导致语义不完整。",
        "metadata": {"source": "day3", "topic": "chunking"},
    },
    {
        "text": "metadata 可以用于来源引用、过滤、权限控制、调试检索质量和结果展示。",
        "metadata": {"source": "day4", "topic": "metadata"},
    },
]


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict[str, Any]


class LocalFaissRetriever:
    def __init__(
        self,
        records: list[dict[str, Any]],
        model_name: str = MODEL_NAME,
    ) -> None:
        self.records = records
        self.model = SentenceTransformer(model_name)

        texts = [record["text"] for record in records]
        embeddings = np.asarray(self.model.encode(texts), dtype="float32")
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        query_embedding = np.asarray(self.model.encode([query]), dtype="float32")
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            record = self.records[idx]
            results.append(
                SearchResult(
                    text=record["text"],
                    score=float(score),
                    metadata=record["metadata"],
                )
            )

        return results


def format_context(results: list[SearchResult]) -> str:
    blocks = []
    for idx, result in enumerate(results, start=1):
        source = result.metadata.get("source", "unknown")
        topic = result.metadata.get("topic", "unknown")
        blocks.append(
            f"[{idx}] source={source}, topic={topic}, score={result.score:.4f}\n"
            f"{result.text}"
        )
    return "\n\n".join(blocks)


def build_prompt(question: str, context: str) -> str:
    return f"""你是一名严谨的 RAG 学习助手。
请只根据给定上下文回答问题。
如果上下文中没有答案，请回答：根据当前上下文无法确定。
回答时请给出使用到的来源编号。

上下文：
{context}

问题：
{question}

答案："""


def main() -> None:
    question = "为什么 RAG 需要 Embedding 和向量索引？"
    retriever = LocalFaissRetriever(DOCS)

    results = retriever.search(question, top_k=4)
    context = format_context(results)
    prompt = build_prompt(question, context)

    print("question:")
    print(question)

    print("\nretrieved context:")
    print(context)

    print("\nfinal prompt for LLM:")
    print(prompt)


if __name__ == "__main__":
    main()

