import json
from pathlib import Path
from typing import Any

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
STORE_DIR = BASE_DIR / "local_faiss_store"
INDEX_PATH = STORE_DIR / "index.faiss"
DOCS_PATH = STORE_DIR / "docs.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


RECORDS = [
    {
        "id": "doc-001",
        "text": "FAISS 是一个用于高效相似度搜索和密集向量聚类的库。",
        "metadata": {"topic": "faiss", "source": "day4"},
    },
    {
        "id": "doc-002",
        "text": "Chroma 是一个面向 AI 应用的开源向量数据库。",
        "metadata": {"topic": "chroma", "source": "day4"},
    },
    {
        "id": "doc-003",
        "text": "SentenceTransformers 可以把句子或段落编码为向量。",
        "metadata": {"topic": "embedding", "source": "day4"},
    },
    {
        "id": "doc-004",
        "text": "RAG 会先检索相关上下文，再让大模型生成答案。",
        "metadata": {"topic": "rag", "source": "day4"},
    },
    {
        "id": "doc-005",
        "text": "metadata 可以帮助 RAG 系统展示来源、做过滤和做权限控制。",
        "metadata": {"topic": "metadata", "source": "day4"},
    },
]


def build_store(records: list[dict[str, Any]]) -> None:
    STORE_DIR.mkdir(exist_ok=True)

    model = SentenceTransformer(MODEL_NAME)
    texts = [record["text"] for record in records]

    embeddings = np.asarray(model.encode(texts), dtype="float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    with DOCS_PATH.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print("FAISS index saved to:", INDEX_PATH)
    print("Document mapping saved to:", DOCS_PATH)


def load_store() -> tuple[faiss.Index, list[dict[str, Any]]]:
    index = faiss.read_index(str(INDEX_PATH))

    with DOCS_PATH.open("r", encoding="utf-8") as f:
        records = json.load(f)

    return index, records


def search(query: str, top_k: int = 3) -> None:
    model = SentenceTransformer(MODEL_NAME)
    index, records = load_store()

    query_embedding = np.asarray(model.encode([query]), dtype="float32")
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    print("\nquery:", query)
    print("Top results from loaded FAISS store:")
    for rank, idx in enumerate(indices[0], start=1):
        record = records[idx]
        print(
            f"rank={rank}, score={scores[0][rank - 1]:.4f}, "
            f"id={record['id']}, topic={record['metadata']['topic']}, "
            f"text={record['text']}"
        )


def main() -> None:
    build_store(RECORDS)
    search("RAG 检索结果为什么要有来源信息？", top_k=3)


if __name__ == "__main__":
    main()

