from pathlib import Path

import chromadb


BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"


DOCS = [
    "FAISS 是一个用于高效相似度搜索和密集向量聚类的库。",
    "Chroma 是一个面向 AI 应用的开源向量数据库。",
    "SentenceTransformers 可以把句子或段落编码为向量。",
    "FastAPI 是一个用于构建 Python Web API 的框架。",
    "RAG 会先检索相关上下文，再让大模型生成答案。",
    "文本切分会影响向量检索的召回质量。",
]

IDS = [
    "doc-001",
    "doc-002",
    "doc-003",
    "doc-004",
    "doc-005",
    "doc-006",
]

METADATAS = [
    {"topic": "faiss", "source": "day4"},
    {"topic": "chroma", "source": "day4"},
    {"topic": "embedding", "source": "day4"},
    {"topic": "fastapi", "source": "day1"},
    {"topic": "rag", "source": "day4"},
    {"topic": "chunking", "source": "day3"},
]


def main() -> None:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name="rag_notes")

    collection.upsert(
        ids=IDS,
        documents=DOCS,
        metadatas=METADATAS,
    )

    query = "什么是向量数据库？"
    results = collection.query(
        query_texts=[query],
        n_results=3,
    )

    print("Chroma path:", CHROMA_DIR)
    print("collection:", collection.name)
    print("count:", collection.count())
    print("query:", query)
    print("\nTop results from Chroma:")

    for rank, (doc_id, doc, metadata, distance) in enumerate(
        zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ),
        start=1,
    ):
        print(
            f"rank={rank}, id={doc_id}, distance={distance:.4f}, "
            f"topic={metadata['topic']}, source={metadata['source']}, doc={doc}"
        )


if __name__ == "__main__":
    main()

