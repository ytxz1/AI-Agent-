from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


DOCS = [
    "FAISS 是一个用于高效相似度搜索和密集向量聚类的库。",
    "Chroma 是一个面向 AI 应用的开源向量数据库。",
    "SentenceTransformers 可以把句子或段落编码为向量。",
    "RAG 会先检索相关上下文，再让大模型生成答案。",
    "metadata 可以帮助 RAG 系统展示来源、做过滤和做权限控制。",
]

IDS = ["doc-001", "doc-002", "doc-003", "doc-004", "doc-005"]
METADATAS = [
    {"topic": "faiss", "source": "day4"},
    {"topic": "chroma", "source": "day4"},
    {"topic": "embedding", "source": "day4"},
    {"topic": "rag", "source": "day4"},
    {"topic": "metadata", "source": "day4"},
]


def main() -> None:
    model = SentenceTransformer(MODEL_NAME)

    doc_embeddings = model.encode(DOCS).tolist()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name="rag_notes_custom_embeddings")

    collection.upsert(
        ids=IDS,
        documents=DOCS,
        metadatas=METADATAS,
        embeddings=doc_embeddings,
    )

    query = "怎样把文本转成向量？"
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3,
    )

    print("model:", MODEL_NAME)
    print("Chroma path:", CHROMA_DIR)
    print("collection:", collection.name)
    print("count:", collection.count())
    print("query:", query)
    print("\nTop results from Chroma with custom embeddings:")

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

