import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


DOCS = [
    "FAISS 是一个用于高效相似度搜索和密集向量聚类的库。",
    "Chroma 是一个面向 AI 应用的开源向量数据库。",
    "SentenceTransformers 可以把句子或段落编码为向量。",
    "FastAPI 是一个用于构建 Python Web API 的框架。",
    "RAG 会先检索相关上下文，再让大模型生成答案。",
    "文本切分会影响向量检索的召回质量。",
]


def main() -> None:
    query = "什么工具适合做本地向量相似度搜索？"
    top_k = 3

    model = SentenceTransformer(MODEL_NAME)

    doc_embeddings = np.asarray(model.encode(DOCS), dtype="float32")
    query_embedding = np.asarray(model.encode([query]), dtype="float32")

    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(doc_embeddings)

    distances, indices = index.search(query_embedding, top_k)

    print("model:", MODEL_NAME)
    print("dimension:", dimension)
    print("index.is_trained:", index.is_trained)
    print("index.ntotal:", index.ntotal)
    print("query:", query)
    print("\nTop results from FAISS IndexFlatL2:")

    for rank, idx in enumerate(indices[0], start=1):
        distance = distances[0][rank - 1]
        print(f"rank={rank}, distance={distance:.4f}, idx={idx}, doc={DOCS[idx]}")


if __name__ == "__main__":
    main()

