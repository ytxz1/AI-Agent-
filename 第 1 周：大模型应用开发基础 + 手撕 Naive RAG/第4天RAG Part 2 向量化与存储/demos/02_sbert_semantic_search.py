from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


DOCS = [
    "FAISS 是一个用于高效相似度搜索和密集向量聚类的库。",
    "Chroma 是一个面向 AI 应用的开源向量数据库。",
    "SentenceTransformers 可以把句子或段落编码为向量。",
    "FastAPI 是一个用于构建 Python Web API 的框架。",
    "RAG 会先检索相关上下文，再让大模型生成答案。",
    "文本切分会影响向量检索的召回质量。",
    "余弦相似度可以衡量两个向量方向是否接近。",
]


def main() -> None:
    query = "如何把文本转成向量并检索相似内容？"

    model = SentenceTransformer(MODEL_NAME)

    doc_embeddings = model.encode(DOCS)
    query_embedding = model.encode(query)

    print("model:", MODEL_NAME)
    print("doc_embeddings shape:", doc_embeddings.shape)
    print("query_embedding shape:", query_embedding.shape)

    scores = cos_sim(query_embedding, doc_embeddings)[0]
    ranking = sorted(
        enumerate(scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    print("\nquery:", query)
    print("\nTop results:")
    for rank, (idx, score) in enumerate(ranking, start=1):
        print(f"rank={rank}, score={float(score):.4f}, doc={DOCS[idx]}")


if __name__ == "__main__":
    main()

