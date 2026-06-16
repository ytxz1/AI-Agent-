# 第4天：RAG Part 2 向量化与存储

> 今日主题：Embedding 原理、向量相似度、FAISS 本地索引、Chroma 本地向量库  
> 核心目标：理解文本如何变成向量，能用 FAISS/Chroma 构建本地向量索引，并把它们放回 RAG 检索链路里。  
> 参考资料：
> - FAISS Getting started：<https://github.com/facebookresearch/faiss/wiki/Getting-started>
> - SentenceTransformers：<https://www.sbert.net/>
> - Chroma Getting started：<https://docs.trychroma.com/docs/overview/getting-started>

建议阅读顺序：

1. [01-今日学习计划.md](01-今日学习计划.md)
2. [02-Embedding原理与FAISS-Chroma向量存储详解.md](02-Embedding原理与FAISS-Chroma向量存储详解.md)
3. [03-本地向量索引实战任务.md](03-本地向量索引实战任务.md)
4. [04-代码实现逐行详解.md](04-代码实现逐行详解.md)

代码运行入口：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python demos\01_similarity_basics.py
python demos\02_sbert_semantic_search.py
python demos\03_faiss_l2_search.py
python demos\04_faiss_cosine_search.py
python demos\05_faiss_persistent_store.py
python demos\06_chroma_basic_store.py
python demos\07_chroma_custom_embeddings.py
python demos\08_naive_rag_local_retriever.py
```

今日关键词：

1. Embedding
2. SentenceTransformer
3. Bi-Encoder
4. Cosine Similarity
5. L2 Distance
6. Inner Product
7. Vector Store
8. FAISS
9. Chroma
10. IndexFlatL2
11. Top-k
12. Metadata
13. PersistentClient
14. Retriever
15. Naive RAG
