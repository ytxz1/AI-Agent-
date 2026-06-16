import json
import os
from pathlib import Path

import Stemmer
import nest_asyncio
from dotenv import load_dotenv
from rich import print

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.retrievers.bm25 import BM25Retriever


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"
TRACE_PATH = OUTPUT_DIR / "retrieval_debug.jsonl"


def build_nodes():
    documents = SimpleDirectoryReader(str(DATA_DIR)).load_data()
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=80)
    return splitter.get_nodes_from_documents(documents)


def build_retrievers(nodes, top_k=10):
    index = VectorStoreIndex(nodes)
    vector_retriever = index.as_retriever(similarity_top_k=top_k)

    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=top_k,
        stemmer=Stemmer.Stemmer("english"),
        language="english",
    )

    hybrid_retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=top_k * 2,
        num_queries=1,
        use_async=True,
    )

    return bm25_retriever, vector_retriever, hybrid_retriever


def build_query_engine(hybrid_retriever, top_n=5):
    reranker = CohereRerank(
        top_n=top_n,
        model="rerank-english-v2.0",
        api_key=os.getenv("COHERE_API_KEY"),
    )

    return RetrieverQueryEngine.from_args(
        retriever=hybrid_retriever,
        node_postprocessors=[reranker],
    )


def simplify_nodes(nodes):
    rows = []
    for rank, node_with_score in enumerate(nodes, start=1):
        node = node_with_score.node
        rows.append(
            {
                "rank": rank,
                "score": node_with_score.score,
                "node_id": node.node_id,
                "metadata": node.metadata,
                "text_preview": node.get_content()[:180].replace("\n", " "),
            }
        )
    return rows


def save_trace(record):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with TRACE_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def debug_retrieve(query, bm25_retriever, vector_retriever, hybrid_retriever):
    bm25_nodes = bm25_retriever.retrieve(query)
    dense_nodes = vector_retriever.retrieve(query)
    hybrid_nodes = hybrid_retriever.retrieve(query)

    trace = {
        "query": query,
        "bm25": simplify_nodes(bm25_nodes),
        "dense": simplify_nodes(dense_nodes),
        "hybrid": simplify_nodes(hybrid_nodes),
    }
    save_trace(trace)
    return trace


def main():
    nest_asyncio.apply()
    load_dotenv(ROOT / ".env")

    if not any(DATA_DIR.glob("*")):
        raise RuntimeError(f"请先把语料文件放入 {DATA_DIR}")

    nodes = build_nodes()
    bm25_retriever, vector_retriever, hybrid_retriever = build_retrievers(nodes)
    query_engine = build_query_engine(hybrid_retriever)

    query = "为什么生产级 RAG 需要 BM25 和 Embedding 混合检索，并在最后加入 Rerank？"
    trace = debug_retrieve(query, bm25_retriever, vector_retriever, hybrid_retriever)

    print("[bold cyan]Hybrid retrieval debug trace saved:[/bold cyan]", TRACE_PATH)
    print("[bold]Hybrid top results:[/bold]")
    print(trace["hybrid"][:5])

    response = query_engine.query(query)
    print("\n[bold green]Answer:[/bold green]")
    print(response)

    print("\n[bold]Reranked source nodes:[/bold]")
    for source in response.source_nodes:
        print(
            {
                "score": source.score,
                "node_id": source.node.node_id,
                "metadata": source.node.metadata,
                "preview": source.node.get_content()[:180].replace("\n", " "),
            }
        )


if __name__ == "__main__":
    main()
