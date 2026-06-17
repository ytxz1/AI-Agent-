"""
Day 12 Milvus SDK demo.

This script demonstrates a realistic RAG-style Milvus workflow:

1. Connect to a Docker Standalone Milvus instance.
2. Recreate a collection with scalar fields and one vector field.
3. Insert sample document chunks.
4. Load the collection.
5. Run vector search.
6. Run vector search with metadata filters.
7. Run scalar query, get, upsert, and delete.

The embedding function is intentionally local and deterministic. It is not a
real semantic embedding model, but it lets you practice Milvus SDK operations
without downloading models or calling external APIs.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any

from pymilvus import DataType, MilvusClient


DEFAULT_URI = "http://localhost:19530"
DEFAULT_TOKEN = "root:Milvus"
DEFAULT_COLLECTION = "day12_rag_chunks"
VECTOR_DIM = 128


@dataclass(frozen=True)
class DocumentChunk:
    id: int
    doc_id: str
    chunk_id: str
    title: str
    text: str
    source: str
    category: str
    tenant_id: str
    page: int
    created_at: int


def tokenize(text: str) -> list[str]:
    """Tokenize English words, numbers, and individual CJK characters."""
    text = text.lower()
    return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text)


def stable_hash(value: str) -> int:
    """Return a stable integer hash across Python processes."""
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def embed_text(text: str, dim: int = VECTOR_DIM) -> list[float]:
    """
    Convert text to a deterministic dense vector.

    This is a hashing-vectorizer style toy embedding:
    - Same text always gets the same vector.
    - Similar texts may share some tokens and therefore some dimensions.
    - It is useful for SDK practice, not for production semantic search.
    """
    vector = [0.0] * dim
    tokens = tokenize(text)

    if not tokens:
        return vector

    for token in tokens:
        hashed = stable_hash(token)
        index = hashed % dim
        sign = 1.0 if ((hashed >> 8) & 1) == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector

    return [value / norm for value in vector]


def now_unix_seconds() -> int:
    return int(time.time())


def build_sample_chunks() -> list[DocumentChunk]:
    ts = now_unix_seconds()
    return [
        DocumentChunk(
            id=1,
            doc_id="milvus-install",
            chunk_id="milvus-install-001",
            title="Milvus Docker 部署",
            text="Milvus Standalone 可以通过 Docker 启动，默认服务端口是 19530，WebUI 端口是 9091。",
            source="milvus_official_docs",
            category="milvus",
            tenant_id="team_a",
            page=1,
            created_at=ts,
        ),
        DocumentChunk(
            id=2,
            doc_id="milvus-sdk",
            chunk_id="milvus-sdk-001",
            title="Milvus Python SDK",
            text="PyMilvus 提供 MilvusClient，用于创建 collection、插入向量、构建索引和执行 search。",
            source="milvus_python_notes",
            category="milvus",
            tenant_id="team_a",
            page=2,
            created_at=ts,
        ),
        DocumentChunk(
            id=3,
            doc_id="rag-pipeline",
            chunk_id="rag-pipeline-001",
            title="RAG 检索链路",
            text="RAG 系统一般包括文档切分、embedding、向量召回、metadata filter、rerank 和 LLM 生成。",
            source="rag_architecture_notes",
            category="rag",
            tenant_id="team_a",
            page=3,
            created_at=ts,
        ),
        DocumentChunk(
            id=4,
            doc_id="qdrant-intro",
            chunk_id="qdrant-intro-001",
            title="Qdrant 基础概念",
            text="Qdrant 使用 collection、point、vector 和 payload 组织向量数据，过滤能力直观。",
            source="qdrant_docs",
            category="qdrant",
            tenant_id="team_b",
            page=1,
            created_at=ts,
        ),
        DocumentChunk(
            id=5,
            doc_id="infinity-intro",
            chunk_id="infinity-intro-001",
            title="Infinity 混合检索",
            text="Infinity 强调面向 AI 应用的混合搜索，支持稠密向量、稀疏向量、全文检索和 tensor 检索。",
            source="infinity_github",
            category="infinity",
            tenant_id="team_b",
            page=1,
            created_at=ts,
        ),
        DocumentChunk(
            id=6,
            doc_id="milvus-production",
            chunk_id="milvus-production-001",
            title="Milvus 生产化关注点",
            text="生产级 Milvus 需要关注索引类型、召回率、查询延迟、备份恢复、监控告警和资源隔离。",
            source="production_checklist",
            category="milvus",
            tenant_id="team_a",
            page=4,
            created_at=ts,
        ),
        DocumentChunk(
            id=7,
            doc_id="metadata-filter",
            chunk_id="metadata-filter-001",
            title="Metadata Filter",
            text="向量搜索结合 tenant_id、category、doc_id 等标量字段过滤，可以实现多租户和权限范围控制。",
            source="rag_security_notes",
            category="rag",
            tenant_id="team_a",
            page=5,
            created_at=ts,
        ),
    ]


def chunk_to_entity(chunk: DocumentChunk) -> dict[str, Any]:
    return {
        "id": chunk.id,
        "doc_id": chunk.doc_id,
        "chunk_id": chunk.chunk_id,
        "title": chunk.title,
        "text": chunk.text,
        "source": chunk.source,
        "category": chunk.category,
        "tenant_id": chunk.tenant_id,
        "page": chunk.page,
        "created_at": chunk.created_at,
        "vector": embed_text(chunk.title + "\n" + chunk.text),
    }


def connect(uri: str, token: str) -> MilvusClient:
    return MilvusClient(uri=uri, token=token)


def assert_milvus_available(client: MilvusClient) -> None:
    try:
        client.list_collections()
    except Exception as exc:
        raise RuntimeError(
            "Cannot connect to Milvus. Please start Milvus Docker first and "
            "check that http://localhost:19530 is reachable."
        ) from exc


def recreate_collection(client: MilvusClient, collection_name: str) -> None:
    if client.has_collection(collection_name):
        print(f"[drop] collection already exists: {collection_name}")
        client.drop_collection(collection_name)

    print(f"[create] collection: {collection_name}")
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
    )

    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
    schema.add_field("doc_id", DataType.VARCHAR, max_length=128)
    schema.add_field("chunk_id", DataType.VARCHAR, max_length=128)
    schema.add_field("title", DataType.VARCHAR, max_length=512)
    schema.add_field("text", DataType.VARCHAR, max_length=4096)
    schema.add_field("source", DataType.VARCHAR, max_length=512)
    schema.add_field("category", DataType.VARCHAR, max_length=128)
    schema.add_field("tenant_id", DataType.VARCHAR, max_length=128)
    schema.add_field("page", DataType.INT64)
    schema.add_field("created_at", DataType.INT64)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )


def insert_chunks(client: MilvusClient, collection_name: str) -> None:
    chunks = build_sample_chunks()
    entities = [chunk_to_entity(chunk) for chunk in chunks]
    print(f"[insert] rows: {len(entities)}")

    result = client.insert(
        collection_name=collection_name,
        data=entities,
    )
    print("[insert result]", result)

    client.flush(collection_name=collection_name)


def load_collection(client: MilvusClient, collection_name: str) -> None:
    print(f"[load] collection: {collection_name}")
    client.load_collection(collection_name=collection_name)
    print("[load state]", client.get_load_state(collection_name=collection_name))


def print_hits(title: str, results: list[Any]) -> None:
    print(f"\n=== {title} ===")
    for query_index, hits in enumerate(results, start=1):
        print(f"query #{query_index}")
        if not hits:
            print("  no hits")
            continue

        for rank, hit in enumerate(hits, start=1):
            entity = hit.get("entity", {})
            print(
                "  "
                f"rank={rank} "
                f"id={hit.get('id')} "
                f"score={hit.get('distance'):.4f} "
                f"tenant={entity.get('tenant_id')} "
                f"category={entity.get('category')} "
                f"title={entity.get('title')} "
                f"source={entity.get('source')}"
            )
            print(f"    text={entity.get('text')}")


def vector_search(
    client: MilvusClient,
    collection_name: str,
    query: str,
    limit: int = 3,
    filter_expr: str | None = None,
) -> list[Any]:
    query_vector = embed_text(query)

    return client.search(
        collection_name=collection_name,
        anns_field="vector",
        data=[query_vector],
        limit=limit,
        filter=filter_expr,
        output_fields=[
            "doc_id",
            "chunk_id",
            "title",
            "text",
            "source",
            "category",
            "tenant_id",
            "page",
        ],
        search_params={"metric_type": "COSINE"},
    )


def demo_basic_search(client: MilvusClient, collection_name: str) -> None:
    query = "Milvus Docker 怎么部署，端口是多少？"
    results = vector_search(client, collection_name, query, limit=3)
    print_hits(f"basic vector search: {query}", results)


def demo_filtered_search(client: MilvusClient, collection_name: str) -> None:
    query = "向量数据库如何做多租户权限过滤？"
    filter_expr = 'tenant_id == "team_a" and category == "rag"'
    results = vector_search(
        client,
        collection_name,
        query,
        limit=5,
        filter_expr=filter_expr,
    )
    print_hits(f"filtered search: {query} | {filter_expr}", results)


def demo_scalar_query(client: MilvusClient, collection_name: str) -> None:
    print("\n=== scalar query ===")
    rows = client.query(
        collection_name=collection_name,
        filter='category == "milvus"',
        output_fields=["id", "title", "source", "category", "tenant_id", "page"],
    )
    for row in rows:
        print(row)


def demo_get_by_ids(client: MilvusClient, collection_name: str) -> None:
    print("\n=== get by primary keys ===")
    rows = client.get(
        collection_name=collection_name,
        ids=[1, 3],
        output_fields=["id", "title", "text", "source"],
    )
    for row in rows:
        print(row)


def demo_upsert(client: MilvusClient, collection_name: str) -> None:
    print("\n=== upsert one row ===")
    updated = DocumentChunk(
        id=6,
        doc_id="milvus-production",
        chunk_id="milvus-production-001",
        title="Milvus 生产化关注点（更新版）",
        text="线上 Milvus 需要持续观察 P95 延迟、索引构建状态、内存水位、磁盘空间、备份任务和恢复演练。",
        source="production_checklist_v2",
        category="milvus",
        tenant_id="team_a",
        page=6,
        created_at=now_unix_seconds(),
    )
    result = client.upsert(
        collection_name=collection_name,
        data=[chunk_to_entity(updated)],
    )
    print(result)

    rows = client.get(
        collection_name=collection_name,
        ids=[6],
        output_fields=["id", "title", "text", "source", "page"],
    )
    print("[after upsert]", rows)


def demo_delete(client: MilvusClient, collection_name: str) -> None:
    print("\n=== delete one row ===")
    result = client.delete(
        collection_name=collection_name,
        filter='doc_id == "infinity-intro"',
    )
    print(result)

    rows = client.query(
        collection_name=collection_name,
        filter='doc_id == "infinity-intro"',
        output_fields=["id", "title", "doc_id"],
    )
    print("[after delete query]", rows)


def demo_retrieve_function(client: MilvusClient, collection_name: str) -> None:
    print("\n=== RAG-style retrieve() function ===")

    def retrieve(
        query: str,
        top_k: int = 5,
        tenant_id: str | None = None,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        filters = []
        if tenant_id:
            filters.append(f'tenant_id == "{tenant_id}"')
        if category:
            filters.append(f'category == "{category}"')

        filter_expr = " and ".join(filters) if filters else None
        results = vector_search(
            client=client,
            collection_name=collection_name,
            query=query,
            limit=top_k,
            filter_expr=filter_expr,
        )

        documents: list[dict[str, Any]] = []
        for hits in results:
            for hit in hits:
                entity = hit.get("entity", {})
                documents.append(
                    {
                        "id": hit.get("id"),
                        "score": hit.get("distance"),
                        "title": entity.get("title"),
                        "text": entity.get("text"),
                        "source": entity.get("source"),
                        "category": entity.get("category"),
                        "tenant_id": entity.get("tenant_id"),
                    }
                )
        return documents

    docs = retrieve(
        query="Milvus Python SDK 怎么搜索和过滤？",
        top_k=3,
        tenant_id="team_a",
    )
    for doc in docs:
        print(doc)


def cleanup_collection(client: MilvusClient, collection_name: str) -> None:
    if client.has_collection(collection_name):
        print(f"[cleanup] drop collection: {collection_name}")
        client.drop_collection(collection_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Milvus Day 12 SDK demo for Docker Standalone."
    )
    parser.add_argument(
        "--uri",
        default=os.getenv("MILVUS_URI", DEFAULT_URI),
        help=f"Milvus URI. Default: {DEFAULT_URI}",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("MILVUS_TOKEN", DEFAULT_TOKEN),
        help=f"Milvus token. Default: {DEFAULT_TOKEN}",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("MILVUS_COLLECTION", DEFAULT_COLLECTION),
        help=f"Collection name. Default: {DEFAULT_COLLECTION}",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Drop the demo collection after the script finishes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = connect(uri=args.uri, token=args.token)

    try:
        assert_milvus_available(client)
        print("[connect] ok")
        print("[collections before]", client.list_collections())

        recreate_collection(client, args.collection)
        insert_chunks(client, args.collection)
        load_collection(client, args.collection)

        demo_basic_search(client, args.collection)
        demo_filtered_search(client, args.collection)
        demo_scalar_query(client, args.collection)
        demo_get_by_ids(client, args.collection)
        demo_upsert(client, args.collection)
        demo_delete(client, args.collection)
        demo_retrieve_function(client, args.collection)

        print("\n[collections after]", client.list_collections())

        if args.cleanup:
            cleanup_collection(client, args.collection)

        return 0
    except Exception as exc:
        print(f"\n[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

