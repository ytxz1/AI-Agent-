# 第 12 天：生产级向量数据库 Milvus

> 学习主题：使用 Docker 部署 Milvus，并掌握 Milvus Python SDK 的核心开发流程。  
> 学习目标：今天结束后，你应该能独立启动 Milvus Standalone、本地连接服务、创建集合、写入向量、构建索引、执行向量搜索与过滤搜索，并能从生产视角理解 Milvus、Infinity、Qdrant 的差异。  
> 推荐学习时长：6 到 8 小时。  
> 当前参考日期：2026-06-15。

---

## 0. 今日学习路线总览

今天的重点不是“知道 Milvus 是向量数据库”，而是把 Milvus 当成一个可部署、可连接、可调试、可接入 RAG 系统的基础设施组件来学习。

你今天要完成四件事：

1. 用 Docker 启动 Milvus Standalone。
2. 用 Python SDK 完成集合、数据、索引、搜索、过滤、删除等核心操作。
3. 理解生产级向量数据库的关键概念：schema、index、metric、consistency、segment、load、flush、backup、monitoring。
4. 横向比较 Milvus、Infinity、Qdrant，知道不同项目为什么会选不同的向量数据库。

建议产出：

- 一个可运行的 Milvus Docker 实例。
- 一个 `milvus_day12_demo.py` 实验脚本。
- 一份自己的学习笔记，记录部署问题、SDK API、索引参数、搜索结果和踩坑。
- 一张“Milvus 接入 RAG 系统”的架构图。

---

## 1. 参考资料

### 1.1 官方资料

- Milvus Docker Standalone 安装文档：<https://milvus.io/docs/zh/install_standalone-docker.md>
- Milvus Quickstart：<https://milvus.io/docs/quickstart.md>
- Milvus 基本向量搜索：<https://milvus.io/docs/zh/single-vector-search.md>
- Milvus 插入、更新、删除：<https://milvus.io/docs/zh/insert-update-delete.md>
- Milvus Collection 管理：<https://milvus.io/docs/zh/manage-collections.md>
- Milvus 向量索引：<https://milvus.io/docs/zh/index-vector-fields.md>
- PyMilvus API：<https://milvus.io/api-reference/pymilvus/>
- Infinity GitHub：<https://github.com/infiniflow/infinity>
- Qdrant 文档：<https://qdrant.tech/documentation/>

### 1.2 今日重点参考结论

Milvus 官方 Docker Standalone 文档当前推荐通过 `standalone_embed.sh` 脚本启动 Docker 容器。脚本启动后，默认容器名为 `milvus`，服务端口为 `19530`，WebUI 端口为 `9091`。

Milvus Python 快速上手当前推荐使用 `pymilvus` 中的 `MilvusClient`。它可以连接 Docker 部署的 Milvus，也可以使用 Milvus Lite 做本地嵌入式实验。今天以 Docker Standalone 为主。

Infinity 更强调 AI-native database、混合检索、稠密向量、稀疏向量、全文检索、tensor/multi-vector 检索和易部署单二进制架构。

Qdrant 是 Rust 实现的向量数据库，核心抽象是 collection、point、vector、payload，使用体验简洁，过滤能力强，也适合 RAG 和语义搜索服务。

---

## 2. 学习前置知识

你需要提前理解这些概念。如果不熟，先用 30 分钟补齐。

### 2.1 向量数据库为什么存在

普通数据库擅长：

- 主键查询。
- 条件过滤。
- 排序分页。
- 聚合统计。
- 事务一致性。

向量数据库擅长：

- 存储高维 embedding。
- 根据向量相似度查找 TopK。
- 支持语义搜索。
- 支持 metadata 过滤。
- 支持海量向量索引。
- 在召回率、延迟、吞吐量、成本之间做工程权衡。

在 RAG 中，向量数据库通常位于这个位置：

```text
文档 -> 切分 Chunk -> Embedding 模型 -> 向量数据库
                                      |
用户问题 -> Query Embedding ----------|
                                      v
                                TopK 召回
                                      v
                                Rerank / Filter
                                      v
                                  LLM 生成
```

### 2.2 向量数据库核心术语

| 术语 | 含义 | 在 Milvus 中的对应 |
| --- | --- | --- |
| Database | 数据库命名空间 | database |
| Collection | 向量数据集合，类似关系数据库表 | collection |
| Field | 字段 | id、vector、text、source、created_at |
| Primary Key | 主键 | 通常是 int64 或 varchar |
| Vector Field | 向量字段 | FLOAT_VECTOR、SPARSE_FLOAT_VECTOR 等 |
| Scalar Field | 标量字段 | varchar、int、bool、json 等 |
| Entity | 一条记录 | 一条文档 chunk |
| Index | 向量索引 | AUTOINDEX、HNSW、IVF_FLAT 等 |
| Metric | 相似度度量 | COSINE、IP、L2 |
| TopK | 返回最相似的 K 条 | search limit |
| Filter | 元数据过滤 | expr |
| Load | 将集合加载到可搜索状态 | load_collection |

### 2.3 常见相似度度量

| 度量 | 含义 | 相似度判断 |
| --- | --- | --- |
| COSINE | 余弦相似度 | 值越大越相似 |
| IP | Inner Product，内积 | 值越大越相似 |
| L2 | 欧氏距离 | 值越小越相似 |

RAG 文本 embedding 常见选择：

- 如果 embedding 已归一化，`COSINE` 和 `IP` 经常都可用。
- 如果使用 OpenAI、BGE、E5、GTE 等文本 embedding，优先从模型文档确认推荐的相似度。
- 生产中不要随手换 metric，因为索引、召回结果、阈值策略都会受影响。

---

## 3. 今日详细学习计划

### 3.1 上午：部署与基础连接

目标：把 Milvus 跑起来，并能用 Python 连通。

#### 任务 1：检查本机环境

```powershell
docker --version
docker compose version
python --version
pip --version
```

你需要确认：

- Docker 可以正常启动。
- Docker Desktop 已运行。
- Python 建议 3.10 或 3.11。
- 当前终端可以执行 `pip install`。

#### 任务 2：创建 Day 12 实验目录

建议目录结构：

```text
第12天生产级向量数据库 (Milvus)/
  README.md
  scripts/
    milvus_day12_demo.py
    milvus_reset_collection.py
  notes/
    troubleshooting.md
  data/
    sample_docs.jsonl
```

今天可以先只写一个 demo 脚本，后续再逐步拆分。

#### 任务 3：启动 Milvus Standalone

官方 Docker Standalone 方式：

```bash
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start
```

Windows 用户建议：

- 优先在 WSL2 Ubuntu 中执行上面的命令。
- 或参考 Milvus 官方 Docker Desktop Windows 文档。
- 如果你在 PowerShell 中使用 Git Bash，也可以通过 Git Bash 执行 `bash standalone_embed.sh start`。

启动后确认：

```bash
docker ps
```

你应该看到类似：

```text
CONTAINER ID   IMAGE             COMMAND   PORTS                      NAMES
...            milvusdb/milvus   ...       0.0.0.0:19530->19530/tcp   milvus
```

Milvus 服务地址：

```text
http://localhost:19530
```

Milvus WebUI：

```text
http://127.0.0.1:9091/webui/
```

#### 任务 4：安装 Python SDK

```bash
pip install -U pymilvus
```

如果你要使用 `pymilvus` 自带的 embedding 模型工具：

```bash
pip install -U "pymilvus[model]"
```

注意：

- `pymilvus[model]` 可能会安装或依赖 PyTorch 等包，耗时较长。
- 如果网络无法访问 Hugging Face，可以先用随机向量完成 SDK 流程。
- 生产项目中一般会显式选择 embedding 模型，而不是依赖 quickstart 的默认模型。

#### 任务 5：连接 Milvus

最小连接代码：

```python
from pymilvus import MilvusClient

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus",
)

print(client.list_collections())
```

如果连接失败，优先检查：

- Docker 容器是否运行。
- `19530` 端口是否暴露。
- 防火墙是否阻止本机访问。
- Docker Desktop/WSL2 网络是否正常。
- `uri` 是否写成了 `http://localhost:19530`。

---

### 3.2 中午：Collection、Schema 与数据写入

目标：理解 Collection 类似“向量表”，并能设计适合 RAG 的 schema。

#### 任务 6：设计一个 RAG Chunk Collection

建议字段：

| 字段 | 类型 | 用途 |
| --- | --- | --- |
| id | 主键 | chunk 唯一 ID |
| vector | FLOAT_VECTOR | 文本 embedding |
| text | VARCHAR | chunk 原文 |
| source | VARCHAR | 文档来源 |
| category | VARCHAR | 文档类别 |
| page | INT64 | 页码或段落位置 |
| created_at | INT64 | 时间戳 |

简化版可先用 `create_collection(collection_name, dimension)` 创建默认 schema：

```python
client.create_collection(
    collection_name="rag_chunks",
    dimension=384,
)
```

这种方式默认包含：

- 主键字段：`id`
- 向量字段：`vector`
- 默认 metric：通常是 `COSINE`

更生产化的方式是显式创建 schema：

```python
from pymilvus import MilvusClient, DataType

schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=True,
)

schema.add_field(
    field_name="id",
    datatype=DataType.INT64,
    is_primary=True,
)

schema.add_field(
    field_name="vector",
    datatype=DataType.FLOAT_VECTOR,
    dim=384,
)

schema.add_field(
    field_name="text",
    datatype=DataType.VARCHAR,
    max_length=4096,
)

schema.add_field(
    field_name="source",
    datatype=DataType.VARCHAR,
    max_length=512,
)

schema.add_field(
    field_name="category",
    datatype=DataType.VARCHAR,
    max_length=128,
)

schema.add_field(
    field_name="page",
    datatype=DataType.INT64,
)
```

#### 任务 7：准备样例数据

为了避免今天被 embedding 模型下载卡住，先用随机向量跑通流程。

```python
import random

DIM = 384

docs = [
    {
        "id": 1,
        "text": "Milvus 是一个开源向量数据库，适合构建语义搜索和 RAG 应用。",
        "source": "milvus_notes.md",
        "category": "vector_db",
        "page": 1,
    },
    {
        "id": 2,
        "text": "Qdrant 使用 collection、point、payload 组织向量数据。",
        "source": "qdrant_notes.md",
        "category": "vector_db",
        "page": 1,
    },
    {
        "id": 3,
        "text": "Infinity 强调面向 LLM 应用的混合搜索能力。",
        "source": "infinity_notes.md",
        "category": "vector_db",
        "page": 2,
    },
    {
        "id": 4,
        "text": "RAG 系统通常包含文档切分、向量化、召回、重排和生成。",
        "source": "rag_notes.md",
        "category": "rag",
        "page": 3,
    },
]

def random_vector(dim: int) -> list[float]:
    return [random.random() for _ in range(dim)]

data = [
    {
        **doc,
        "vector": random_vector(DIM),
    }
    for doc in docs
]
```

#### 任务 8：写入数据

```python
res = client.insert(
    collection_name="rag_chunks",
    data=data,
)

print(res)
```

你要观察：

- insert_count 是否等于数据条数。
- ids 是否符合预期。
- 是否因为字段类型、向量维度、字符串长度报错。

常见错误：

| 错误 | 原因 | 解决 |
| --- | --- | --- |
| vector dimension mismatch | 插入向量维度和 collection dim 不一致 | 检查 embedding 模型维度 |
| field not exist | schema 中没有该字段 | 开启 dynamic field 或补充字段 |
| varchar length exceed | 文本超过 max_length | 调大 max_length 或只存 chunk |
| primary key duplicated | 主键重复 | 改用 upsert 或生成唯一 ID |

---

### 3.3 下午：索引、加载与搜索

目标：掌握 Milvus 的核心查询路径：建索引、加载、search、filter、query。

#### 任务 9：理解索引为什么重要

如果没有索引，向量搜索接近暴力扫描。数据量小的时候看不出问题，数据量大后延迟和成本会迅速上升。

Milvus 支持多种索引：

| 索引 | 适用场景 | 特点 |
| --- | --- | --- |
| AUTOINDEX | 新手、默认推荐、希望交给 Milvus 自动选择 | 简化参数选择 |
| HNSW | 低延迟、高召回常见选择 | 内存占用较高 |
| IVF_FLAT | 中大规模、可调召回和速度 | 需要设置 nlist、nprobe |
| IVF_SQ8 | 更省内存 | 有量化损失 |
| DISKANN | 更大规模、磁盘友好 | 适合特定部署条件 |

今天建议先使用 `AUTOINDEX`，再了解 HNSW 参数。

#### 任务 10：创建索引

```python
index_params = client.prepare_index_params()

index_params.add_index(
    field_name="vector",
    index_type="AUTOINDEX",
    metric_type="COSINE",
)

client.create_index(
    collection_name="rag_chunks",
    index_params=index_params,
)
```

HNSW 示例：

```python
index_params = client.prepare_index_params()

index_params.add_index(
    field_name="vector",
    index_type="HNSW",
    metric_type="COSINE",
    params={
        "M": 16,
        "efConstruction": 200,
    },
)

client.create_index(
    collection_name="rag_chunks",
    index_params=index_params,
)
```

HNSW 参数理解：

| 参数 | 含义 | 趋势 |
| --- | --- | --- |
| M | 图中每个节点的连接数 | 越大召回可能越好，内存越高 |
| efConstruction | 建图时的候选集合大小 | 越大索引质量越好，构建越慢 |
| ef | 搜索时的候选集合大小 | 越大召回越好，延迟越高 |

#### 任务 11：加载 Collection

```python
client.load_collection(collection_name="rag_chunks")
```

可以查看加载状态：

```python
state = client.get_load_state(collection_name="rag_chunks")
print(state)
```

理解：

- Collection 创建和写入不等于马上可高性能搜索。
- 搜索前需要确保 collection 已被加载。
- 在生产环境中，load 会消耗内存资源。

#### 任务 12：基本向量搜索

```python
query_vector = random_vector(DIM)

results = client.search(
    collection_name="rag_chunks",
    data=[query_vector],
    anns_field="vector",
    limit=3,
    output_fields=["text", "source", "category", "page"],
    search_params={
        "metric_type": "COSINE",
    },
)

for hits in results:
    for hit in hits:
        print(hit)
```

重点观察：

- `id`：命中的主键。
- `distance` 或 `score`：相似度分数。
- `entity`：返回的 metadata 字段。
- `limit`：TopK 数量。
- `output_fields`：控制返回字段，避免返回过多内容。

#### 任务 13：带过滤条件的搜索

```python
results = client.search(
    collection_name="rag_chunks",
    data=[query_vector],
    anns_field="vector",
    limit=3,
    filter='category == "vector_db"',
    output_fields=["text", "source", "category", "page"],
    search_params={
        "metric_type": "COSINE",
    },
)

for hits in results:
    for hit in hits:
        print(hit)
```

RAG 中常见过滤条件：

```text
source == "xxx.pdf"
category == "policy"
created_at >= 1710000000
tenant_id == "team_a"
doc_id in ["doc_1", "doc_2"]
```

过滤搜索的价值：

- 多租户数据隔离。
- 限定知识库范围。
- 限定时间范围。
- 限定文档类型。
- 支持权限控制。

#### 任务 14：Query 非向量查询

如果你知道主键或需要按条件查 metadata，可以用 query：

```python
rows = client.query(
    collection_name="rag_chunks",
    filter='category == "rag"',
    output_fields=["id", "text", "source", "page"],
)

for row in rows:
    print(row)
```

区别：

| API | 用途 |
| --- | --- |
| search | 向量相似度搜索 |
| query | 标量条件查询 |
| get | 根据主键获取 |
| insert | 插入 |
| upsert | 插入或更新 |
| delete | 删除 |

---

### 3.4 晚上：生产化理解与横向比较

目标：从“能用”走向“知道如何上线”。

#### 任务 15：理解生产级向量数据库的关键问题

生产中你需要考虑：

1. 数据规模  
   向量数量是 10 万、100 万、1 亿还是 10 亿？

2. 向量维度  
   384、768、1024、1536、3072 不同维度会显著影响内存、索引、搜索成本。

3. 写入模式  
   是离线批量导入，还是实时增量写入？

4. 查询模式  
   是低 QPS 管理系统，还是高 QPS 在线推荐/搜索？

5. 召回率要求  
   是“差不多能用”，还是需要严格评估 Recall@K？

6. 延迟要求  
   P50、P95、P99 分别是多少？

7. 多租户隔离  
   用 database、collection、partition，还是 metadata filter？

8. 权限控制  
   应用层过滤，还是数据库层隔离？

9. 备份恢复  
   是否有定期备份，是否演练恢复？

10. 监控告警  
    是否监控 QPS、延迟、内存、磁盘、segment、index build？

11. 成本控制  
    索引内存、磁盘、CPU、GPU、云服务费用如何权衡？

#### 任务 16：Milvus、Infinity、Qdrant 对比

| 维度 | Milvus | Infinity | Qdrant |
| --- | --- | --- | --- |
| 定位 | 大规模生产级向量数据库 | AI-native database，强调混合检索 | 易用、高性能向量搜索引擎/数据库 |
| 主要语言 | Go/C++ 等 | C++/Python 等 | Rust |
| 核心能力 | 向量检索、混合检索、分布式、索引体系 | dense/sparse/tensor/full-text 混合搜索 | 向量搜索、payload 过滤、易部署 |
| 部署模式 | Lite、Standalone、Distributed、K8s | 单二进制、Docker、Python embedded | Docker、binary、cloud、K8s |
| Python 体验 | PyMilvus/MilvusClient | infinity-sdk | qdrant-client |
| 适合场景 | 大规模 RAG、语义搜索、推荐、企业生产环境 | 需要全文、稠密、稀疏、多向量融合的 LLM 应用 | 快速构建语义搜索、过滤检索、服务化 RAG |
| 学习重点 | schema、index、load、search、分布式、监控 | 混合搜索表达能力 | collection、point、payload、filter |
| 生产复杂度 | 中到高 | 中 | 低到中 |

简单选择建议：

- 想学习生产级向量数据库体系：优先 Milvus。
- 想研究混合检索和 AI-native database：关注 Infinity。
- 想快速落地、API 简洁、过滤好用：Qdrant 很值得实践。
- 如果团队已有 Kubernetes、Prometheus、对象存储和大规模数据治理需求：Milvus 的生产生态更值得深入。

---

## 4. 完整 Python SDK 实战脚本

建议保存为：

```text
scripts/milvus_day12_demo.py
```

完整脚本：

```python
import random
from typing import Any

from pymilvus import DataType, MilvusClient


MILVUS_URI = "http://localhost:19530"
MILVUS_TOKEN = "root:Milvus"
COLLECTION_NAME = "rag_chunks_day12"
DIM = 384


def random_vector(dim: int) -> list[float]:
    return [random.random() for _ in range(dim)]


def connect() -> MilvusClient:
    return MilvusClient(
        uri=MILVUS_URI,
        token=MILVUS_TOKEN,
    )


def recreate_collection(client: MilvusClient) -> None:
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)

    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
    )

    schema.add_field(
        field_name="id",
        datatype=DataType.INT64,
        is_primary=True,
    )
    schema.add_field(
        field_name="vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=DIM,
    )
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=4096,
    )
    schema.add_field(
        field_name="source",
        datatype=DataType.VARCHAR,
        max_length=512,
    )
    schema.add_field(
        field_name="category",
        datatype=DataType.VARCHAR,
        max_length=128,
    )
    schema.add_field(
        field_name="page",
        datatype=DataType.INT64,
    )

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )


def build_demo_data() -> list[dict[str, Any]]:
    docs = [
        {
            "id": 1,
            "text": "Milvus 是一个开源向量数据库，适合构建语义搜索和 RAG 应用。",
            "source": "milvus_notes.md",
            "category": "vector_db",
            "page": 1,
        },
        {
            "id": 2,
            "text": "Qdrant 使用 collection、point、payload 组织向量数据。",
            "source": "qdrant_notes.md",
            "category": "vector_db",
            "page": 1,
        },
        {
            "id": 3,
            "text": "Infinity 强调面向 LLM 应用的混合搜索能力。",
            "source": "infinity_notes.md",
            "category": "vector_db",
            "page": 2,
        },
        {
            "id": 4,
            "text": "RAG 系统通常包含文档切分、向量化、召回、重排和生成。",
            "source": "rag_notes.md",
            "category": "rag",
            "page": 3,
        },
        {
            "id": 5,
            "text": "生产级向量数据库需要关注备份、监控、索引构建、资源隔离和查询延迟。",
            "source": "production_notes.md",
            "category": "production",
            "page": 4,
        },
    ]

    return [
        {
            **doc,
            "vector": random_vector(DIM),
        }
        for doc in docs
    ]


def insert_data(client: MilvusClient, data: list[dict[str, Any]]) -> None:
    result = client.insert(
        collection_name=COLLECTION_NAME,
        data=data,
    )
    print("Insert result:", result)


def search_all(client: MilvusClient) -> None:
    query_vector = random_vector(DIM)
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        anns_field="vector",
        limit=3,
        output_fields=["text", "source", "category", "page"],
        search_params={
            "metric_type": "COSINE",
        },
    )

    print("\n=== Basic vector search ===")
    for hits in results:
        for hit in hits:
            print(hit)


def search_with_filter(client: MilvusClient) -> None:
    query_vector = random_vector(DIM)
    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vector],
        anns_field="vector",
        limit=3,
        filter='category == "vector_db"',
        output_fields=["text", "source", "category", "page"],
        search_params={
            "metric_type": "COSINE",
        },
    )

    print("\n=== Filtered vector search ===")
    for hits in results:
        for hit in hits:
            print(hit)


def query_by_metadata(client: MilvusClient) -> None:
    rows = client.query(
        collection_name=COLLECTION_NAME,
        filter='category == "rag"',
        output_fields=["id", "text", "source", "page"],
    )

    print("\n=== Metadata query ===")
    for row in rows:
        print(row)


def upsert_one(client: MilvusClient) -> None:
    row = {
        "id": 4,
        "text": "RAG 系统由 indexing pipeline 和 query pipeline 组成。",
        "source": "rag_notes.md",
        "category": "rag",
        "page": 10,
        "vector": random_vector(DIM),
    }

    result = client.upsert(
        collection_name=COLLECTION_NAME,
        data=[row],
    )
    print("\n=== Upsert result ===")
    print(result)


def delete_one(client: MilvusClient) -> None:
    result = client.delete(
        collection_name=COLLECTION_NAME,
        filter="id == 5",
    )
    print("\n=== Delete result ===")
    print(result)


def main() -> None:
    client = connect()
    print("Collections before demo:", client.list_collections())

    recreate_collection(client)
    print("Collection recreated:", COLLECTION_NAME)

    data = build_demo_data()
    insert_data(client, data)

    client.flush(collection_name=COLLECTION_NAME)
    client.load_collection(collection_name=COLLECTION_NAME)
    print("Load state:", client.get_load_state(collection_name=COLLECTION_NAME))

    search_all(client)
    search_with_filter(client)
    query_by_metadata(client)
    upsert_one(client)
    delete_one(client)

    print("\nCollections after demo:", client.list_collections())


if __name__ == "__main__":
    main()
```

运行：

```bash
python scripts/milvus_day12_demo.py
```

预期结果：

- 能成功连接 Milvus。
- 能创建 `rag_chunks_day12` collection。
- 能插入 5 条数据。
- 能执行 TopK 向量搜索。
- 能执行 metadata filter。
- 能 query、upsert、delete。

---

## 5. 使用真实 Embedding 的版本

随机向量只能验证流程，不能验证语义效果。跑通 SDK 后，应替换为真实 embedding。

### 5.1 使用 pymilvus 默认模型

安装：

```bash
pip install -U "pymilvus[model]"
```

示例：

```python
from pymilvus import model

embedding_fn = model.DefaultEmbeddingFunction()

docs = [
    "Milvus 是一个向量数据库。",
    "Qdrant 是一个向量搜索引擎。",
    "RAG 系统需要把问题转成向量后召回文档。",
]

doc_vectors = embedding_fn.encode_documents(docs)
query_vectors = embedding_fn.encode_queries(["什么是向量数据库？"])

print("Embedding dim:", embedding_fn.dim)
print("First vector length:", len(doc_vectors[0]))
```

然后创建 collection 时使用：

```python
dimension=embedding_fn.dim
```

### 5.2 使用你自己的 Embedding 模型

生产项目更常见：

```python
from openai import OpenAI

client = OpenAI()

def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]
```

注意：

- 你必须确认模型输出维度。
- collection 的 vector dim 必须和 embedding 维度完全一致。
- 更换 embedding 模型通常意味着需要重建索引和重新写入向量。

---

## 6. Milvus 在 RAG 系统中的工程设计

### 6.1 推荐 Collection Schema

```text
collection: kb_chunks

id: int64 primary key
vector: float_vector(dim=embedding_dim)
text: varchar(max_length=8192)
doc_id: varchar(max_length=128)
chunk_id: varchar(max_length=128)
source: varchar(max_length=512)
title: varchar(max_length=512)
tenant_id: varchar(max_length=128)
category: varchar(max_length=128)
created_at: int64
updated_at: int64
metadata: json
```

字段设计建议：

- `id` 用数据库内部主键，稳定即可。
- `doc_id` 表示原始文档。
- `chunk_id` 表示切分后的块。
- `tenant_id` 用于多租户过滤。
- `source` 存文件路径、URL 或业务来源。
- `metadata` 存扩展属性，但常用过滤字段建议单独建 scalar 字段。

### 6.2 多租户设计选择

| 方案 | 优点 | 缺点 | 适用 |
| --- | --- | --- | --- |
| 每个租户一个 collection | 隔离强，权限简单 | collection 数量多，管理复杂 | 租户少且数据大 |
| 一个 collection + tenant_id filter | 简单，资源利用率高 | 依赖过滤正确性 | 租户多且数据中小 |
| 每个业务一个 database | 命名空间清晰 | 仍要管理资源 | 多业务线 |
| partition key | 查询可裁剪范围 | 设计更复杂 | 大规模多分区 |

今天先掌握：

```python
filter='tenant_id == "team_a"'
```

### 6.3 RAG 写入链路

```text
原始文档
  -> 文档解析
  -> 文本清洗
  -> Chunk 切分
  -> 生成 chunk_id/doc_id
  -> 调用 embedding 模型
  -> 写入 Milvus
  -> flush/load/index
  -> 记录写入日志
```

写入时要注意：

- 每个 chunk 要有稳定 ID。
- 保留原文和来源。
- 保留权限信息。
- 批量写入，不要一条条写入生产数据。
- embedding 失败要可重试。
- 写入 Milvus 成功后，应用层索引状态要同步更新。

### 6.4 RAG 查询链路

```text
用户问题
  -> query rewrite 可选
  -> query embedding
  -> Milvus vector search + metadata filter
  -> TopK candidates
  -> rerank
  -> context packing
  -> LLM answer
  -> citation/source 输出
```

Milvus 查询建议：

- `limit` 不要太小，给 reranker 留候选空间。
- 使用 `output_fields` 只返回必要字段。
- 对权限、租户、文档范围使用 filter。
- 根据线上评估调 `TopK`、metric、index params。

---

## 7. 生产运维检查清单

### 7.1 部署检查

- Docker 镜像版本是否固定，而不是永远使用 latest。
- 数据卷是否挂载到可靠磁盘。
- 端口是否只暴露给可信网络。
- 是否配置资源限制。
- 是否配置重启策略。
- 是否有日志采集。
- 是否有健康检查。

### 7.2 数据检查

- collection schema 是否记录在代码或迁移脚本中。
- embedding 模型名称、版本、维度是否记录。
- metric type 是否和 embedding 模型匹配。
- 主键是否稳定且不会重复。
- 是否有重复 chunk 清理策略。
- 删除文档时是否同步删除向量。

### 7.3 性能检查

- 数据量增长后是否重新评估索引类型。
- TopK 和 filter 是否导致延迟升高。
- P95/P99 延迟是否达标。
- 是否区分冷启动和热查询。
- 是否定期评估 Recall@K。

### 7.4 可靠性检查

- 是否有备份策略。
- 是否做过恢复演练。
- 是否监控磁盘空间。
- 是否监控内存。
- 是否监控容器重启。
- 是否有版本升级计划。

### 7.5 安全检查

- 默认 token 是否修改。
- 是否限制公网访问。
- 是否按租户过滤数据。
- 是否在应用层做权限校验。
- 日志中是否避免泄露敏感文本。

---

## 8. 常见问题排查

### 8.1 Docker 启动失败

检查：

```bash
docker ps -a
docker logs milvus
```

可能原因：

- Docker Desktop 未启动。
- 端口冲突。
- 镜像拉取失败。
- WSL2 网络异常。
- 磁盘空间不足。

### 8.2 Python 连接失败

检查：

```python
from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
print(client.list_collections())
```

可能原因：

- Milvus 容器没启动。
- 端口不是 `19530`。
- 本机和 WSL2 网络不通。
- token 错误。
- Python 环境装错包。

### 8.3 插入时报维度错误

原因：

```text
collection dim != embedding vector length
```

解决：

```python
print(len(vector))
```

然后重建 collection：

```python
client.drop_collection("your_collection")
client.create_collection("your_collection", dimension=len(vector))
```

生产中不要随便 drop，需要迁移策略。

### 8.4 搜索结果看起来不语义相关

可能原因：

- 你使用的是随机向量。
- 文档 embedding 和 query embedding 不是同一个模型。
- metric type 选择不匹配。
- chunk 切分质量差。
- TopK 太小。
- filter 过严。
- 需要 rerank。

### 8.5 插入后搜不到

检查：

```python
client.flush(collection_name="xxx")
client.load_collection(collection_name="xxx")
```

以及：

```python
print(client.query(collection_name="xxx", filter="id == 1", output_fields=["*"]))
```

---

## 9. 今日验收标准

完成以下清单，才算真正掌握 Day 12。

### 9.1 基础验收

- 能用 Docker 启动 Milvus。
- 能打开 Milvus WebUI。
- 能用 Python `MilvusClient` 连接。
- 能创建 collection。
- 能插入向量数据。
- 能创建索引。
- 能 load collection。
- 能执行 TopK search。
- 能使用 filter。
- 能 query metadata。
- 能 upsert 和 delete。

### 9.2 理解验收

你需要能回答：

1. Collection 和关系数据库表有什么相似和不同？
2. 向量字段的 dim 为什么必须固定？
3. COSINE、IP、L2 有什么区别？
4. 为什么需要向量索引？
5. HNSW 的 M、efConstruction、ef 大致影响什么？
6. RAG 中为什么要 metadata filter？
7. 什么时候一个 collection 不够，需要 partition 或多 collection？
8. Milvus、Infinity、Qdrant 的定位差异是什么？
9. 生产环境为什么要关心 backup、monitoring、upgrade？
10. 更换 embedding 模型会对向量库造成什么影响？

### 9.3 实战验收

实现一个最小 RAG 检索器：

```python
def retrieve(query: str, top_k: int = 5, category: str | None = None) -> list[dict]:
    ...
```

要求：

- 输入 query。
- 生成 query embedding。
- 调用 Milvus search。
- 支持 category filter。
- 返回 text、source、score。
- 对空结果做处理。

---

## 10. 课后扩展任务

### 10.1 加入真实文档

选择一份 PDF 或 Markdown：

1. 解析文本。
2. 按 300 到 800 token 切分。
3. 生成 embedding。
4. 写入 Milvus。
5. 查询 5 个问题。
6. 观察召回质量。

### 10.2 加入 Rerank

流程：

```text
Milvus Top20 召回 -> Reranker Top5 -> LLM
```

你可以比较：

- 只用 Milvus Top5。
- Milvus Top20 + Rerank Top5。

观察：

- 答案准确率。
- 延迟。
- 成本。

### 10.3 对比 Qdrant

用 Qdrant 实现同样的 collection、insert、search、filter。

重点比较：

- API 是否更直观。
- payload filter 写法。
- Docker 部署体验。
- Python SDK 体验。
- 搜索结果结构。

### 10.4 对比 Infinity

重点关注：

- dense vector search。
- full-text search。
- hybrid search。
- reranker 支持。
- 单二进制/嵌入式部署体验。

思考：

```text
如果我正在构建 Advanced RAG 系统：
Milvus 负责什么？
Infinity 负责什么？
Qdrant 负责什么？
什么时候它们可以互相替代？
什么时候它们不能互相替代？
```

---

## 11. 今日总结模板

学习完成后，建议写下：

```markdown
# Day 12 总结：生产级向量数据库 Milvus

## 今天完成了什么

- [ ] Docker 启动 Milvus
- [ ] Python SDK 连接
- [ ] 创建 collection
- [ ] 插入数据
- [ ] 创建索引
- [ ] 向量搜索
- [ ] metadata filter
- [ ] query/upsert/delete

## 我理解最深的 3 个概念

1.
2.
3.

## 我遇到的问题

1.
2.
3.

## Milvus 接入 RAG 的关键点

-
-
-

## Milvus、Infinity、Qdrant 对比结论

-
-
-

## 明天要继续改进什么

-
-
-
```

---

## 12. 一句话心法

Milvus 不只是一个“存向量的地方”。在生产 RAG 中，它是知识召回系统的核心基础设施。今天要练到的能力，不是背 API，而是知道如何把 embedding、schema、index、filter、search、monitoring 串成一条可靠的检索链路。

