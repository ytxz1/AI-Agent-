# Docker 打包与部署 RAG 项目详解

> 本文目标：把第 5-6 天完成的 `naive-rag-api` 项目打包成 Docker 镜像，并通过 Docker Compose 在本地成功运行。

## 0. 你最终要做到什么

最终你应该能在 RAG 项目根目录执行：

```powershell
docker compose up --build
```

然后打开：

```text
http://127.0.0.1:8000/docs
```

并完成：

1. 健康检查成功。
2. 上传文档成功。
3. 查看文档列表成功。
4. 提问成功。
5. 返回 answer 和 sources。
6. 停止并重启容器后，已上传文档仍然可检索。

## 1. 部署前项目结构检查

进入你的 RAG 项目目录：

```powershell
cd "第5-6天手撕 Naive RAG 系统\naive-rag-api"
```

推荐结构如下：

```text
naive-rag-api/
  app/
    __init__.py
    main.py
    core/
      config.py
    api/
      routes/
        health.py
        documents.py
        chat.py
    schemas/
    services/
    prompts/
  data/
    uploads/
    chroma/
    metadata/
  tests/
  requirements.txt
  .env
  .env.example
  README.md
```

最少必须存在：

```text
app/main.py
requirements.txt
.env.example
```

如果没有 `requirements.txt`，先在虚拟环境中生成：

```powershell
pip freeze > requirements.txt
```

更推荐手工整理一个干净版本，例如：

```text
fastapi
uvicorn[standard]
python-dotenv
pydantic-settings
python-multipart
langchain
langchain-core
langchain-community
langchain-text-splitters
langchain-openai
langchain-chroma
chromadb
pypdf
```

如果你使用本地 embedding，再加：

```text
langchain-huggingface
sentence-transformers
```

注意：`pip freeze` 会把很多间接依赖和本机环境细节也写进去。学习阶段可以先用，项目稳定后建议整理为直接依赖清单。

## 2. 部署前先确认本地可运行

容器化之前，先本地启动一次：

```powershell
uvicorn app.main:app --reload
```

检查：

```powershell
curl http://127.0.0.1:8000/health
```

如果这一步失败，先修本地项目。常见原因：

1. 依赖没装全。
2. `.env` 不存在。
3. `OPENAI_API_KEY` 没配置。
4. `app.main:app` 路径写错。
5. 文件夹不在项目根目录。
6. Chroma 初始化路径不可写。

## 3. 配置 `.env.example`

文件：`.env.example`

```text
APP_NAME=Naive RAG API
APP_ENV=dev
UPLOAD_DIR=data/uploads
CHROMA_DIR=data/chroma
METADATA_DIR=data/metadata
CHROMA_COLLECTION=naive_rag
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
DEFAULT_TOP_K=4
CHAT_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=your_api_key_here
```

复制一份真实配置：

```powershell
Copy-Item .env.example .env
```

然后编辑 `.env`：

```text
OPENAI_API_KEY=你的真实 key
```

重要原则：

1. `.env.example` 可以提交。
2. `.env` 不提交。
3. `.env` 不打进镜像。
4. 运行容器时再注入 `.env`。

## 4. 编写 `.dockerignore`

文件：`.dockerignore`

```dockerignore
.git
.gitignore

.venv
venv
env

__pycache__
*.pyc
*.pyo
*.pyd
.pytest_cache
.mypy_cache
.ruff_cache

.env
.env.*
!.env.example

data
logs
tmp

*.sqlite
*.db

Dockerfile
docker-compose.override.yml
README.local.md
```

为什么要写 `.dockerignore`？

1. 减少构建上下文，构建更快。
2. 避免把 `.env` 密钥打进镜像。
3. 避免把本地向量库数据打进镜像。
4. 避免把 `.venv` 这种本机环境复制进 Linux 容器。
5. 避免缓存文件污染镜像。

注意：是否忽略 `Dockerfile` 本身没有绝对标准。Docker 构建时仍然能读取 Dockerfile，因为它是由 `-f` 指定或默认读取的；忽略它只是避免 Dockerfile 被 `COPY . .` 复制进最终镜像。学习阶段保留或忽略都可以，这里选择忽略。

## 5. 编写 Dockerfile

文件：`Dockerfile`

```dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY app ./app
COPY .env.example ./.env.example

RUN mkdir -p data/uploads data/chroma data/metadata

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 5.1 每一行在做什么

```dockerfile
FROM python:3.12-slim
```

使用官方 Python 3.12 的 slim 镜像。它比完整版镜像更小，但仍适合大多数 Python Web 项目。

```dockerfile
ENV PYTHONDONTWRITEBYTECODE=1
```

避免 Python 写入 `.pyc` 文件，镜像和容器目录更干净。

```dockerfile
ENV PYTHONUNBUFFERED=1
```

让日志实时输出，方便 `docker logs` 查看服务启动和报错。

```dockerfile
ENV PIP_NO_CACHE_DIR=1
```

减少 pip 缓存占用，镜像更小。

```dockerfile
WORKDIR /app
```

容器内工作目录设为 `/app`。后续命令都在这里执行。

```dockerfile
RUN apt-get update ...
```

安装必要系统依赖。部分 Python 包可能需要编译，`build-essential` 可以减少安装失败概率。`curl` 用于容器内调试或健康检查。

```dockerfile
COPY requirements.txt .
RUN pip install ...
```

先复制依赖文件，再安装依赖。这样只要 `requirements.txt` 不变，Docker 可以复用依赖安装缓存。

```dockerfile
COPY app ./app
```

复制业务代码。

```dockerfile
RUN mkdir -p data/uploads data/chroma data/metadata
```

创建数据目录。运行时会用 volume 挂载覆盖 `/app/data`，但镜像内也保留默认目录，避免未挂载时启动失败。

```dockerfile
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

容器启动时运行 FastAPI 服务。必须监听 `0.0.0.0`，否则宿主机访问不到容器内服务。

## 6. 构建镜像

在 `naive-rag-api` 目录执行：

```powershell
docker build -t naive-rag-api:dev .
```

查看镜像：

```powershell
docker images
```

你应该能看到类似：

```text
REPOSITORY       TAG       IMAGE ID       CREATED          SIZE
naive-rag-api    dev       xxxxxxxx       x minutes ago    xxxMB
```

### 6.1 构建失败排查

#### 问题 1：`requirements.txt` 不存在

现象：

```text
COPY failed: file not found in build context
```

解决：

```powershell
pip freeze > requirements.txt
```

或手动创建干净的依赖清单。

#### 问题 2：pip 安装失败

常见原因：

1. 网络访问失败。
2. 某个包版本不存在。
3. Python 版本和依赖不兼容。
4. 缺少系统编译依赖。

解决思路：

1. 先在本地虚拟环境安装验证。
2. 固定问题包版本。
3. 使用 Python 3.11 或 3.12 的官方镜像。
4. 必要时增加系统依赖。

#### 问题 3：镜像构建很慢

常见原因：

1. `.dockerignore` 没写，构建上下文太大。
2. `data/chroma` 被复制进镜像。
3. 每次代码变更都触发依赖重装。

解决：

1. 确保 `.dockerignore` 忽略 `data`、`.venv`。
2. Dockerfile 中先 `COPY requirements.txt`，再 `COPY app`。

## 7. 用 docker run 启动容器

先创建数据目录：

```powershell
New-Item -ItemType Directory -Force data, data\uploads, data\chroma, data\metadata
```

启动：

```powershell
docker run --rm `
  --name naive-rag-api `
  --env-file .env `
  -p 8000:8000 `
  -v "${PWD}\data:/app/data" `
  naive-rag-api:dev
```

参数解释：

| 参数 | 含义 |
|---|---|
| `--rm` | 容器停止后自动删除容器实例 |
| `--name naive-rag-api` | 给容器命名 |
| `--env-file .env` | 注入环境变量 |
| `-p 8000:8000` | 宿主机 8000 映射到容器 8000 |
| `-v "${PWD}\data:/app/data"` | 把本地 data 目录挂载到容器 `/app/data` |
| `naive-rag-api:dev` | 使用刚构建的镜像 |

打开：

```text
http://127.0.0.1:8000/docs
```

健康检查：

```powershell
curl http://127.0.0.1:8000/health
```

如果你看到 `{"status":"ok",...}`，说明服务启动成功。

## 8. 编写 docker-compose.yml

文件：`docker-compose.yml`

```yaml
services:
  rag-api:
    build:
      context: .
      dockerfile: Dockerfile
    image: naive-rag-api:dev
    container_name: naive-rag-api
    env_file:
      - .env
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 20s
```

启动：

```powershell
docker compose up --build
```

后台启动：

```powershell
docker compose up --build -d
```

查看日志：

```powershell
docker compose logs -f rag-api
```

查看容器状态：

```powershell
docker compose ps
```

停止：

```powershell
docker compose down
```

只重新构建：

```powershell
docker compose build --no-cache
```

## 9. API 验收命令

### 9.1 健康检查

```powershell
curl http://127.0.0.1:8000/health
```

期望：

```json
{
  "status": "ok",
  "app": "Naive RAG API",
  "vector_store_dir": "data/chroma"
}
```

字段可能与你的实际代码略有差异，重点是 HTTP 状态码为 200，并且服务没有报错。

### 9.2 准备测试文档

创建目录：

```powershell
New-Item -ItemType Directory -Force samples
```

文件：`samples/rag_notes.md`

```markdown
# RAG 学习笔记

RAG 是 Retrieval Augmented Generation 的缩写，中文常译为检索增强生成。

一个基础 RAG 系统通常包含两个阶段：索引阶段和查询阶段。

索引阶段包括文档加载、文本切分、Embedding 编码和向量存储。

查询阶段包括用户问题向量化、相似度检索、上下文组装和大模型生成。

chunk size 会影响检索效果。chunk 太小可能导致上下文不完整，chunk 太大可能引入噪声。

metadata 用于记录文档来源、页码、文件名、chunk 编号等信息，方便答案溯源。
```

### 9.3 上传文档

```powershell
curl -X POST "http://127.0.0.1:8000/api/v1/documents/upload" `
  -F "file=@.\samples\rag_notes.md"
```

期望：

```json
{
  "document_id": "...",
  "filename": "rag_notes.md",
  "chunk_count": 1,
  "status": "indexed"
}
```

如果这里失败，优先检查：

1. 是否安装 `python-multipart`。
2. 上传接口路径是否与你代码一致。
3. 容器内 `/app/data/uploads` 是否可写。
4. `OPENAI_API_KEY` 是否可用，因为上传时可能会调用 embedding。

### 9.4 查看文档列表

```powershell
curl http://127.0.0.1:8000/api/v1/documents
```

期望能看到刚上传的文件。

### 9.5 发起 RAG 问答

```powershell
curl -X POST "http://127.0.0.1:8000/api/v1/chat/query" `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"RAG 的索引阶段包括哪些步骤？\",\"top_k\":4}"
```

期望：

1. `answer` 中提到文档加载、文本切分、Embedding、向量存储。
2. `sources` 不为空。
3. `sources[0].filename` 是 `rag_notes.md`。

### 9.6 无答案问题测试

```powershell
curl -X POST "http://127.0.0.1:8000/api/v1/chat/query" `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"这份文档的作者生日是哪一天？\",\"top_k\":4}"
```

期望：

1. 不应该编造具体生日。
2. 应该回答类似“根据已上传文档，我不知道”。
3. 即使 sources 返回了低相关片段，answer 也不能胡编。

## 10. 持久化验收

RAG 项目部署时，最容易被忽略的是数据持久化。

执行下面流程：

### 10.1 启动服务

```powershell
docker compose up --build -d
```

### 10.2 上传文档并提问

完成上面的上传和问答。

### 10.3 停止容器

```powershell
docker compose down
```

### 10.4 重新启动

```powershell
docker compose up -d
```

### 10.5 不重新上传，直接提问

```powershell
curl -X POST "http://127.0.0.1:8000/api/v1/chat/query" `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"metadata 有什么作用？\",\"top_k\":4}"
```

如果能回答，说明至少这些目录持久化成功：

```text
data/chroma
data/metadata
```

你还可以检查宿主机目录：

```powershell
Get-ChildItem data -Recurse
```

应该能看到上传文件、metadata JSON、Chroma 数据文件。

## 11. Docker 中的路径问题

本地运行时，你可能看到：

```text
data/chroma
```

容器内运行时，真实路径是：

```text
/app/data/chroma
```

但代码里仍然可以写相对路径：

```python
chroma_dir: Path = Path("data/chroma")
```

因为 Dockerfile 设置了：

```dockerfile
WORKDIR /app
```

所以容器内的相对路径 `data/chroma` 会解析成 `/app/data/chroma`。

这就是为什么 `WORKDIR` 很重要。没有它，代码可能在容器中从错误目录启动，导致相对路径全部失效。

## 12. 为什么不要把 data 打进镜像

RAG 项目的 `data` 目录通常包含：

1. 用户上传的原始文档。
2. Chroma 向量库。
3. metadata registry。
4. 临时处理文件。

这些数据不应该打进镜像，因为：

1. 镜像会越来越大。
2. 用户数据和代码耦合。
3. 更新代码时可能覆盖数据。
4. 密钥或敏感文档可能泄漏。
5. 多环境部署时数据不可控。

正确做法：

```yaml
volumes:
  - ./data:/app/data
```

这表示：

1. 镜像提供运行环境。
2. 容器运行服务。
3. 宿主机保存数据。

## 13. 常见问题排障

### 13.1 访问不到 `http://127.0.0.1:8000`

检查容器是否启动：

```powershell
docker compose ps
```

检查日志：

```powershell
docker compose logs -f rag-api
```

常见原因：

1. `uvicorn` 没有监听 `0.0.0.0`。
2. 端口映射没有写 `8000:8000`。
3. 宿主机 8000 端口已被占用。
4. 服务启动时 import 报错，容器已退出。

解决：

```dockerfile
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

如果端口冲突，可以改 compose：

```yaml
ports:
  - "8001:8000"
```

然后访问：

```text
http://127.0.0.1:8001/docs
```

### 13.2 `ModuleNotFoundError`

现象：

```text
ModuleNotFoundError: No module named 'app'
```

排查：

1. Dockerfile 是否 `WORKDIR /app`。
2. 是否 `COPY app ./app`。
3. `app/__init__.py` 是否存在。
4. 启动命令是否是 `uvicorn app.main:app`。

### 13.3 `OPENAI_API_KEY` 缺失

现象：

```text
OpenAIError: The api_key client option must be set
```

排查：

```powershell
docker compose exec rag-api env
```

看是否有：

```text
OPENAI_API_KEY=...
```

解决：

1. 确认 `.env` 存在。
2. 确认 compose 中有 `env_file: .env`。
3. 修改 `.env` 后重启容器：

```powershell
docker compose down
docker compose up -d
```

### 13.4 上传文件时报 `python-multipart` 缺失

现象：

```text
RuntimeError: Form data requires "python-multipart" to be installed
```

解决：

在 `requirements.txt` 中加入：

```text
python-multipart
```

重新构建：

```powershell
docker compose build --no-cache
docker compose up -d
```

### 13.5 Chroma 数据没有持久化

现象：

1. 上传文档后能问答。
2. 重启容器后文档消失。

排查：

1. compose 是否挂载 `./data:/app/data`。
2. `.env` 中 `CHROMA_DIR` 是否仍然是 `data/chroma`。
3. 代码是否真的使用 `settings.chroma_dir`。
4. 宿主机 `data/chroma` 是否有文件。

正确配置：

```yaml
volumes:
  - ./data:/app/data
```

```text
CHROMA_DIR=data/chroma
METADATA_DIR=data/metadata
UPLOAD_DIR=data/uploads
```

### 13.6 Windows 路径挂载失败

如果 `docker run -v "${PWD}\data:/app/data"` 失败，可以改用绝对路径：

```powershell
docker run --rm `
  --env-file .env `
  -p 8000:8000 `
  -v "D:\你的项目路径\naive-rag-api\data:/app/data" `
  naive-rag-api:dev
```

更推荐使用 compose，因为：

```yaml
volumes:
  - ./data:/app/data
```

在项目目录内更稳定、更可读。

### 13.7 容器启动后马上退出

查看日志：

```powershell
docker logs naive-rag-api
```

或：

```powershell
docker compose logs rag-api
```

常见原因：

1. 依赖缺失。
2. import 路径错误。
3. 环境变量缺失。
4. 配置类初始化失败。
5. Chroma 初始化失败。

思路：容器退出不是 Docker 本身的问题，通常是应用进程崩了。先看日志第一条 Python traceback。

## 14. 开发模式和部署模式的区别

本地开发常用：

```powershell
uvicorn app.main:app --reload
```

Docker 部署不建议默认使用 `--reload`：

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

原因：

1. `--reload` 会启动额外监听进程。
2. 容器里文件变化监听没有必要。
3. 部署环境更需要稳定启动。

如果你想用 Docker 做开发热更新，可以单独写 `docker-compose.dev.yml`：

```yaml
services:
  rag-api:
    build: .
    env_file:
      - .env
    ports:
      - "8000:8000"
    volumes:
      - ./app:/app/app
      - ./data:/app/data
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

启动：

```powershell
docker compose -f docker-compose.dev.yml up --build
```

学习阶段可以先不做 dev compose，避免复杂度过早上升。

## 15. 镜像优化方向

基础版 Dockerfile 已经够学习使用。后续可以优化：

1. 固定依赖版本。
2. 使用非 root 用户运行。
3. 使用 multi-stage build。
4. 增加 `/health` 健康检查。
5. 增加启动前配置检查脚本。
6. 增加 CI 中的 `docker build` 校验。
7. 区分 dev、test、prod 环境配置。

### 15.1 非 root 用户示例

```dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY app ./app
COPY .env.example ./.env.example

RUN mkdir -p data/uploads data/chroma data/metadata \
    && useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

注意：Windows + Docker Desktop + volume 挂载时，权限问题可能表现得不明显；Linux 服务器上更容易遇到目录权限问题。

## 16. 最终验收清单

### Docker 构建验收

1. `Dockerfile` 存在。
2. `.dockerignore` 存在。
3. `docker build -t naive-rag-api:dev .` 成功。
4. 镜像中不包含 `.env`。
5. 镜像中不包含 `.venv`。
6. 镜像中不包含本地 `data/chroma` 历史数据。

### Docker 运行验收

1. `docker compose up --build` 成功。
2. 容器状态为 running。
3. `docker compose logs` 没有 traceback。
4. `http://127.0.0.1:8000/docs` 可访问。
5. `/health` 返回 200。

### RAG 功能验收

1. 可以上传 `.md` 文件。
2. 上传后返回 `document_id`。
3. 上传后返回 `chunk_count > 0`。
4. `data/uploads` 有原始文件。
5. `data/metadata` 有文档记录。
6. `data/chroma` 有向量库文件。
7. `/api/v1/documents` 能看到文档。
8. `/api/v1/chat/query` 能回答文档相关问题。
9. 回答包含 sources。
10. 无答案问题不会明显编造。

### 持久化验收

1. 上传文档后停止容器。
2. 重新启动容器。
3. 不重新上传文档也能查询。
4. 删除容器不会删除宿主机 `data`。

### 文档验收

1. README 写清楚启动命令。
2. README 写清楚环境变量。
3. README 写清楚接口测试命令。
4. README 写清楚常见问题。
5. 周度总结写清楚本周收获和下一步计划。

## 17. 一份可直接复制的部署说明

你可以把下面内容放到 RAG 项目的 `README.md` 中。

```markdown
## Docker 启动

### 1. 准备环境变量

```powershell
Copy-Item .env.example .env
```

编辑 `.env`，填入：

```text
OPENAI_API_KEY=your_api_key_here
```

### 2. 启动服务

```powershell
docker compose up --build
```

后台启动：

```powershell
docker compose up --build -d
```

### 3. 访问 API 文档

```text
http://127.0.0.1:8000/docs
```

### 4. 健康检查

```powershell
curl http://127.0.0.1:8000/health
```

### 5. 上传文档

```powershell
curl -X POST "http://127.0.0.1:8000/api/v1/documents/upload" `
  -F "file=@.\samples\rag_notes.md"
```

### 6. 提问

```powershell
curl -X POST "http://127.0.0.1:8000/api/v1/chat/query" `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"RAG 的索引阶段包括哪些步骤？\",\"top_k\":4}"
```

### 7. 停止服务

```powershell
docker compose down
```
```

## 18. 今日你应该真正理解的东西

Docker 部署 RAG 项目，本质上是在划清三条边界：

1. 代码边界：`app/` 和依赖进入镜像。
2. 配置边界：`.env` 在运行时注入，不进入镜像。
3. 数据边界：`data/` 挂载到宿主机，不进入镜像。

只要这三条边界清楚，你的项目就会从“只在我电脑上能跑”变成“可以被别人复现运行”。

