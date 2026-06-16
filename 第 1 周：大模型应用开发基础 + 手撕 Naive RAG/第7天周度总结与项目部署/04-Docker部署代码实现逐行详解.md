# Docker 部署代码实现逐行详解

> 本文专门解释已经写入 `第5-6天手撕 Naive RAG 系统/naive-rag-api` 项目的 Docker 部署代码。目标不是只会复制命令，而是看懂每个文件为什么存在、每一段代码解决什么问题。

## 1. 本次新增了哪些代码

项目目录新增：

```text
naive-rag-api/
  .dockerignore
  .env.docker.example
  Dockerfile
  docker-compose.yml
  docker-compose.dev.yml
  scripts/
    start-docker.ps1
    smoke-test.ps1
```

这些文件分工如下：

| 文件 | 作用 |
|---|---|
| `.dockerignore` | 控制哪些文件不要进入 Docker 构建上下文 |
| `.env.docker.example` | Docker 首次运行推荐配置，默认离线可跑 |
| `Dockerfile` | 描述如何把项目构建成镜像 |
| `docker-compose.yml` | 描述如何启动正式运行容器 |
| `docker-compose.dev.yml` | 描述如何启动开发热更新容器 |
| `scripts/start-docker.ps1` | 一键创建 `.env`、数据目录并启动 Docker |
| `scripts/smoke-test.ps1` | 一键验证 health、upload、documents、query 四个核心接口 |

## 2. 为什么这个项目适合 Docker 化

你的 RAG 项目已经具备清晰的工程边界：

```text
app/            业务代码
requirements.txt  Python 依赖
.env            运行配置和密钥
data/           上传文件、metadata、Chroma 向量库
samples/        测试样例
tests/          自动化测试
```

Docker 化要做的事情就是把这些边界进一步固定：

1. `app/` 和 `requirements.txt` 进入镜像。
2. `.env` 不进入镜像，运行时注入。
3. `data/` 不进入镜像，运行时挂载。
4. 容器内统一用 `/app` 作为工作目录。
5. 容器内 FastAPI 监听 `0.0.0.0:8000`。

## 3. `.dockerignore` 代码详解

文件位置：

```text
第5-6天手撕 Naive RAG 系统/naive-rag-api/.dockerignore
```

完整代码：

```dockerignore
.git
.gitignore

.venv
venv
env

__pycache__
*.py[cod]
*.pyo
*.pyd
.Python
.pytest_cache
.mypy_cache
.ruff_cache

.env
.env.*
!.env.example
!.env.docker.example

data
logs
tmp

*.sqlite
*.sqlite3
*.db

Dockerfile
docker-compose.override.yml
README.local.md
```

### 3.1 忽略 Git 文件

```dockerignore
.git
.gitignore
```

Docker 构建镜像不需要 Git 历史。`.git` 可能很大，复制进去会拖慢构建。

### 3.2 忽略虚拟环境

```dockerignore
.venv
venv
env
```

你的本地 `.venv` 是 Windows 环境下的虚拟环境，Docker 容器里是 Linux 环境。把 Windows 虚拟环境复制进 Linux 容器不仅没用，还容易污染镜像。

正确做法是在 Dockerfile 里重新执行：

```dockerfile
python -m pip install -r requirements.txt
```

### 3.3 忽略 Python 缓存

```dockerignore
__pycache__
*.py[cod]
*.pyo
*.pyd
.Python
.pytest_cache
.mypy_cache
.ruff_cache
```

这些都是运行、测试、静态检查留下的缓存，不是业务代码。忽略它们可以让镜像更干净。

### 3.4 忽略真实环境变量

```dockerignore
.env
.env.*
!.env.example
!.env.docker.example
```

这里非常关键。

`.env` 里可能有：

```text
OPENAI_API_KEY=...
DEEPSEEK_API_KEY=...
```

它不能进入镜像。镜像一旦被分享、上传、复制，密钥就可能泄漏。

但是示例文件可以保留：

```dockerignore
!.env.example
!.env.docker.example
```

`!` 表示例外。也就是说：

1. 忽略真实 `.env`。
2. 保留 `.env.example`。
3. 保留 `.env.docker.example`。

### 3.5 忽略数据目录

```dockerignore
data
logs
tmp
```

RAG 项目的 `data` 目录通常包含：

1. 用户上传原文。
2. Chroma 向量库。
3. metadata registry。

这些不是镜像的一部分，而是运行时数据。

正确做法是通过 compose 挂载：

```yaml
volumes:
  - ./data:/app/data
```

### 3.6 忽略数据库文件

```dockerignore
*.sqlite
*.sqlite3
*.db
```

Chroma 本地持久化会生成类似：

```text
data/chroma/chroma.sqlite3
```

它不应该进入镜像。

## 4. `.env.docker.example` 代码详解

文件位置：

```text
第5-6天手撕 Naive RAG 系统/naive-rag-api/.env.docker.example
```

完整代码：

```text
APP_NAME=Naive RAG API
APP_ENV=docker
API_PREFIX=/api/v1

UPLOAD_DIR=data/uploads
CHROMA_DIR=data/chroma
METADATA_DIR=data/metadata
CHROMA_COLLECTION=naive_rag

MAX_FILE_SIZE_MB=10
CHUNK_SIZE=800
CHUNK_OVERLAP=120
DEFAULT_TOP_K=4

EMBEDDING_PROVIDER=hash
CHAT_PROVIDER=mock

DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

OPENAI_API_KEY=
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini
```

### 4.1 为什么单独做 Docker 示例配置

你已有 `.env.example`，但它默认更偏真实模型配置：

```text
CHAT_PROVIDER=deepseek
```

如果没有 `DEEPSEEK_API_KEY`，容器启动或问答时就会失败。

所以 Docker 学习阶段新增 `.env.docker.example`，默认：

```text
EMBEDDING_PROVIDER=hash
CHAT_PROVIDER=mock
```

这样不需要任何 API key，就能跑通：

1. FastAPI 启动。
2. 文档上传。
3. 文本切分。
4. hash embedding。
5. Chroma 入库。
6. 检索。
7. mock 回答。
8. sources 返回。

这非常适合部署验收。部署先证明链路能跑，再切换真实模型。

### 4.2 路径为什么仍然写相对路径

```text
UPLOAD_DIR=data/uploads
CHROMA_DIR=data/chroma
METADATA_DIR=data/metadata
```

容器中工作目录是：

```text
/app
```

所以相对路径会变成：

```text
/app/data/uploads
/app/data/chroma
/app/data/metadata
```

而 compose 又会把本机目录挂载进去：

```yaml
volumes:
  - ./data:/app/data
```

最终效果：

```text
本机 naive-rag-api/data  <->  容器 /app/data
```

## 5. `Dockerfile` 代码详解

文件位置：

```text
第5-6天手撕 Naive RAG 系统/naive-rag-api/Dockerfile
```

完整代码：

```dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY app ./app
COPY samples ./samples
COPY .env.example ./.env.example
COPY .env.docker.example ./.env.docker.example

RUN mkdir -p data/uploads data/chroma data/metadata

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 5.1 基础镜像

```dockerfile
FROM python:3.12-slim
```

表示镜像基于 Python 3.12 的精简 Linux 环境。

为什么不用你本机的 Python 3.14？

1. Python 3.14 太新，很多依赖可能还没完全适配。
2. Docker 里的部署环境应该选择稳定版本。
3. Python 3.12 对 FastAPI、LangChain、Chroma 更稳妥。

### 5.2 Python 运行环境变量

```dockerfile
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
```

含义：

| 变量 | 作用 |
|---|---|
| `PYTHONDONTWRITEBYTECODE=1` | 不生成 `.pyc` 文件 |
| `PYTHONUNBUFFERED=1` | 日志实时输出 |
| `PIP_NO_CACHE_DIR=1` | pip 不保存安装缓存 |

对 Docker 来说，日志实时输出很重要。否则你看 `docker logs` 时，可能日志不及时刷新。

### 5.3 设置工作目录

```dockerfile
WORKDIR /app
```

后续命令都在 `/app` 下执行。

这会影响你的配置：

```python
upload_dir: Path = Path("data/uploads")
```

容器里会解析成：

```text
/app/data/uploads
```

### 5.4 安装 curl

```dockerfile
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
```

安装 `curl` 是为了健康检查：

```dockerfile
HEALTHCHECK ... CMD curl -f http://127.0.0.1:8000/health || exit 1
```

`rm -rf /var/lib/apt/lists/*` 用来清理 apt 缓存，减小镜像体积。

### 5.5 先复制依赖文件

```dockerfile
COPY requirements.txt .
```

只复制 `requirements.txt`，暂时不复制全部代码。

原因是 Docker 有构建缓存。只要 `requirements.txt` 不变，下面这步就能复用缓存：

```dockerfile
RUN python -m pip install -r requirements.txt
```

这样你改业务代码时，不需要每次重新安装依赖。

### 5.6 安装 Python 依赖

```dockerfile
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt
```

你的项目依赖来自：

```text
requirements.txt
```

其中关键包包括：

| 依赖 | 作用 |
|---|---|
| `fastapi` | HTTP API 框架 |
| `uvicorn[standard]` | ASGI 服务 |
| `pydantic-settings` | `.env` 配置读取 |
| `python-multipart` | 支持文件上传 |
| `langchain` | RAG 链路组织 |
| `langchain-chroma` | Chroma 向量库集成 |
| `chromadb` | 本地向量数据库 |
| `langchain-openai` | OpenAI / DeepSeek 兼容接口 |

### 5.7 复制业务代码

```dockerfile
COPY app ./app
COPY samples ./samples
COPY .env.example ./.env.example
COPY .env.docker.example ./.env.docker.example
```

复制进入镜像的只有：

1. `app`：服务代码。
2. `samples`：样例文档。
3. 环境变量示例文件。

不会复制：

1. `.env`。
2. `.venv`。
3. `data`。
4. `__pycache__`。

这些由 `.dockerignore` 控制。

### 5.8 创建数据目录

```dockerfile
RUN mkdir -p data/uploads data/chroma data/metadata
```

即使没有挂载 volume，容器内也有基础目录，避免启动时报目录不存在。

实际运行时，compose 会用本地 `./data` 覆盖容器里的 `/app/data`：

```yaml
volumes:
  - ./data:/app/data
```

### 5.9 声明端口

```dockerfile
EXPOSE 8000
```

这只是文档式声明，告诉别人这个镜像里的服务默认使用 8000 端口。

真正让宿主机能访问容器的是 compose 里的：

```yaml
ports:
  - "8000:8000"
```

### 5.10 健康检查

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1
```

含义：

| 参数 | 作用 |
|---|---|
| `interval=30s` | 每 30 秒检查一次 |
| `timeout=10s` | 10 秒内无响应算失败 |
| `start-period=20s` | 启动后先等 20 秒再判断健康 |
| `retries=3` | 连续 3 次失败才标记 unhealthy |

健康检查访问：

```text
http://127.0.0.1:8000/health
```

对应代码是：

```python
@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(...)
```

### 5.11 容器启动命令

```dockerfile
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

这里最重要的是：

```text
--host 0.0.0.0
```

如果写成：

```text
--host 127.0.0.1
```

服务只监听容器内部回环地址，宿主机可能访问不到。

容器部署中，FastAPI 必须监听：

```text
0.0.0.0
```

## 6. `docker-compose.yml` 代码详解

文件位置：

```text
第5-6天手撕 Naive RAG 系统/naive-rag-api/docker-compose.yml
```

完整代码：

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
      test: ["CMD", "curl", "-f", "http://127.0.0.1:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 20s
```

### 6.1 定义服务

```yaml
services:
  rag-api:
```

`rag-api` 是服务名。后续日志、状态、执行命令都会用它：

```powershell
docker compose logs rag-api
docker compose exec rag-api sh
```

### 6.2 构建镜像

```yaml
build:
  context: .
  dockerfile: Dockerfile
```

含义：

1. 构建上下文是当前目录。
2. 使用当前目录下的 `Dockerfile`。

构建上下文会受 `.dockerignore` 控制。

### 6.3 指定镜像名

```yaml
image: naive-rag-api:dev
```

构建完成后的镜像叫：

```text
naive-rag-api:dev
```

查看：

```powershell
docker images
```

### 6.4 指定容器名

```yaml
container_name: naive-rag-api
```

容器名固定后，可以直接看日志：

```powershell
docker logs naive-rag-api
```

### 6.5 注入环境变量

```yaml
env_file:
  - .env
```

容器启动时读取 `.env`。

第一次运行推荐：

```powershell
copy .env.docker.example .env
```

如果你想用 DeepSeek，改 `.env`：

```text
EMBEDDING_PROVIDER=hash
CHAT_PROVIDER=deepseek
DEEPSEEK_API_KEY=你的 key
```

如果你想完全离线验收，保持：

```text
EMBEDDING_PROVIDER=hash
CHAT_PROVIDER=mock
```

### 6.6 端口映射

```yaml
ports:
  - "8000:8000"
```

格式是：

```text
宿主机端口:容器端口
```

所以：

```text
http://127.0.0.1:8000
```

会访问容器内的：

```text
http://容器:8000
```

如果本机 8000 被占用，改成：

```yaml
ports:
  - "8001:8000"
```

然后访问：

```text
http://127.0.0.1:8001/docs
```

### 6.7 数据持久化

```yaml
volumes:
  - ./data:/app/data
```

这是 RAG Docker 部署里最重要的一行之一。

它让本机：

```text
naive-rag-api/data
```

映射到容器：

```text
/app/data
```

你的代码写入：

```text
/app/data/uploads
/app/data/chroma
/app/data/metadata
```

实际会保存在本机：

```text
naive-rag-api/data/uploads
naive-rag-api/data/chroma
naive-rag-api/data/metadata
```

这样即使容器删掉，数据仍然在本机。

### 6.8 自动重启策略

```yaml
restart: unless-stopped
```

含义：

1. 容器异常退出时自动重启。
2. 如果你手动停止它，就不会反复拉起。

学习项目可以使用这个策略；生产环境还要配合日志、监控和告警。

## 7. `docker-compose.dev.yml` 代码详解

文件位置：

```text
第5-6天手撕 Naive RAG 系统/naive-rag-api/docker-compose.dev.yml
```

完整代码：

```yaml
services:
  rag-api:
    build:
      context: .
      dockerfile: Dockerfile
    image: naive-rag-api:dev
    container_name: naive-rag-api-dev
    env_file:
      - .env
    ports:
      - "8000:8000"
    volumes:
      - ./app:/app/app
      - ./samples:/app/samples
      - ./data:/app/data
    command: python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

它和正式 compose 的区别：

1. 挂载 `./app:/app/app`，本机改代码，容器内立即看到。
2. 使用 `--reload`，代码变化后自动重启服务。
3. 容器名叫 `naive-rag-api-dev`，避免和正式容器混淆。

启动：

```powershell
docker compose -f docker-compose.dev.yml up --build
```

什么时候用它？

1. 你正在改 FastAPI 接口。
2. 你正在调 RAG service。
3. 你想在 Docker 环境中开发。

什么时候不用它？

1. 写验收文档。
2. 模拟部署环境。
3. 给别人演示稳定版本。

## 8. `scripts/start-docker.ps1` 代码详解

文件位置：

```text
第5-6天手撕 Naive RAG 系统/naive-rag-api/scripts/start-docker.ps1
```

完整代码：

```powershell
param(
    [switch] $Dev,
    [switch] $Detached
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path ".env")) {
    Copy-Item ".env.docker.example" ".env"
    Write-Host "Created .env from .env.docker.example."
    Write-Host "Default Docker mode uses EMBEDDING_PROVIDER=hash and CHAT_PROVIDER=mock."
}

New-Item -ItemType Directory -Force "data", "data\uploads", "data\chroma", "data\metadata" | Out-Null

$composeFiles = @("docker-compose.yml")
if ($Dev) {
    $composeFiles = @("docker-compose.dev.yml")
}

$argsList = @()
foreach ($file in $composeFiles) {
    $argsList += @("-f", $file)
}

$argsList += @("up", "--build")
if ($Detached) {
    $argsList += "-d"
}

docker compose @argsList
```

### 8.1 参数

```powershell
param(
    [switch] $Dev,
    [switch] $Detached
)
```

支持三种用法：

```powershell
.\scripts\start-docker.ps1
.\scripts\start-docker.ps1 -Detached
.\scripts\start-docker.ps1 -Dev
```

含义：

| 命令 | 效果 |
|---|---|
| 不带参数 | 前台启动正式 compose |
| `-Detached` | 后台启动正式 compose |
| `-Dev` | 前台启动开发热更新 compose |

### 8.2 出错立即停止

```powershell
$ErrorActionPreference = "Stop"
```

PowerShell 默认有些错误不会中断脚本。这里改成遇到错误就停，避免后续命令在错误状态下继续执行。

### 8.3 自动进入项目根目录

```powershell
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
```

`$PSScriptRoot` 是脚本所在目录：

```text
naive-rag-api/scripts
```

它的父目录就是项目根目录：

```text
naive-rag-api
```

这样你无论从哪里运行脚本，它都会先切到正确目录。

### 8.4 自动创建 `.env`

```powershell
if (-not (Test-Path ".env")) {
    Copy-Item ".env.docker.example" ".env"
}
```

如果项目没有 `.env`，脚本会自动从 `.env.docker.example` 复制。

这让第一次启动更顺滑：

```powershell
.\scripts\start-docker.ps1
```

不需要你先手动执行：

```powershell
copy .env.docker.example .env
```

### 8.5 自动创建数据目录

```powershell
New-Item -ItemType Directory -Force "data", "data\uploads", "data\chroma", "data\metadata" | Out-Null
```

提前创建：

1. `data/uploads`
2. `data/chroma`
3. `data/metadata`

避免 volume 挂载时目录不存在或结构混乱。

### 8.6 动态选择 compose 文件

```powershell
$composeFiles = @("docker-compose.yml")
if ($Dev) {
    $composeFiles = @("docker-compose.dev.yml")
}
```

如果你传 `-Dev`，就使用开发版 compose。

否则使用正式版 compose。

### 8.7 组装 docker compose 参数

```powershell
$argsList += @("up", "--build")
if ($Detached) {
    $argsList += "-d"
}

docker compose @argsList
```

最终执行的命令可能是：

```powershell
docker compose -f docker-compose.yml up --build
```

或：

```powershell
docker compose -f docker-compose.yml up --build -d
```

或：

```powershell
docker compose -f docker-compose.dev.yml up --build
```

## 9. `scripts/smoke-test.ps1` 代码详解

文件位置：

```text
第5-6天手撕 Naive RAG 系统/naive-rag-api/scripts/smoke-test.ps1
```

完整代码：

```powershell
param(
    [string] $BaseUrl = "http://127.0.0.1:8000",
    [string] $SampleFile = "samples\rag_notes.md"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path $SampleFile)) {
    throw "Sample file not found: $SampleFile"
}

Write-Host "1. Checking health..."
$health = Invoke-RestMethod -Method Get -Uri "$BaseUrl/health"
$health | ConvertTo-Json -Depth 10

Write-Host "2. Uploading sample document..."
$uploadOutput = curl.exe -sS -X POST "$BaseUrl/api/v1/documents/upload" -F "file=@$SampleFile"
Write-Host $uploadOutput

Write-Host "3. Listing documents..."
$documents = Invoke-RestMethod -Method Get -Uri "$BaseUrl/api/v1/documents"
$documents | ConvertTo-Json -Depth 10

Write-Host "4. Asking a RAG question..."
$body = @{
    question = "RAG 的索引阶段包括哪些步骤？"
    top_k = 4
} | ConvertTo-Json -Depth 10

$answer = Invoke-RestMethod `
    -Method Post `
    -Uri "$BaseUrl/api/v1/chat/query" `
    -ContentType "application/json; charset=utf-8" `
    -Body $body

$answer | ConvertTo-Json -Depth 10

Write-Host "Smoke test finished."
```

### 9.1 为什么叫 smoke test

smoke test 是冒烟测试，意思是先验证系统有没有明显冒烟坏掉。

这里验证四个关键接口：

1. `/health`
2. `/api/v1/documents/upload`
3. `/api/v1/documents`
4. `/api/v1/chat/query`

如果这四步都成功，说明 Docker 容器中的 RAG 主链路基本跑通。

### 9.2 健康检查

```powershell
$health = Invoke-RestMethod -Method Get -Uri "$BaseUrl/health"
```

期望返回：

```json
{
  "status": "ok",
  "app": "Naive RAG API",
  "environment": "docker",
  "embedding_provider": "hash",
  "chat_provider": "mock",
  "vector_store_dir": "data/chroma"
}
```

### 9.3 上传文件为什么用 `curl.exe`

```powershell
$uploadOutput = curl.exe -sS -X POST "$BaseUrl/api/v1/documents/upload" -F "file=@$SampleFile"
```

Windows PowerShell 里 `curl` 常常是 `Invoke-WebRequest` 的别名。为了明确使用真正的 curl，脚本写成：

```powershell
curl.exe
```

文件上传使用 multipart form：

```text
-F "file=@samples\rag_notes.md"
```

对应 FastAPI 代码：

```python
@router.post("/upload", response_model=DocumentUploadResponse)
def upload_document(file: UploadFile = File(...)) -> DocumentUploadResponse:
    return document_service.upload_document(file)
```

### 9.4 文档列表

```powershell
$documents = Invoke-RestMethod -Method Get -Uri "$BaseUrl/api/v1/documents"
```

这一步验证 metadata registry 是否写入成功。

你的项目会把文档信息写到：

```text
data/metadata/documents.json
```

### 9.5 RAG 问答

```powershell
$body = @{
    question = "RAG 的索引阶段包括哪些步骤？"
    top_k = 4
} | ConvertTo-Json -Depth 10
```

然后 POST 到：

```text
/api/v1/chat/query
```

这一步会触发：

```text
问题 -> embedding -> Chroma 检索 -> chat model -> answer + sources
```

如果当前 `.env` 是：

```text
CHAT_PROVIDER=mock
```

回答会明确提示是 mock 模式，但仍会返回检索片段。这足够验证部署主链路。

## 10. 最小运行步骤

进入项目目录：

```powershell
cd "D:\vscode项目\AI Agent 开发工程师学习路线图（工程落地版）\第 1 周：大模型应用开发基础 + 手撕 Naive RAG\第5-6天手撕 Naive RAG 系统\naive-rag-api"
```

准备 Docker 环境变量：

```powershell
copy .env.docker.example .env
```

启动：

```powershell
docker compose up --build
```

另开一个终端执行验收：

```powershell
.\scripts\smoke-test.ps1
```

停止：

```powershell
docker compose down
```

## 11. 一键运行步骤

如果你不想手动复制 `.env` 和创建数据目录，直接执行：

```powershell
.\scripts\start-docker.ps1
```

后台运行：

```powershell
.\scripts\start-docker.ps1 -Detached
```

冒烟测试：

```powershell
.\scripts\smoke-test.ps1
```

开发模式：

```powershell
.\scripts\start-docker.ps1 -Dev
```

## 12. 切换到 DeepSeek

编辑 `.env`：

```text
EMBEDDING_PROVIDER=hash
CHAT_PROVIDER=deepseek
DEEPSEEK_API_KEY=你的 DeepSeek API key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

重新启动：

```powershell
docker compose down
docker compose up --build
```

说明：

1. `hash` embedding 仍然离线。
2. DeepSeek 只负责最终回答生成。
3. 如果没有 key，会报 `DEEPSEEK_API_KEY is required when CHAT_PROVIDER=deepseek`。

## 13. 切换到 OpenAI

编辑 `.env`：

```text
EMBEDDING_PROVIDER=openai
CHAT_PROVIDER=openai
OPENAI_API_KEY=你的 OpenAI API key
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini
```

重新启动：

```powershell
docker compose down
docker compose up --build
```

说明：

1. 上传文档时会调用 OpenAI embedding。
2. 提问时会调用 OpenAI chat model。
3. 这会产生真实网络请求和费用。

## 14. 验证数据持久化

执行：

```powershell
docker compose up --build -d
.\scripts\smoke-test.ps1
docker compose down
docker compose up -d
```

重启后不要重新上传文档，直接问：

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/chat/query" -H "Content-Type: application/json" -d "{\"question\":\"metadata 有什么作用？\",\"top_k\":4}"
```

如果还能回答并返回 sources，说明：

1. `data/chroma` 持久化成功。
2. `data/metadata` 持久化成功。
3. compose 的 volume 配置有效。

## 15. 常见错误和定位方式

### 15.1 容器启动后访问不到

查看状态：

```powershell
docker compose ps
```

查看日志：

```powershell
docker compose logs rag-api
```

重点检查：

1. 是否监听 `0.0.0.0`。
2. 是否端口映射 `8000:8000`。
3. 是否 Python import 报错。
4. 是否 `.env` 中模型 provider 配错。

### 15.2 没有 API key

如果你看到：

```text
DEEPSEEK_API_KEY is required when CHAT_PROVIDER=deepseek
```

有两个选择。

离线学习：

```text
CHAT_PROVIDER=mock
```

真实模型：

```text
CHAT_PROVIDER=deepseek
DEEPSEEK_API_KEY=你的 key
```

### 15.3 上传文件失败

检查依赖：

```text
python-multipart
```

它已经在 `requirements.txt` 里：

```text
python-multipart>=0.0.9
```

如果容器里仍然报错，重新构建：

```powershell
docker compose build --no-cache
docker compose up
```

### 15.4 重启后数据丢失

检查 compose：

```yaml
volumes:
  - ./data:/app/data
```

检查 `.env`：

```text
CHROMA_DIR=data/chroma
METADATA_DIR=data/metadata
UPLOAD_DIR=data/uploads
```

检查本机目录：

```powershell
Get-ChildItem data -Recurse
```

## 16. 你现在应该能说清楚的结论

这套 Docker 代码完成了三件事：

1. 用 `Dockerfile` 固化运行环境。
2. 用 `docker-compose.yml` 固化启动方式。
3. 用 volume 和 `.env` 把数据、配置从镜像中分离出来。

对 RAG 项目来说，真正关键的是：

1. 文档和向量库不能打进镜像。
2. API key 不能打进镜像。
3. Chroma 目录必须持久化。
4. 部署验收必须跑到 `/api/v1/chat/query`，只看 `/health` 不够。

