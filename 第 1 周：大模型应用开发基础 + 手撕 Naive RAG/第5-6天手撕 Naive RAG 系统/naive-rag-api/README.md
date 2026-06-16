# Naive RAG API

一个用于学习的端到端文档问答 API：FastAPI 负责 HTTP 服务，LangChain 负责文档结构、文本切分、Embedding 接口和 Chroma 向量库集成。

## 快速开始

先进入项目目录：

```powershell
cd "D:\vscode项目\AI Agent 开发工程师学习路线图（工程落地版）\第 1 周：大模型应用开发基础 + 手撕 Naive RAG\第5-6天手撕 Naive RAG 系统\naive-rag-api"
```

创建并使用项目自己的虚拟环境：

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```

如果使用 DeepSeek，请先在 `.env` 中填写：

```text
DEEPSEEK_API_KEY=你的 DeepSeek API key
```

启动服务：

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

打开：

```text
http://127.0.0.1:8000/docs
```

## 正确运行方式

推荐始终使用项目虚拟环境里的 Python：

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

也可以先激活虚拟环境，再运行：

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload
```

如果你想直接运行入口文件，也可以：

```powershell
.\.venv\Scripts\python.exe app\main.py
```

不推荐使用全局 Python 直接运行：

```powershell
& "C:\Program Files\Python314\python.exe" "...\naive-rag-api\app\main.py"
```

原因是全局 Python 通常没有安装本项目依赖，容易出现：

```text
ModuleNotFoundError: No module named 'langchain_chroma'
```

如果已经看到这个错误，说明依赖没有安装到当前 Python 环境。请回到项目目录后执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 默认运行方式

默认配置使用 DeepSeek 聊天模型 + 本地 hash embedding：

```text
EMBEDDING_PROVIDER=hash
CHAT_PROVIDER=deepseek
DEEPSEEK_API_KEY=你的 DeepSeek API key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

这样可以用 DeepSeek 生成最终答案，同时继续用本地 hash embedding 完成文档向量化。DeepSeek API 兼容 OpenAI Chat API，OpenAI-compatible base URL 通常是 `https://api.deepseek.com`。如果你的 DeepSeek 控制台显示了更新的模型名，直接修改 `DEEPSEEK_MODEL` 即可。

如果你暂时没有 DeepSeek API key，可以改成离线 mock：

```text
EMBEDDING_PROVIDER=hash
CHAT_PROVIDER=mock
```

mock 适合学习 API 主链路。回答质量不会像真实大模型，但可以验证：

1. 上传文档。
2. 文本切分。
3. 写入 Chroma。
4. 相似度检索。
5. 返回答案和来源。

如果要使用 OpenAI 聊天模型：

```text
EMBEDDING_PROVIDER=openai
CHAT_PROVIDER=openai
OPENAI_API_KEY=你的 key
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini
```

注意：DeepSeek 在本项目里只负责 chat/generation。Embedding 仍使用 `hash` 离线实现，或者你可以另接 OpenAI/其他 embedding 服务。

## API

1. `GET /health`
2. `POST /api/v1/documents/upload`
3. `GET /api/v1/documents`
4. `GET /api/v1/documents/{document_id}`
5. `DELETE /api/v1/documents/{document_id}`
6. `POST /api/v1/chat/query`

## curl 示例

```powershell
curl http://127.0.0.1:8000/health
```

```powershell
curl -X POST "http://127.0.0.1:8000/api/v1/documents/upload" `
  -F "file=@.\samples\rag_notes.md"
```

```powershell
curl -X POST "http://127.0.0.1:8000/api/v1/chat/query" `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"RAG 的索引阶段包括哪些步骤？\",\"top_k\":4}"
```

## Docker 运行

第一次用 Docker 跑项目时，推荐先使用离线配置，验证完整链路：

```powershell
copy .env.docker.example .env
```

离线配置默认使用：

```text
EMBEDDING_PROVIDER=hash
CHAT_PROVIDER=mock
```

这样不需要 API key，也可以验证服务启动、文档上传、文本切分、Chroma 入库、检索和问答响应。

启动：

```powershell
docker compose up --build
```

后台启动：

```powershell
docker compose up --build -d
```

执行冒烟测试：

```powershell
.\scripts\smoke-test.ps1
```

停止：

```powershell
docker compose down
```

开发热更新模式：

```powershell
docker compose -f docker-compose.dev.yml up --build
```

也可以使用封装脚本：

```powershell
.\scripts\start-docker.ps1
.\scripts\start-docker.ps1 -Detached
.\scripts\start-docker.ps1 -Dev
```

如果要使用 DeepSeek，把 `.env` 改为：

```text
EMBEDDING_PROVIDER=hash
CHAT_PROVIDER=deepseek
DEEPSEEK_API_KEY=你的 DeepSeek API key
```

如果要使用 OpenAI，把 `.env` 改为：

```text
EMBEDDING_PROVIDER=openai
CHAT_PROVIDER=openai
OPENAI_API_KEY=你的 OpenAI API key
```
