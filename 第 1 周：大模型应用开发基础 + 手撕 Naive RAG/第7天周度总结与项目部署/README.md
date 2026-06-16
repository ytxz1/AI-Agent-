# 第7天：周度总结与项目部署

> 今日主题：把本周完成的 Naive RAG 项目用 Docker 打包，并在本地成功运行。
>
> 核心目标：你不仅要“知道 Docker 是什么”，还要能把 FastAPI + LangChain + Chroma 的 RAG 服务做成可复现、可迁移、可启动、可验证的容器化项目。

## 今日最终产出

1. 一个能构建成功的 `Dockerfile`。
2. 一个能一键启动服务的 `docker-compose.yml`。
3. 一个不把密钥和脏数据打进镜像的 `.dockerignore`。
4. 一个可复用的 `.env.example`。
5. 一个通过 Docker 启动的 RAG API 服务。
6. 一套完整的运行验收命令：健康检查、上传文档、查询文档、RAG 问答。
7. 一份本周复盘：完成了什么、理解了什么、踩了什么坑、下一步如何优化。

## 建议阅读顺序

1. [01-今日学习计划.md](01-今日学习计划.md)
2. [02-Docker打包与部署RAG项目详解.md](02-Docker打包与部署RAG项目详解.md)
3. [03-周度总结与项目验收清单.md](03-周度总结与项目验收清单.md)
4. [04-Docker部署代码实现逐行详解.md](04-Docker部署代码实现逐行详解.md)

## 本周项目上下文

前 6 天你已经完成了这些能力：

1. 第 1 天：FastAPI 基础，知道如何暴露 HTTP API。
2. 第 2 天：LangChain 核心概念，理解 model、prompt、chain、retriever。
3. 第 3 天：RAG Part 1，掌握文档加载与文本切分。
4. 第 4 天：RAG Part 2，掌握 embedding 与向量存储。
5. 第 5-6 天：手撕 Naive RAG 系统，完成上传、入库、检索、问答、sources 返回。

第 7 天要做的是把它从“我的电脑上能跑”推进到“任何装了 Docker 的机器都能按同一套命令跑”。

## 推荐项目目录

第 5-6 天项目建议位于：

```text
第5-6天手撕 Naive RAG 系统/
  naive-rag-api/
    app/
    data/
    tests/
    requirements.txt
    .env
    .env.example
    README.md
```

第 7 天完成后，项目目录应增加：

```text
第5-6天手撕 Naive RAG 系统/
  naive-rag-api/
    Dockerfile
    docker-compose.yml
    .dockerignore
    scripts/
      wait-for-api.ps1
    data/
      uploads/
      chroma/
      metadata/
```

## 今日核心链路

```mermaid
flowchart LR
    A["整理 RAG 项目依赖"] --> B["编写 .dockerignore"]
    B --> C["编写 Dockerfile"]
    C --> D["构建镜像"]
    D --> E["编写 docker-compose.yml"]
    E --> F["挂载 data 数据卷"]
    F --> G["注入 .env 环境变量"]
    G --> H["启动容器"]
    H --> I["健康检查"]
    I --> J["上传测试文档"]
    J --> K["发起 RAG 问答"]
    K --> L["复盘与验收"]
```

## 关键原则

1. 镜像里放代码和依赖，不放 API Key。
2. 容器里运行服务，数据目录用 volume 挂载出来。
3. 构建阶段要尽量稳定，运行阶段要尽量可配置。
4. Docker 启动成功不等于项目成功，必须完成 API 级验收。
5. RAG 项目部署验收一定要检查向量库是否持久化。

## 今日关键词

1. Docker
2. Dockerfile
3. Image
4. Container
5. docker compose
6. Environment Variables
7. Volume
8. Port Mapping
9. Health Check
10. Build Context
11. .dockerignore
12. requirements.txt
13. Uvicorn
14. FastAPI
15. Chroma Persist Directory
16. OPENAI_API_KEY
17. Reproducible Deployment
18. Smoke Test
19. Weekly Review
20. Production Readiness
