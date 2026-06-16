"""
04_lcel_rag_deepseek_optional.py

目标：
1. 把 03 的离线 RAG 换成真实 DeepSeek 模型
2. 仍然保留简单关键词检索器，避免今天被向量数据库分散注意力
3. 重点观察：只替换 model，LCEL 主链路几乎不用变

准备：
1. 复制 .env.example 为 .env
2. 填写 DEEPSEEK_API_KEY
3. 安装依赖：pip install -r requirements.txt

运行：
python code/04_lcel_rag_deepseek_optional.py
"""

import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI


load_dotenv()


KNOWLEDGE_BASE = [
    Document(
        page_content=(
            "LangChain 是一个用于开发大模型应用的框架。"
            "它提供模型输入输出、检索、链、智能体、记忆和回调等模块。"
        ),
        metadata={"source": "note-langchain"},
    ),
    Document(
        page_content=(
            "LCEL 是 LangChain Expression Language。"
            "它用 Runnable 作为统一抽象，可以通过管道符组合 prompt、model、parser、retriever 等组件。"
        ),
        metadata={"source": "note-lcel"},
    ),
    Document(
        page_content=(
            "RAG 是 Retrieval-Augmented Generation，即检索增强生成。"
            "它先检索相关上下文，再让模型基于上下文回答问题。"
        ),
        metadata={"source": "note-rag"},
    ),
]


def keyword_retriever(question: str):
    question_lower = question.lower()
    docs = []

    for doc in KNOWLEDGE_BASE:
        text = doc.page_content.lower()
        if (
            "lcel" in question_lower
            and "lcel" in text
            or "rag" in question_lower
            and "rag" in text
            or "langchain" in question_lower
            and "langchain" in text
        ):
            docs.append(doc)

    return docs or KNOWLEDGE_BASE[:1]


def format_docs(docs):
    return "\n\n".join(
        f"来源：{doc.metadata.get('source')}\n内容：{doc.page_content}"
        for doc in docs
    )


def main():
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        raise RuntimeError(
            "请先配置 DEEPSEEK_API_KEY。可以复制 .env.example 为 .env，然后填入你的 DeepSeek API Key。"
        )

    retriever = RunnableLambda(keyword_retriever)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一名严谨的 AI 应用开发导师。请只根据上下文回答。"),
            ("human", "上下文：\n{context}\n\n问题：{question}"),
        ]
    )

    model = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=deepseek_api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=0,
    )
    parser = StrOutputParser()

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | parser
    )

    question = "LCEL 为什么适合组织 LangChain 应用链路？"

    print("\n========== invoke 输出 ==========")
    print(rag_chain.invoke(question))

    print("\n========== stream 输出 ==========")
    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
