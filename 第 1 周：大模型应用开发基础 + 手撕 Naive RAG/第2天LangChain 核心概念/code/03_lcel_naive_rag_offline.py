"""
03_lcel_naive_rag_offline.py

目标：
1. 用 LCEL 实现一个“离线可运行”的 Naive RAG
2. 不依赖 DeepSeek API，不依赖向量数据库
3. 重点理解 RAG 数据流：question -> retriever -> context -> prompt -> model -> parser

运行：
python code/03_lcel_naive_rag_offline.py
"""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


KNOWLEDGE_BASE = [
    Document(
        page_content=(
            "LangChain 是一个用于开发大模型应用的框架。"
            "它把 Prompt、Model、OutputParser、Retriever、Tool、Memory 等能力抽象成组件。"
        ),
        metadata={"source": "note-langchain", "title": "LangChain 总览"},
    ),
    Document(
        page_content=(
            "LCEL 是 LangChain Expression Language。"
            "它可以用管道符把 Runnable 组件组合起来，例如 prompt | model | parser。"
            "LCEL 支持 invoke、batch、stream、并行和异步调用。"
        ),
        metadata={"source": "note-lcel", "title": "LCEL 说明"},
    ),
    Document(
        page_content=(
            "Retrieval 是检索增强生成的基础。"
            "RAG 会先根据用户问题检索相关文档，再把文档内容作为上下文交给模型生成答案。"
        ),
        metadata={"source": "note-rag", "title": "RAG 基础"},
    ),
    Document(
        page_content=(
            "Agent 适合流程不固定、需要模型选择工具的任务。"
            "普通 Chain 更适合流程固定、输入输出明确的任务。"
        ),
        metadata={"source": "note-agent", "title": "Agent 和 Chain"},
    ),
    Document(
        page_content=(
            "Memory 用于保存多轮对话历史。"
            "它的本质是应用保存历史消息，并在后续请求中重新提供给模型。"
        ),
        metadata={"source": "note-memory", "title": "Memory 说明"},
    ),
]


def keyword_retriever(question: str):
    """
    一个极简检索器。

    真实 RAG 通常使用 embedding + vector store。
    这里为了让代码零成本可运行，使用关键词命中数量做排序。
    """
    keywords = [
        "LangChain",
        "LCEL",
        "Runnable",
        "invoke",
        "batch",
        "stream",
        "Retrieval",
        "RAG",
        "Agent",
        "Chain",
        "Memory",
        "Prompt",
        "Model",
        "Parser",
    ]

    normalized_question = question.lower()
    active_keywords = [
        keyword.lower()
        for keyword in keywords
        if keyword.lower() in normalized_question
    ]

    if not active_keywords:
        return KNOWLEDGE_BASE[:2]

    scored_docs = []

    for doc in KNOWLEDGE_BASE:
        text = (doc.page_content + " " + doc.metadata.get("title", "")).lower()
        score = 0

        for keyword in active_keywords:
            if keyword in text:
                score += 2

        if score > 0:
            scored_docs.append((score, doc))

    scored_docs.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in scored_docs[:3]]


def format_docs(docs):
    if not docs:
        return "没有检索到相关上下文。"

    formatted = []
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        title = doc.metadata.get("title", "unknown")
        formatted.append(
            f"[文档 {index}]\n"
            f"标题：{title}\n"
            f"来源：{source}\n"
            f"内容：{doc.page_content}"
        )

    return "\n\n".join(formatted)


def fake_rag_model(prompt_value) -> str:
    """
    一个模拟模型。

    它不会真正理解语言，只是根据 prompt 里的上下文生成一个可观察的回答。
    这样你可以专注观察 LCEL 和 RAG 的数据流。
    """
    messages = prompt_value.to_messages()
    full_prompt = "\n".join(message.content for message in messages)
    question = full_prompt.split("问题：", maxsplit=1)[-1].split("请给出", maxsplit=1)[0]

    if "LCEL" in question:
        answer = (
            "LCEL 是 LangChain 的表达式语言，用于把 Runnable 组件组合成链路。"
            "典型写法是 prompt | model | parser。"
            "它还统一支持 invoke、batch、stream 等调用方式。"
        )
    elif "RAG" in question or "检索" in question:
        answer = (
            "RAG 是检索增强生成。它会先根据问题检索相关文档，"
            "再把检索结果作为上下文交给模型生成答案。"
        )
    elif "Agent" in question or "Chain" in question:
        answer = (
            "Agent 适合任务流程不固定、需要模型选择工具的场景。"
            "如果流程固定，优先使用 Chain 或 LCEL。"
        )
    elif "Memory" in question or "记忆" in question:
        answer = (
            "Memory 用于保存多轮对话历史。它的本质不是模型真的记住了，"
            "而是应用把历史消息保存下来，并在后续请求中重新放回上下文。"
        )
    else:
        answer = "我会优先根据提供的上下文回答；如果上下文不足，需要说明不知道。"

    return answer + "\n\n以上为离线模拟模型输出，用于学习 LCEL 数据流。"


def build_rag_chain():
    retriever = RunnableLambda(keyword_retriever)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一名严谨的 AI 应用开发导师。请只根据上下文回答问题。",
            ),
            (
                "human",
                "上下文：\n{context}\n\n问题：{question}\n\n请给出清晰、分点的回答。",
            ),
        ]
    )

    model = RunnableLambda(fake_rag_model)
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

    return rag_chain


def debug_steps(question: str):
    print("\n========== 1. 用户问题 ==========")
    print(question)

    print("\n========== 2. 检索器返回 Document 列表 ==========")
    docs = keyword_retriever(question)
    for doc in docs:
        print(doc)

    print("\n========== 3. 格式化后的 context ==========")
    context = format_docs(docs)
    print(context)

    print("\n========== 4. prompt 渲染结果 ==========")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一名严谨的 AI 应用开发导师。请只根据上下文回答问题。"),
            ("human", "上下文：\n{context}\n\n问题：{question}\n\n请给出清晰、分点的回答。"),
        ]
    )
    prompt_result = prompt.invoke({"context": context, "question": question})
    print(prompt_result)


def main():
    question = "LCEL 在 LangChain 里有什么作用？"

    debug_steps(question)

    rag_chain = build_rag_chain()

    print("\n========== 5. 完整 RAG Chain 输出 ==========")
    answer = rag_chain.invoke(question)
    print(answer)

    print("\n========== 6. batch 批量问答 ==========")
    questions = [
        "什么是 RAG？",
        "Agent 和 Chain 有什么区别？",
        "Memory 解决什么问题？",
    ]
    answers = rag_chain.batch(questions)
    for q, a in zip(questions, answers):
        print(f"\n问题：{q}")
        print(f"回答：{a}")


if __name__ == "__main__":
    main()
