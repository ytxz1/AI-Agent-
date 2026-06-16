"""
02_lcel_parallel_and_lambda.py

目标：
1. 理解 RunnableLambda：把普通 Python 函数变成 LCEL 节点
2. 理解 RunnableParallel：让多个链路并行处理同一个输入
3. 理解 batch 和 stream 的基本用法

运行：
python code/02_lcel_parallel_and_lambda.py
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel


def clean_text(text: str) -> str:
    return " ".join(text.strip().split())


def fake_summary_model(prompt_value) -> str:
    human_message = prompt_value.to_messages()[-1].content
    return f"模拟摘要：这段内容主要在讲 {human_message[:30]}..."


def fake_keywords_model(prompt_value) -> str:
    human_message = prompt_value.to_messages()[-1].content
    words = [
        "LangChain",
        "LCEL",
        "Runnable",
        "Prompt",
        "Parser",
    ]
    return "，".join(words) + f"\n输入片段：{human_message[:20]}..."


def fake_question_model(prompt_value) -> str:
    human_message = prompt_value.to_messages()[-1].content
    return f"模拟思考题：如果要把这段内容做成 RAG，你会如何切分文档？\n原始任务：{human_message[:30]}..."


def main():
    cleaner = RunnableLambda(clean_text)

    summary_prompt = ChatPromptTemplate.from_template("请总结下面内容：{text}")
    keywords_prompt = ChatPromptTemplate.from_template("请提取下面内容的关键词：{text}")
    question_prompt = ChatPromptTemplate.from_template("请基于下面内容生成一个思考题：{text}")

    parser = StrOutputParser()

    summary_chain = summary_prompt | RunnableLambda(fake_summary_model) | parser
    keywords_chain = keywords_prompt | RunnableLambda(fake_keywords_model) | parser
    question_chain = question_prompt | RunnableLambda(fake_question_model) | parser

    analysis_chain = cleaner | RunnableParallel(
        {
            "summary": summary_chain,
            "keywords": keywords_chain,
            "question": question_chain,
        }
    )

    text = """
        LangChain 是一个用于开发大模型应用的框架。
        LCEL 可以把 Prompt、Model、Parser、Retriever 等组件组合成链路。
        Runnable 是 LCEL 的核心抽象。
    """

    print("\n========== 1. invoke：单次调用 ==========")
    result = analysis_chain.invoke(text)
    print(result)

    print("\n========== 2. batch：批量调用 ==========")
    batch_results = analysis_chain.batch(
        [
            "PromptTemplate 负责把变量渲染成模型输入。",
            "Retriever 负责根据问题取回相关文档。",
            "Callback 可以记录链路执行过程。",
        ]
    )
    for index, item in enumerate(batch_results, start=1):
        print(f"\n--- 第 {index} 条结果 ---")
        print(item)

    print("\n========== 3. stream：流式输出 ==========")
    # 当前 analysis_chain 的最终输出是 dict，因此 stream 会逐步产出 dict 片段。
    for chunk in analysis_chain.stream(text):
        print(chunk)


if __name__ == "__main__":
    main()

