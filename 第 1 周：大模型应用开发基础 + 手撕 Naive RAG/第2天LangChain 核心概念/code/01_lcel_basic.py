"""
01_lcel_basic.py

目标：
1. 用最小代码理解 LCEL 的核心写法：prompt | model | parser
2. 观察 prompt、model、parser 每一步的输入输出
3. 不依赖真实大模型 API，先把 LangChain 的数据流跑通

运行：
python code/01_lcel_basic.py
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda


def fake_chat_model(prompt_value):
    """
    这是一个“假的聊天模型”，用来模拟 ChatModel 的返回值。

    真实 ChatModel 通常返回 AIMessage。
    为了让离线示例在不同 langchain_core 版本中都输出普通字符串，
    这里直接返回字符串，让 StrOutputParser 继续保持在链路中。
    """
    messages = prompt_value.to_messages()
    human_message = messages[-1].content

    return (
        "这是一个模拟模型输出。\n"
        f"我收到的人类消息是：{human_message}\n"
            "在真实项目中，这一步会由 DeepSeek、通义千问、OpenAI 或本地模型完成。"
    )


def main():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一名耐心的 AI 应用开发导师。"),
            ("human", "请用初学者能理解的方式解释：{topic}"),
        ]
    )

    model = RunnableLambda(fake_chat_model)
    parser = StrOutputParser()

    chain = prompt | model | parser

    user_input = {"topic": "LangChain LCEL"}

    print("\n========== 1. 单独运行 prompt ==========")
    prompt_result = prompt.invoke(user_input)
    print(type(prompt_result))
    print(prompt_result)

    print("\n========== 2. 单独运行 model ==========")
    model_result = model.invoke(prompt_result)
    print(type(model_result))
    print(model_result)

    print("\n========== 3. 单独运行 parser ==========")
    parser_result = parser.invoke(model_result)
    print(type(parser_result))
    print(parser_result)

    print("\n========== 4. 运行完整 LCEL chain ==========")
    final_result = chain.invoke(user_input)
    print(type(final_result))
    print(final_result)


if __name__ == "__main__":
    main()
