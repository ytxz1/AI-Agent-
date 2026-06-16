from langchain_core.documents import Document

from app.core.config import settings


class MockChatModel:
    """A tiny offline answer generator for local pipeline verification."""

    def generate(self, question: str, documents: list[Document]) -> str:
        if not documents:
            return "根据已上传文档，我不知道。"

        snippets = []
        for index, document in enumerate(documents[:3], start=1):
            text = " ".join(document.page_content.split())
            snippets.append(f"{index}. {text[:220]}")

        return (
            "这是 mock 模式下基于检索片段生成的回答。真实项目中请把 "
            "CHAT_PROVIDER 设置为 deepseek，并配置 DEEPSEEK_API_KEY。\n\n"
            f"问题：{question}\n\n"
            "可参考的检索片段：\n"
            + "\n".join(snippets)
        )


class OpenAIChatModel:
    def __init__(self) -> None:
        from langchain.chat_models import init_chat_model

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when CHAT_PROVIDER=openai")
        self.model = init_chat_model(settings.chat_model, model_provider="openai")

    def generate(self, question: str, documents: list[Document]) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        from app.prompts.rag_prompt import RAG_SYSTEM_PROMPT, build_user_prompt

        context = format_context(documents)
        response = self.model.invoke(
            [
                SystemMessage(content=RAG_SYSTEM_PROMPT),
                HumanMessage(content=build_user_prompt(context=context, question=question)),
            ]
        )
        return str(response.content)


class DeepSeekChatModel:
    def __init__(self) -> None:
        from langchain_openai import ChatOpenAI

        if not settings.deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY is required when CHAT_PROVIDER=deepseek")

        self.model = ChatOpenAI(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=0,
        )

    def generate(self, question: str, documents: list[Document]) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        from app.prompts.rag_prompt import RAG_SYSTEM_PROMPT, build_user_prompt

        context = format_context(documents)
        response = self.model.invoke(
            [
                SystemMessage(content=RAG_SYSTEM_PROMPT),
                HumanMessage(content=build_user_prompt(context=context, question=question)),
            ]
        )
        return str(response.content)


def format_context(documents: list[Document]) -> str:
    parts = []
    for index, document in enumerate(documents, start=1):
        filename = document.metadata.get("filename", "unknown")
        chunk_index = document.metadata.get("chunk_index", "unknown")
        parts.append(
            f"[片段 {index}] 文件: {filename}, chunk: {chunk_index}\n"
            f"{document.page_content}"
        )
    return "\n\n".join(parts)


def create_chat_model():
    provider = settings.chat_provider.lower().strip()

    if provider == "mock":
        return MockChatModel()

    if provider == "openai":
        return OpenAIChatModel()

    if provider == "deepseek":
        return DeepSeekChatModel()

    raise ValueError(f"Unsupported CHAT_PROVIDER: {settings.chat_provider}")
