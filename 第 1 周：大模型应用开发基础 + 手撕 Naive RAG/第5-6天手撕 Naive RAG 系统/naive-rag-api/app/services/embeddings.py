import hashlib
import math
import re
from collections import Counter

from langchain_core.embeddings import Embeddings

from app.core.config import settings


TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]+", re.UNICODE)


class HashEmbeddings(Embeddings):
    """Small deterministic embedding model for offline learning and tests.

    This is not a semantic embedding model. It hashes tokens into a fixed-size
    vector, which is enough to exercise the full RAG pipeline without network
    calls or API keys.
    """

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        tokens = self._tokenize(text)
        counts = Counter(tokens)
        vector = [0.0] * self.dimensions

        for token, count in counts.items():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign * (1.0 + math.log(count))

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def _tokenize(self, text: str) -> list[str]:
        normalized = text.lower()
        tokens = TOKEN_PATTERN.findall(normalized)

        # Character n-grams make the offline hash model less brittle for
        # Chinese text, where whitespace-based tokenization is not enough.
        for cjk_span in CJK_PATTERN.findall(normalized):
            tokens.extend(self._ngrams(cjk_span, n=2))
            tokens.extend(self._ngrams(cjk_span, n=3))

        return tokens

    def _ngrams(self, text: str, n: int) -> list[str]:
        if len(text) < n:
            return [text]
        return [text[index : index + n] for index in range(len(text) - n + 1)]


def create_embeddings() -> Embeddings:
    provider = settings.embedding_provider.lower().strip()

    if provider == "hash":
        return HashEmbeddings()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
        return OpenAIEmbeddings(model=settings.embedding_model)

    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {settings.embedding_provider}")
