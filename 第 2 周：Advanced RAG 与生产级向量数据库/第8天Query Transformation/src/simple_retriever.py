from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from .models import Document, RetrievedNode, TransformedQuery


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    tokens = [item.lower() for item in TOKEN_PATTERN.findall(text)]
    bigrams = [tokens[i] + tokens[i + 1] for i in range(len(tokens) - 1)]
    return tokens + bigrams


class SimpleTfidfRetriever:
    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.doc_tokens = {doc.doc_id: tokenize(doc.title + " " + doc.text) for doc in documents}
        self.doc_vectors = {doc.doc_id: Counter(tokens) for doc, tokens in zip(documents, self.doc_tokens.values())}
        self.idf = self._build_idf()

    def retrieve(self, query: TransformedQuery, top_k: int = 5) -> list[RetrievedNode]:
        query_vector = Counter(tokenize(query.text))
        scored = []

        for doc in self.documents:
            doc_vector = self.doc_vectors[doc.doc_id]
            score = self._cosine_similarity(query_vector, doc_vector) * query.weight
            if score > 0:
                scored.append((doc, score))

        scored.sort(key=lambda item: item[1], reverse=True)

        return [
            RetrievedNode(
                doc=doc,
                score=score,
                rank=rank,
                query_text=query.text,
                strategy=query.strategy,
            )
            for rank, (doc, score) in enumerate(scored[:top_k], start=1)
        ]

    def _build_idf(self) -> dict[str, float]:
        df: dict[str, int] = defaultdict(int)
        total_docs = len(self.documents)

        for tokens in self.doc_tokens.values():
            for token in set(tokens):
                df[token] += 1

        return {
            token: math.log((1 + total_docs) / (1 + count)) + 1
            for token, count in df.items()
        }

    def _weighted(self, vector: Counter[str]) -> dict[str, float]:
        return {
            token: count * self.idf.get(token, 1.0)
            for token, count in vector.items()
        }

    def _cosine_similarity(self, left: Counter[str], right: Counter[str]) -> float:
        left_weighted = self._weighted(left)
        right_weighted = self._weighted(right)
        common_tokens = set(left_weighted) & set(right_weighted)

        dot = sum(left_weighted[token] * right_weighted[token] for token in common_tokens)
        left_norm = math.sqrt(sum(value * value for value in left_weighted.values()))
        right_norm = math.sqrt(sum(value * value for value in right_weighted.values()))

        if left_norm == 0 or right_norm == 0:
            return 0.0

        return dot / (left_norm * right_norm)

