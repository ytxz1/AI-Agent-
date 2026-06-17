from __future__ import annotations

import json
from typing import Protocol

from .models import TransformedQuery


HYDE_PROMPT = """你是一个 RAG 查询改写器，当前策略是 HyDE。
请根据用户问题生成一段“可能出现在知识库中的说明性文档”。
这段文本只用于向量检索，不会直接作为最终答案。

要求：
1. 使用清晰、专业、可检索的表达。
2. 覆盖问题中的关键概念、同义词和相关术语。
3. 不要编造具体数字、日期、实验结果或不存在的专有名词。
4. 输出 1 段，长度控制在 120-250 中文字。

用户问题：
{query}
"""


MULTI_QUERY_PROMPT = """你是一个 RAG 查询改写器。
请把用户问题改写成 {num_queries} 个不同角度的检索查询。

要求：
1. 每个查询都必须保留用户原始意图。
2. 不要回答问题，只输出查询。
3. 查询之间要有明显差异，覆盖不同术语、同义表达或子角度。
4. 不要引入原问题没有的硬性事实条件。
5. 输出 JSON 数组，数组元素是字符串。

用户问题：
{query}
"""


DECOMPOSITION_PROMPT = """你是一个 RAG 查询拆解器。
请把用户问题拆解成若干个可以独立检索的子问题。

要求：
1. 子问题必须共同覆盖原始问题。
2. 每个子问题只问一个清晰的信息点。
3. 不要回答问题。
4. 如果原问题很简单，只返回原问题本身。
5. 输出 JSON 数组。
6. 子问题数量控制在 1-6 个。

用户问题：
{query}
"""


class LLM(Protocol):
    def complete(self, prompt: str) -> str:
        ...


class QueryTransformer(Protocol):
    def transform(self, query: str) -> list[TransformedQuery]:
        ...


class OriginalQueryTransformer:
    def transform(self, query: str) -> list[TransformedQuery]:
        return [TransformedQuery(text=query, strategy="original", weight=1.0)]


class HyDETransformer:
    def __init__(self, llm: LLM, include_original: bool = True):
        self.llm = llm
        self.include_original = include_original

    def transform(self, query: str) -> list[TransformedQuery]:
        transformed = []

        if self.include_original:
            transformed.append(TransformedQuery(text=query, strategy="original", weight=1.0))

        hypothetical_doc = self.llm.complete(HYDE_PROMPT.format(query=query)).strip()
        transformed.append(
            TransformedQuery(
                text=hypothetical_doc,
                strategy="hyde",
                weight=0.85,
                metadata={"source_query": query, "prompt": "HYDE_PROMPT"},
            )
        )

        return deduplicate_queries(transformed)


class MultiQueryTransformer:
    def __init__(self, llm: LLM, num_queries: int = 4, include_original: bool = True):
        self.llm = llm
        self.num_queries = num_queries
        self.include_original = include_original

    def transform(self, query: str) -> list[TransformedQuery]:
        prompt = MULTI_QUERY_PROMPT.format(query=query, num_queries=self.num_queries)
        raw_response = self.llm.complete(prompt)
        rewritten_queries = parse_json_string_list(raw_response)

        transformed = []
        if self.include_original:
            transformed.append(TransformedQuery(text=query, strategy="original", weight=1.0))

        for index, item in enumerate(rewritten_queries[: self.num_queries]):
            transformed.append(
                TransformedQuery(
                    text=item,
                    strategy="multi_query",
                    weight=0.9,
                    metadata={"source_query": query, "index": index, "prompt": "MULTI_QUERY_PROMPT"},
                )
            )

        return deduplicate_queries(transformed)


class DecompositionTransformer:
    def __init__(self, llm: LLM, include_original: bool = True):
        self.llm = llm
        self.include_original = include_original

    def transform(self, query: str) -> list[TransformedQuery]:
        raw_response = self.llm.complete(DECOMPOSITION_PROMPT.format(query=query))
        sub_questions = parse_json_string_list(raw_response)

        transformed = []
        if self.include_original:
            transformed.append(TransformedQuery(text=query, strategy="original", weight=1.0))

        for index, item in enumerate(sub_questions[:6]):
            transformed.append(
                TransformedQuery(
                    text=item,
                    strategy="decomposition",
                    weight=0.9,
                    metadata={"source_query": query, "index": index, "prompt": "DECOMPOSITION_PROMPT"},
                )
            )

        return deduplicate_queries(transformed)


class QueryTransformationPipeline:
    def __init__(self, transformers: list[QueryTransformer]):
        self.transformers = transformers

    def transform(self, query: str) -> list[TransformedQuery]:
        transformed = []
        for transformer in self.transformers:
            transformed.extend(transformer.transform(query))
        return deduplicate_queries(transformed)


def parse_json_string_list(raw_response: str) -> list[str]:
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        return fallback_parse_lines(raw_response)

    if not isinstance(parsed, list):
        return fallback_parse_lines(raw_response)

    return clean_query_list([str(item) for item in parsed])


def fallback_parse_lines(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        cleaned = line.strip().strip("-").strip()
        if cleaned:
            lines.append(cleaned)
    return clean_query_list(lines)


def clean_query_list(queries: list[str]) -> list[str]:
    cleaned = []
    for query in queries:
        item = query.strip().strip('"').strip("'")
        if len(item) > 200:
            item = item[:200]
        if item:
            cleaned.append(item)
    return cleaned


def deduplicate_queries(queries: list[TransformedQuery]) -> list[TransformedQuery]:
    seen = set()
    result = []

    for query in queries:
        normalized = query.text.strip().lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(query)

    return result
