from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedNode:
    doc: Document
    score: float
    rank: int
    query_text: str
    strategy: str


@dataclass
class TransformedQuery:
    text: str
    strategy: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FusedResult:
    doc: Document
    fusion_score: float
    best_score: float
    strategies: list[str]
    matched_queries: list[str]
    ranks: list[int]

