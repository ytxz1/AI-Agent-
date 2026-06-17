from __future__ import annotations

from collections import defaultdict

from .models import FusedResult, RetrievedNode


def fuse_with_rrf(results: list[RetrievedNode], top_k: int = 5, rrf_k: int = 60) -> list[FusedResult]:
    fusion_scores: dict[str, float] = defaultdict(float)
    best_scores: dict[str, float] = defaultdict(float)
    docs = {}
    strategies: dict[str, set[str]] = defaultdict(set)
    matched_queries: dict[str, list[str]] = defaultdict(list)
    ranks: dict[str, list[int]] = defaultdict(list)

    for item in results:
        doc_id = item.doc.doc_id
        docs[doc_id] = item.doc
        fusion_scores[doc_id] += 1.0 / (rrf_k + item.rank)
        best_scores[doc_id] = max(best_scores[doc_id], item.score)
        strategies[doc_id].add(item.strategy)
        matched_queries[doc_id].append(item.query_text)
        ranks[doc_id].append(item.rank)

    ranked_doc_ids = sorted(
        fusion_scores,
        key=lambda doc_id: (fusion_scores[doc_id], best_scores[doc_id]),
        reverse=True,
    )

    return [
        FusedResult(
            doc=docs[doc_id],
            fusion_score=fusion_scores[doc_id],
            best_score=best_scores[doc_id],
            strategies=sorted(strategies[doc_id]),
            matched_queries=matched_queries[doc_id],
            ranks=ranks[doc_id],
        )
        for doc_id in ranked_doc_ids[:top_k]
    ]

