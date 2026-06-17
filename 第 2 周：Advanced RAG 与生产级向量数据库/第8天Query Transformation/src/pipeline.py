from __future__ import annotations

from .fusion import fuse_with_rrf
from .models import FusedResult, RetrievedNode, TransformedQuery
from .simple_retriever import SimpleTfidfRetriever


class RetrievalPipeline:
    def __init__(self, retriever: SimpleTfidfRetriever, per_query_top_k: int = 4, final_top_k: int = 5):
        self.retriever = retriever
        self.per_query_top_k = per_query_top_k
        self.final_top_k = final_top_k

    def retrieve(self, transformed_queries: list[TransformedQuery]) -> tuple[list[RetrievedNode], list[FusedResult]]:
        all_results = []

        for query in transformed_queries:
            all_results.extend(self.retriever.retrieve(query, top_k=self.per_query_top_k))

        fused = fuse_with_rrf(all_results, top_k=self.final_top_k)
        return all_results, fused

