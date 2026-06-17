# Query Transformation 查询改写

Query Transformation 发生在检索之前，目标是把用户原始问题转换成更适合检索的表达。

常见策略包括 original query、query rewrite、HyDE、Multi-Query 和 Query Decomposition。HyDE 会根据用户问题生成一段假设性文档，再用这段文本进行检索；Multi-Query 会从多个角度生成多个检索 query；Decomposition 会把复杂问题拆成多个子问题。

Query Transformation 适合处理过短、过宽泛、口语化、意图复杂或和知识库表达不一致的问题。但它也有风险：改写可能引入意图漂移，HyDE 可能生成不准确的假设文本。

生产系统中应始终保留 original query。改写 query 只用于检索，不应该直接作为事实证据。Reranker 阶段通常仍使用用户原始 query 来判断候选 chunk 是否真正相关。

