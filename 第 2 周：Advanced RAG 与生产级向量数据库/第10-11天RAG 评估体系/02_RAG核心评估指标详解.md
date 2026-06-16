# RAG 核心评估指标详解

## 一、RAG 评估为什么复杂

普通问答系统只看“答案对不对”往往还勉强够用，但 RAG 不行。RAG 的答案来自多个环节：

1. 用户问题是否被正确理解。
2. 检索器是否找到相关文档。
3. 重排器是否把最有用文档放前面。
4. 上下文是否完整、干净、不过长。
5. LLM 是否只基于上下文回答。
6. 答案是否满足用户意图和业务格式。

所以 RAG 评估必须是分层的。一个答案错了，可能不是模型差，而是检索没找对；一个答案看起来对，也可能是模型凭常识答对而不是基于你的知识库答对。

## 二、评估对象分层

| 层级 | 评估对象 | 典型指标 | 解决的问题 |
|---|---|---|---|
| Query | 原始问题、改写问题、扩展查询 | query rewrite success、intent match | 问题有没有被正确表达给检索器 |
| Retrieval | retrieved_contexts、doc ids、scores | Hit Rate、Recall@K、MRR、nDCG、Context Recall | 该找的材料有没有找回来 |
| Ranking | 检索结果顺序 | Context Precision、MRR、nDCG | 相关材料是否靠前 |
| Context | 拼接后的上下文 | Context Relevancy、Noise Sensitivity | 上下文是否干净、有用、抗噪 |
| Generation | response | Faithfulness、Answer Relevancy、Answer Correctness | 答案是否忠实、相关、正确 |
| Product | 用户任务结果 | pass rate、CSAT、人工评分、latency、cost | 系统是否真的可用 |

## 三、检索侧指标

### 1. Hit Rate@K

定义：前 K 个检索结果里，只要出现至少一个相关文档，就记为命中。

适合：

- FAQ、单跳问答。
- 只需要一个证据就能回答的问题。
- 快速衡量检索器是否“至少找到了线索”。

公式：

```text
HitRate@K = 命中问题数 / 总问题数
```

例子：

```text
问题：Milvus collection 如何创建索引？
gold doc id: doc_12
top_3 retrieved ids: [doc_03, doc_12, doc_99]
HitRate@3 = 1
```

局限：

- 不关心相关文档排第几。
- 不关心是否找全多个证据。
- 对多跳问题不够敏感。

### 2. Precision@K

定义：前 K 个结果里有多少比例是相关的。

公式：

```text
Precision@K = top K 中相关文档数 / K
```

适合：

- 控制上下文噪声。
- top_k 较大时判断是否塞入太多无关内容。

例子：

```text
top_5 = [相关, 无关, 相关, 无关, 无关]
Precision@5 = 2 / 5 = 0.4
```

工程含义：

- Precision 高：送给 LLM 的上下文更干净。
- Precision 低：LLM 更容易被噪声带偏，token 成本更高。

### 3. Recall@K

定义：所有应该找回的相关文档中，有多少出现在前 K 个结果里。

公式：

```text
Recall@K = top K 中相关文档数 / 全部相关文档数
```

适合：

- 多跳问答。
- 合规、金融、医疗等必须找全证据的场景。
- 判断 top_k 是否太小，chunk 是否切坏，query 是否表达不充分。

例子：

```text
gold docs = [doc_1, doc_2, doc_3, doc_4]
top_5 = [doc_1, doc_8, doc_2, doc_9, doc_10]
Recall@5 = 2 / 4 = 0.5
```

### 4. MRR

MRR 是 Mean Reciprocal Rank，关注第一个相关结果出现得有多早。

公式：

```text
单条样本 RR = 1 / 第一个相关文档的排名
MRR = 所有样本 RR 的平均值
```

例子：

```text
第一个相关文档排第 1：RR = 1.0
第一个相关文档排第 2：RR = 0.5
第一个相关文档排第 5：RR = 0.2
没有相关文档：RR = 0
```

适合：

- 用户只会看前几个搜索结果的场景。
- RAG prompt 只放前 3-5 个 chunk 的场景。

### 5. nDCG

nDCG 适合相关性不是二值的场景。比如：

- 3 分：直接完整支持答案。
- 2 分：部分支持答案。
- 1 分：主题相关但不能回答。
- 0 分：无关。

它同时考虑“相关性强弱”和“排名位置”。越相关的文档排得越靠前，nDCG 越高。

适合：

- 搜索系统。
- 重排器评估。
- 有人工多级相关性标注的数据集。

### 6. RAGAs Context Precision

Context Precision 衡量检索器是否把相关 chunk 排在前面。它本质上关注“相关内容是否尽早出现”。

需要的字段通常包括：

```text
user_input
retrieved_contexts
reference 或 response
```

RAGAs 文档中说明，Context Precision 会计算每个位置的 precision@k，并结合该位置是否相关来得到最终分数。

高分意味着：

- 有用 context 排在前面。
- LLM 更可能优先看到正确证据。
- prompt 预算利用率更好。

低分意味着：

- 相关文档被埋在后面。
- 需要 reranker、metadata filter、hybrid search 或更好的 chunking。

### 7. RAGAs Context Recall

Context Recall 衡量 reference 中需要的关键信息，有多少能被 retrieved contexts 支持。

需要的字段通常包括：

```text
user_input
retrieved_contexts
reference
```

RAGAs 的 LLM-based Context Recall 会把 reference 拆成 claim，再判断每个 claim 是否能从 retrieved contexts 推出。

高分意味着：

- 必要证据基本找全。
- 答案错的原因可能在 generation。

低分意味着：

- 检索没覆盖答案证据。
- 优先优化 query rewrite、embedding、hybrid、top_k、chunk 策略。

## 四、上下文质量指标

### 1. Context Relevancy

衡量 retrieved contexts 对 user_input 是否相关。

它和 Context Precision 的差别：

- Context Precision 更强调“相关内容是否排前面”。
- Context Relevancy 更强调“拿到的上下文本身是否对问题有用”。

低分常见原因：

- embedding 模型和领域不匹配。
- chunk 太大，主题混杂。
- query 太短或包含歧义。
- 没有 metadata filter。

### 2. Noise Sensitivity

Noise Sensitivity 衡量系统在有噪声上下文时是否容易产生错误答案。RAGAs 中该指标分数越低越好。

它尤其适合评估：

- top_k 增大后是否引入太多噪声。
- reranker 是否有效过滤无关内容。
- prompt 是否会被冲突信息诱导。
- no-answer 场景下模型是否胡编。

典型坏例子：

```text
问题：A 产品支持哪些导出格式？
正确上下文：支持 CSV 和 JSON。
噪声上下文：B 产品支持 PDF。
坏答案：A 产品支持 CSV、JSON 和 PDF。
```

这个答案看起来丰富，但 PDF 是从噪声里错误迁移来的。

## 五、生成侧指标

### 1. Faithfulness

Faithfulness 衡量 response 是否和 retrieved contexts 事实一致。RAGAs 官方定义中，它会检查答案中的 claims 是否能被 retrieved context 支持，分数范围通常是 0 到 1，越高越忠实。

公式理解：

```text
Faithfulness = 被上下文支持的答案声明数 / 答案声明总数
```

高分意味着：

- 答案基本都能从上下文推出。
- 幻觉较少。

低分意味着：

- LLM 编造了上下文没有的信息。
- prompt 缺少“只基于资料回答”的约束。
- 上下文存在冲突，模型错误选择了某一方。
- 模型用自己参数知识补全了答案。

注意：

Faithfulness 不等于 Answer Correctness。一个答案可以忠实于错误上下文，但事实仍然不正确。

### 2. Answer Relevancy

Answer Relevancy 衡量 response 是否回答了 user_input。RAGAs 文档中该指标会基于 response 反推若干人工问题，再计算这些问题和原始 user_input 的 embedding 相似度。

高分意味着：

- 答案围绕用户问题。
- 没有明显答非所问。

低分意味着：

- 答案跑题。
- 答案只回答了问题的一部分。
- 答案加入太多无关细节。

注意：

Answer Relevancy 不评估事实正确性。一个答案可能很相关，但完全是错的。

### 3. Answer Correctness

Answer Correctness 衡量 response 是否和 reference 一致。

适合：

- 有标准答案的评估集。
- 问答、客服、文档助手、考试类任务。

局限：

- 标准答案可能不唯一。
- reference 写得太短会误伤合理答案。
- 对开放式问题需要 rubric，而不是单一 reference。

### 4. Factual Correctness

Factual Correctness 更关注事实声明是否正确，通常需要 reference 或事实来源。

它和 Faithfulness 的关系：

| 指标 | 对比对象 | 评价重点 |
|---|---|---|
| Faithfulness | response vs retrieved_contexts | 是否基于检索上下文 |
| Factual Correctness | response vs reference/ground truth | 是否事实正确 |

典型组合：

```text
Faithfulness 高 + Correctness 高：理想
Faithfulness 低 + Correctness 高：可能靠模型常识答对，但没有基于知识库
Faithfulness 高 + Correctness 低：检索上下文可能错、旧、冲突
Faithfulness 低 + Correctness 低：检索和生成都需要查
```

## 六、端到端业务指标

自动指标不能完全替代业务指标。上线前至少补充：

| 指标 | 含义 |
|---|---|
| pass rate | 按业务 rubric 判断通过率 |
| escalation rate | 需要转人工或拒答的比例 |
| no-answer accuracy | 文档无答案时是否正确拒答 |
| citation accuracy | 引用是否真的支持答案 |
| latency p50/p95/p99 | 用户等待时间 |
| cost per query | 单次问答成本 |
| token usage | prompt 和 completion token |
| stability | 同一问题多次回答是否稳定 |

## 七、指标组合建议

### 最小评估组合

适合第一版快速上手：

```text
Context Precision
Context Recall
Faithfulness
Answer Relevancy
Answer Correctness 或自定义 correctness pass/fail
```

### 检索优化组合

适合调 embedding、hybrid、reranker、top_k：

```text
Hit Rate@K
Recall@K
MRR
nDCG
Context Precision
Context Recall
latency
```

### 幻觉治理组合

适合做 grounded generation：

```text
Faithfulness
Factual Correctness
Noise Sensitivity
Citation Accuracy
No-answer Accuracy
```

### 上线回归组合

适合每次发版：

```text
Pass Rate
Context Recall
Faithfulness
Answer Relevancy
Latency p95
Cost per Query
Top Failure Categories
```

## 八、指标解读矩阵

| Context Recall | Context Precision | Faithfulness | 诊断 |
|---:|---:|---:|---|
| 低 | 低 | 低 | 检索整体差，先修 query/index/chunk |
| 高 | 低 | 低 | 找到了但噪声多，优先 rerank/compress |
| 高 | 高 | 低 | 检索没问题，生成不受约束，改 prompt/model |
| 低 | 高 | 高 | 找到的少但回答保守，扩 recall |
| 高 | 高 | 高 | 主链路健康，关注 correctness/latency/cost |

## 九、人工评估 Rubric 模板

自动指标之外，建议抽样人工评分。

| 维度 | 0 分 | 1 分 | 2 分 |
|---|---|---|---|
| 正确性 | 主要事实错误 | 部分正确 | 完全正确 |
| 完整性 | 漏掉关键点 | 覆盖部分关键点 | 覆盖所有关键点 |
| 忠实性 | 大量无依据内容 | 少量无依据内容 | 全部有依据 |
| 相关性 | 答非所问 | 部分回答 | 直接回答 |
| 可用性 | 不能用于决策 | 需要人工修正 | 可直接使用 |

通过标准示例：

```text
总分 >= 8 且忠实性 >= 2，视为 pass。
若出现严重事实错误，无论总分多少，直接 fail。
```

## 十、参考资料要点

- RAGAs 的 RAG 指标包括 Context Precision、Context Recall、Noise Sensitivity、Response/Answer Relevancy、Faithfulness 等。
- RAGAs 的 Faithfulness 检查答案声明是否能从 retrieved contexts 推出。
- RAGAs 的 Context Recall 需要 reference，用 reference claims 判断 retrieved contexts 是否覆盖必要信息。
- RAGAs 的 Evaluate and Improve RAG App 教程强调先建立评估集、定义指标、跑 baseline、分析失败、做针对性优化、再次对比。
- DeepEval 提供 RAG 指标、Agent 指标、多轮指标，并支持类似 Pytest 的评估组织方式。
- FlashRAG 更偏研究复现和 RAG pipeline 组件化，提供 retriever、reranker、generator、compressor 等组件和 benchmark 数据。
- LightEval 更偏 LLM 基准测试，适合评估模型本身而不是某个具体 RAG 应用。
