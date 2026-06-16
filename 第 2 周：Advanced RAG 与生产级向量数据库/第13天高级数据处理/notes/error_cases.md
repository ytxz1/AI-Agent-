# 复杂 PDF 解析错误案例库

> 每次解析失败都要记录。高级数据处理的能力不是“某一次跑通”，而是持续收集错误页，并把错误页变成可复现、可评估、可修复的测试集。

## 错误案例模板

### Case 001

- 原始文件：
- 页码：
- 工具：
- 工具版本：
- 后端 / 参数：
- 错误类型：
- 严重程度：
- 是否影响 RAG：
- 期望结果：
- 实际结果：
- 相关输出文件：
- 初步原因：
- 可能修复：
- 是否加入 gold set：

## 错误类型枚举

| 类型 | 说明 |
|---|---|
| `missing_text` | 文本漏抽 |
| `duplicated_text` | 文本重复 |
| `wrong_order` | 阅读顺序错误 |
| `header_footer_noise` | 页眉页脚污染 |
| `table_missing` | 表格未识别 |
| `table_structure_error` | 表格结构错误 |
| `image_missing` | 图片未抽取 |
| `caption_mismatch` | caption 关联错误 |
| `formula_error` | 公式识别错误 |
| `ocr_error` | OCR 识别错误 |
| `bbox_error` | 坐标错误 |
| `encoding_error` | 编码或乱码 |
| `parser_crash` | 解析器崩溃 |

## 严重程度

| 等级 | 含义 |
|---|---|
| P0 | 解析失败，文档不可用 |
| P1 | 核心内容错误，RAG 结果会误导用户 |
| P2 | 部分结构错误，需要 fallback 或人工修正 |
| P3 | 轻微噪声，对 RAG 影响有限 |

