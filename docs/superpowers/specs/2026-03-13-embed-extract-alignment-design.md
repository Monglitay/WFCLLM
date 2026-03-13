# 设计文档：嵌入端与提取端节点对齐诊断实验

**日期**：2026-03-13
**目标**：排查 WFCLLM 水印生成（Embedding）与检测（Extraction）命中率不一致的 Bug

---

## 背景

Generator 生成含水印代码时能成功嵌入水印，但将生成的代码送入 Detector 进行检测时水印命中率极低。怀疑在生成端（流式 AST 事件）和提取端（静态 Tree-sitter 解析）之间存在非对称性。

三个具体怀疑点：

1. **文本边界差异（怀疑一）**：生成端 `event.block_text`（增量累积字符串的字节切片）与提取端 `block.source`（`child.text.decode()`）在空白字符上可能存在细微差异，导致 LSH 签名突变。
2. **parent_node_type 推断差异（怀疑二）**：`interceptor._walk()` 的父类型传播逻辑与 `scorer._resolve_parent_type()` 的 `parent_id` 查找逻辑在某些嵌套结构下可能产生不同结果，导致 valid_set 密钥空间偏移。
3. **Compound/Simple 过滤不对称（怀疑三）**：生成端 fallback/cascade 机制可能在 compound block 上成功嵌入水印，但提取端 `detector.py` 只检测 simple block，导致这部分嵌入量在检测时完全不可见。

---

## 方案选择

选择**方案二：子类化 WatermarkGenerator**。

在 `experiment/` 下继承 `WatermarkGenerator`，重写 `generate()` 主循环，在关键判断点（simple 通过、retry 成功、retry 耗尽、fallback 成功、cascade 成功/失败）插入事件记录。不修改 `wfcllm/` 任何代码。

---

## 整体架构

```
experiment/
└── embed_extract_alignment/
    ├── diagnostic_generator.py   # 继承 WatermarkGenerator，捕获 embed events
    ├── aligner.py                # 对比两端 block，输出对齐结果
    ├── report.py                 # 生成 JSON 报告 + 控制台摘要
    └── run.py                    # 主入口：加载 HumanEval，跑 N 条，输出报告
```

### 数据流

```
HumanEval prompt
    → DiagnosticGenerator.generate()
        → 复制 WatermarkGenerator.generate() 主循环，在关键点记录 EmbedEvent
    → generated_code + List[EmbedEvent]

generated_code
    → WatermarkDetector.detect()
    → 反查 extract_statement_blocks() → List[StatementBlock]

(List[EmbedEvent], List[StatementBlock])
    → Aligner.align()
    → PromptReport

→ 控制台摘要 + JSON 报告
```

---

## 核心数据结构

### EmbedEvent

```python
@dataclass
class EmbedEvent:
    path: Literal["simple", "fallback", "cascade"]
    block_text: str
    parent_node_type: str
    node_type: str
    passed: bool
```

### AlignedPair

```python
@dataclass
class AlignedPair:
    embed: EmbedEvent
    extract: StatementBlock
    text_match: bool           # block_text == block.source
    parent_match: bool         # parent_node_type 是否一致
    embed_score: int           # 生成端：passed=1, else=0
    extract_score: int         # 提取端：BlockScore.score
    score_agree: bool
```

### PromptReport

```python
@dataclass
class PromptReport:
    prompt_id: str
    generated_code: str
    embed_events: list[EmbedEvent]
    aligned_pairs: list[AlignedPair]
    unmatched_embeds: list[EmbedEvent]      # 生成端有、提取端找不到
    unmatched_extracts: list[StatementBlock] # 提取端有、生成端没记录

    # 摘要统计
    embed_total: int
    embed_simple_passed: int
    embed_fallback_passed: int
    embed_cascade_passed: int
    extract_simple_count: int
    text_mismatch_count: int
    parent_mismatch_count: int
    score_disagree_count: int
```

---

## DiagnosticGenerator 实现细节

继承 `WatermarkGenerator`，**完整复制** `generate()` 主循环到子类，在以下五个判断点插入记录：

| 判断点 | path | passed |
|--------|------|--------|
| `verify_result.passed` 为 True（首次验证通过） | simple | True |
| `retry_result.success` 为 True | simple | True |
| retry 耗尽（`stats.failed_blocks += 1`） | simple | False |
| fallback 成功（`stats.fallback_blocks += 1`） | fallback | True |
| cascade 成功/失败 | cascade | True/False |

接受代码重复——这是 `experiment/` 诊断代码，不追求复用。

---

## 对齐策略

1. 主对齐：`embed.block_text.strip() == extract.source.strip()`（文本精确匹配，容忍首尾空白）
2. 退化对齐：文本不匹配时，用 `(start_line, end_line)` 做位置对齐
3. 无法对齐的记入 `unmatched_embeds` / `unmatched_extracts`

---

## run.py 主入口

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--n_samples` | 20 | 跑前 N 条 HumanEval |
| `--output_dir` | `data/diag_reports/` | 报告输出目录 |
| `--secret_key` | 必填 | 水印密钥 |

### 运行流程

1. 加载 LLM + encoder（`data/models/codet5-base/`）
2. 从 `data/datasets/humaneval` 加载前 N 条 prompt
3. 对每条 prompt：
   - `DiagnosticGenerator.generate(prompt)` → `(result, embed_events)`
   - `WatermarkDetector.detect(generated_code)` → `detection_result`
   - `extract_statement_blocks(generated_code)` → `all_blocks`
   - `Aligner.align(embed_events, simple_blocks, all_blocks)` → `PromptReport`
4. 汇总 → `SummaryReport`
5. 输出 JSON + 控制台摘要

### 控制台摘要格式

```
[HumanEval/0] embed=12 simple=9✓ fallback=2✓ failed=1
  extract_simple=8  aligned=7  unmatched_embed=2  unmatched_extract=1
  text_mismatch=0  parent_mismatch=1  score_disagree=3
```

### 输出文件

```
data/diag_reports/
├── summary_YYYYMMDD_HHMMSS.json    # 跨所有 prompt 的聚合统计
└── details_YYYYMMDD_HHMMSS.jsonl   # 每条 prompt 的完整 PromptReport
```

`summary.json` 关键诊断字段：

```json
{
  "n_prompts": 20,
  "total_embed_events": 240,
  "compound_only_events": 45,
  "compound_ratio": 0.1875,
  "text_mismatch_total": 3,
  "parent_mismatch_total": 8,
  "score_disagree_total": 12,
  "avg_embed_rate": 0.72,
  "avg_detect_z": 1.83
}
```

---

## 诊断结论判读

| 指标异常 | 对应怀疑点 | 下一步 |
|---------|-----------|--------|
| `text_mismatch_total` 高 | 怀疑一 | 对比 interceptor vs ast_parser 的字节切片逻辑 |
| `parent_mismatch_total` 高 | 怀疑二 | 对比 `_walk()` 与 `_resolve_parent_type()` 的父类型传播 |
| `compound_ratio` 高（>20%） | 怀疑三 | 考虑在 detector 中支持 compound block 检测，或在 generator 中禁用 fallback |
| `score_disagree_total` 高但以上均低 | 其他 | 检查 LSH 空间、keying 参数一致性 |
