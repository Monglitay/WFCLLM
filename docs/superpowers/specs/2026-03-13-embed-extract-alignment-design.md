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

在 `experiment/` 下继承 `WatermarkGenerator`，重写 `generate()` 主循环，在关键判断点插入事件记录。不修改 `wfcllm/` 任何代码。

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
    → gen.generate(prompt)
        → 复制 WatermarkGenerator.generate() 主循环，在关键点记录 EmbedEvent
        → 返回 GenerateResult（与父类签名完全一致）
        → self.embed_events: list[EmbedEvent] 填充完毕

embed_events = gen.embed_events
generated_code = result.code

generated_code
    → WatermarkDetector.detect(generated_code)
    → detection_result（含 z_score、is_watermarked、block_details: list[BlockScore]）

    → extract_statement_blocks(generated_code) → all_blocks: list[StatementBlock]
    → simple_blocks = [b for b in all_blocks if b.block_type == "simple"]

Aligner.align(
    embed_events,
    simple_blocks,
    all_blocks,
    block_scores=detection_result.block_details,
    generated_code=generated_code,
) → PromptReport

→ 汇总 → SummaryReport → 控制台摘要 + JSON 报告
```

---

## 核心数据结构

### EmbedEvent

```python
@dataclass
class EmbedEvent:
    path: Literal["simple", "fallback", "cascade"]
    block_text: str            # 生成端捕获的原始文本
    parent_node_type: str      # 生成端推断的父节点类型
    node_type: str             # 当前节点类型
    passed: bool               # 该 block 的最终嵌入结果（含 retry 后结果）
```

**不存储行号**：`InterceptEvent` 只提供 `token_start_idx`/`token_count`，无行号信息。对齐时的行号由 `Aligner` 在 `generated_code` 中通过文本搜索（`block_text` 首次出现位置）推算，见对齐策略。

### AlignedPair

```python
@dataclass
class AlignedPair:
    embed: EmbedEvent          # 生成端事件
    extract: StatementBlock    # 提取端块（文本或位置匹配的结果）
    text_match: bool           # embed.block_text.strip() == extract.source.strip()
    parent_match: bool         # embed.parent_node_type == resolved_parent_type(extract, all_blocks)
    embed_passed: bool         # embed.passed
    extract_score: int | None  # BlockScore.score（仅 simple block 有值，compound 对应 None）
    score_agree: bool          # extract_score is not None and int(embed_passed) == extract_score
```

**compound block 说明**：fallback/cascade 路径的 `EmbedEvent` 对应 compound block，`WatermarkDetector` 不对其评分，因此 `extract_score=None`，`score_agree=False`。这类 pair 记入 `aligned_pairs`，但不计入 `score_disagree_count`。

### PromptReport

```python
@dataclass
class PromptReport:
    prompt_id: str
    generated_code: str
    embed_events: list[EmbedEvent]
    aligned_pairs: list[AlignedPair]
    unmatched_embeds: list[EmbedEvent]        # 生成端有、提取端找不到
    unmatched_extracts: list[StatementBlock]  # 提取端有、生成端没记录
    detect_z_score: float
    detect_is_watermarked: bool

    # 摘要统计（均可从上方列表直接派生）
    embed_total: int               # len(embed_events)
    embed_simple_passed: int       # path=="simple" and passed==True
    embed_simple_failed: int       # path=="simple" and passed==False
    embed_fallback_passed: int     # path=="fallback" and passed==True
    embed_fallback_failed: int     # path=="fallback" and passed==False（fallback 测试但未通过）
    embed_cascade_passed: int      # path=="cascade" and passed==True
    embed_cascade_failed: int      # path=="cascade" and passed==False
    embed_compound_total: int      # fallback+cascade 的全部事件数（怀疑三核心指标）
    extract_simple_count: int      # len(simple_blocks)
    embed_unmatched_count: int     # len(unmatched_embeds)
    extract_unmatched_count: int   # len(unmatched_extracts)
    compound_aligned_count: int    # aligned_pairs 中 extract_score is None 的数量
    text_mismatch_count: int       # aligned_pairs 中 text_match==False 的数量
    parent_mismatch_count: int     # aligned_pairs 中 parent_match==False 的数量
    score_disagree_count: int      # aligned_pairs 中 score_agree==False 且 extract_score is not None 的数量
```

**控制台 failed 字段**的值为 `embed_simple_failed`（对应 `EmbedStats.failed_blocks`）。

### SummaryReport

```python
@dataclass
class SummaryReport:
    n_prompts: int
    total_embed_events: int      # sum(r.embed_total for r in reports)
    compound_only_events: int    # sum(r.embed_compound_total for r in reports)
                                 # = sum of all fallback+cascade events（非仅 passed）
    compound_ratio: float        # compound_only_events / total_embed_events
    text_mismatch_total: int     # sum(r.text_mismatch_count)
    parent_mismatch_total: int   # sum(r.parent_mismatch_count)
    score_disagree_total: int    # sum(r.score_disagree_count)
    avg_embed_rate: float        # mean(r.embed_simple_passed / r.embed_total)
                                 # for r in reports if r.embed_total > 0
    avg_detect_z: float          # mean(r.detect_z_score for r in reports)
```

---

## DiagnosticGenerator 实现细节

### 返回协议

`DiagnosticGenerator.generate(prompt)` 返回 `GenerateResult`，与父类签名完全一致（drop-in compatible）。调用后通过 `gen.embed_events: list[EmbedEvent]` 读取事件列表。

```python
class DiagnosticGenerator(WatermarkGenerator):
    embed_events: list[EmbedEvent]  # populated after each generate() call

    def generate(self, prompt: str) -> GenerateResult:
        self.embed_events = []
        # 完整复制父类主循环，在下方六个判断点调用 self._record(...)
        ...
        return GenerateResult(code=ctx.generated_text, stats=stats)
```

### 六个记录点

| # | 触发条件 | path | passed |
|---|----------|------|--------|
| 1 | `verify_result.passed` 为 True（首次验证通过，无需 retry） | simple | True |
| 2 | `retry_result.success` 为 True（retry 后通过） | simple | True |
| 3 | retry 耗尽（进入 `stats.failed_blocks += 1` 分支） | simple | False |
| 4 | fallback 通过（`result.passed == True` in `_try_passive_fallback` 逻辑） | fallback | True |
| 5 | fallback 未通过（compound block 被测试但 `result.passed == False`） | fallback | False |
| 6 | cascade 成功或失败（`stats.cascade_blocks += 1` 或 cascade 验证未通过） | cascade | True/False |

**记录点 6 说明**：cascade 的 `block_text` 和 `parent_node_type` 来自**新生成的** compound block（`compound_event`），不是触发 cascade 的原始 block。这是诊断怀疑一（文本边界）时的正确口径。

接受代码重复——这是 `experiment/` 诊断代码，不追求复用。

---

## 对齐策略（Aligner.align）

```python
def align(
    embed_events: list[EmbedEvent],
    simple_blocks: list[StatementBlock],
    all_blocks: list[StatementBlock],
    block_scores: list[BlockScore],
    generated_code: str,
) -> PromptReport
```

### 主对齐（文本精确匹配）

```python
text_match = embed.block_text.strip() == extract.source.strip()
```

遍历所有 `EmbedEvent`：
- path=="simple"：在 `simple_blocks` 中找文本匹配的 `StatementBlock`
- path in ("fallback","cascade")：在 `all_blocks` 中找文本匹配的 `StatementBlock`（含 compound）

### 退化对齐（位置匹配）

当文本匹配失败时，通过以下算法计算 `EmbedEvent` 的行范围，与 `StatementBlock.start_line/end_line` 对比：

```python
# 在 generated_code 中查找 block_text 首次出现的位置
idx = generated_code.find(embed.block_text)
if idx != -1:
    start_line = generated_code[:idx].count("\n") + 1
    end_line = generated_code[:idx + len(embed.block_text)].count("\n") + 1
    # 与 StatementBlock.start_line/end_line 对比
```

若 `block_text` 在 `generated_code` 中不存在（极端情况），放入 `unmatched_embeds`。

### `extract_score` 查找

```python
score_map = {s.block_id: s.score for s in block_scores}
extract_score = score_map.get(extract.block_id, None)
# simple block 必有 block_id 在 score_map 中；compound block 不在 score_map 中，返回 None
```

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
   - `result = gen.generate(prompt)`；`embed_events = gen.embed_events`
   - `detection_result = detector.detect(result.code)`
   - `all_blocks = extract_statement_blocks(result.code)`；`simple_blocks = [b for b in all_blocks if b.block_type == "simple"]`
   - `report = Aligner.align(embed_events, simple_blocks, all_blocks, detection_result.block_details, result.code)`
   - 打印控制台摘要
4. 汇总所有 `PromptReport` → `SummaryReport`
5. 写入 JSON + JSONL

### 控制台摘要格式

```
[HumanEval/0] embed=12 simple=9✓ fallback=2✓ cascade=0 failed=1
  extract_simple=8  aligned=7  unmatched_embed=2  unmatched_extract=1
  compound_aligned=2  text_mismatch=0  parent_mismatch=1  score_disagree=3  z=2.41
```

### 输出文件

```
data/diag_reports/
├── summary_YYYYMMDD_HHMMSS.json    # SummaryReport（单个 JSON 对象）
└── details_YYYYMMDD_HHMMSS.jsonl   # 每行一条 PromptReport（序列化为 JSON）
```

---

## 诊断结论判读

| 指标异常 | 对应怀疑点 | 下一步 |
|---------|-----------|--------|
| `text_mismatch_total` 高 | 怀疑一 | 对比 interceptor vs ast_parser 的字节切片逻辑 |
| `parent_mismatch_total` 高 | 怀疑二 | 对比 `_walk()` 与 `_resolve_parent_type()` 的父类型传播 |
| `compound_ratio` 高（>20%） | 怀疑三 | 考虑在 detector 中支持 compound block 检测，或在 generator 中禁用 fallback |
| `score_disagree_total` 高但以上均低 | 其他 | 检查 LSH 空间、keying 参数一致性 |
