# WFCLLM 可复现门控语义窗口水印实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。**但本项目当前用户约束明确禁止 subagent，因此在该约束解除前只能使用 `superpowers-zh:executing-plans` 单代理顺序执行。**

**目标：** 在不破坏现有 `evidence_retry_seed7x3` 正式基线的前提下，新增 `gated_semantic_window_v1`：使用同一个小门模型在生成端和提取端把最终 Python 代码确定性地划成互不重叠、每个最多三个语句单元的窗口；当窗口适合嵌入但原始窗口未进入由“部署密钥 + 规范父节点描述”派生的允许 LSH 区域时，回退并重新生成整个窗口；同时把门模型数据生成、训练、量化稳定性验证接入主编排。

**架构：** 新增共享的 `wfcllm.windowing` 作为生成端与提取端唯一的语句单元、父节点描述和窗口划分实现；新增 `wfcllm.gate` 负责无密钥门模型输入、数据、标签、训练、bundle 与跨运行时验证；新增独立的 gated generation/detection 路径，复用现有语义编码、LSH、严格 final-code-only 输入、审计和报告能力。现有代理窗口检测器与 evidence-retry 生成器不改语义，继续作为基线。正式 gated 流程为 `gate-data -> gate-train -> gate-validate -> generate -> calibrate -> detect -> report -> audit`。

**技术栈：** Python 3.11、Tree-sitter Python、PyTorch、Transformers、dataclasses、JSON/JSONL、SHA-256/HMAC、pytest、Conda 环境 `WFCLLM`，默认离线运行。

---

## 执行约束

- 本计划以已批准规格 [2026-07-15-gated-semantic-window-watermark-design.md](../specs/2026-07-15-gated-semantic-window-watermark-design.md) 为唯一设计依据；实现发现歧义时先回到规格讨论，不能自行扩大方法范围。
- 当前工作分支是 `codex/gated-semantic-window-watermark`。执行前后都必须保护现有未跟踪内容：`.aris/`、`MANIFEST.md`、`docs/design/semantic_atomic_unit_gate_final*.md`、`review-stage/`。
- 用户当前明确禁止 subagent、Git commit、push、PR 和训练运行。因此本计划中的每个任务只设置本地验证检查点，不执行 `git add` 或 `git commit`；真实门模型训练命令只写入文档和 CLI 测试，不在本轮或未经再次授权时运行。
- 实现代码必须采用测试先行：先写失败测试并确认失败原因，再写最小实现，再运行目标测试。
- 所有项目测试默认带 `HF_HUB_OFFLINE=1`；涉及 Transformers 时再带 `TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1`。
- 不能把 pass/test/correctness、运行时执行结果、静态质量分数或候选质量代理接入门模型标签、生成重试、候选选择、校准或检测。它们只能进入明确标记的 posthoc 报告。
- `inputs/final_code.jsonl` 继续是正式正样本检测器唯一输入，每行字段必须严格等于 `id`、`dataset`、`prompt`、`final_code`。
- 部署密钥、训练密钥原文、模型权重、tokenizer 文件、数据集、checkpoint、日志和大体积 JSONL 不得提交 Git。
- `HumanEval` 全量保留为最终评估集，不能进入 gate-data、gate-train 或 gate-validate 的拟合数据。
- 任何一次任务结束时都运行 `git diff --check` 与 `git status --short`，只检查，不暂存、不提交。

## 固定合同与默认值

以下值是第一版实现合同，不在执行过程中临时改动：

- 方法名：`gated_semantic_window_v1`。
- parser/window 合同版本：`python-statement-window/v1`。
- gate 输入合同版本：`wfcllm-gate-input/v1`。
- gate bundle 合同版本：`wfcllm-gate-bundle/v1`。
- gated detector 模式：`wfcllm-gated-semantic-window/v1`。
- 每个窗口最多 3 个语句单元，互不重叠，不能跨直接父级作用域。
- `return`、函数调用、`yield`、`yield from`、`await` 是允许但保守处理的候选类型；`pass`、`break`、`continue`、`raise`、`import`、`from ... import ...`、`global`、`nonlocal`、`del`、`assert`、函数定义开头、类定义开头第一版排除。
- 括号内部表达式不独立成为语句单元；复合语句开头是一个语句单元。
- 原始窗口是候选 0，始终先检查，不计入重写预算；默认最多 3 次完整窗口重写，实验预算为 1、3、6。
- gate 输出两个概率：`p_close` 与 `p_suitable`。两者都不读取密钥、目标 LSH 区域、候选日志或重试次数。
- gate 输入 tokenizer 最大 512 tokens；溢出时确定性地“关闭当前窗口并跳过嵌入”。
- gate-data 对同一起点构造 W1/W2/W3；候选 0 加 6 个按顺序保存的重写；R=1/3/6 使用同一轨迹的前缀。
- 训练密钥 32 个，验证留出密钥 8 个，部署密钥与二者隔离。
- `suitable` 目标默认要求：R=3 的训练密钥命中率不低于 0.60、结构有效重写率不低于 2/3、数值不稳定率不高于 0.10。
- gate-validate 要求：全部离散决策一致率不低于 99.9%；正式接受窗口跨验证模式一致率为 100%；`suitable` 假阳性率不高于 5%。
- posthoc 代码正确率非劣界：相对现有基线 pass@1 下降不超过 2 个百分点；该条件不能反向参与生成或检测决策。

## 文件结构与职责

### 新增共享窗口层

- `wfcllm/windowing/__init__.py`：只导出稳定公共 API。
- `wfcllm/windowing/contracts.py`：`StatementUnit`、`ParentDescriptor`、`GateScores`、`GateDecision`、`SemanticWindow`、枚举与版本常量。
- `wfcllm/windowing/python.py`：Tree-sitter 驱动的 Python 语句单元提取，处理分号、单行复合语句、复合头、多行语句和缩进。
- `wfcllm/windowing/normalization.py`：规范文本、父节点描述和 gate 输入中结构字段的序列化。
- `wfcllm/windowing/partitioner.py`：生成端与提取端共享的有状态窗口关闭逻辑、阈值策略、溢出 close-and-skip。

### 新增门模型层

- `wfcllm/gate/__init__.py`：导出配置、模型、bundle 和三阶段 pipeline。
- `wfcllm/gate/config.py`：数据、训练、验证和推理配置 dataclass。
- `wfcllm/gate/schema.py`：训练组、候选轨迹、LSH 观测、标签与 manifest 的 JSON 合同。
- `wfcllm/gate/input.py`：把最终代码可恢复信息序列化为固定 gate 输入。
- `wfcllm/gate/key_bank.py`：只加载本地训练/验证 key bank，输出不可逆 key ID，不把原文写入 artifact。
- `wfcllm/gate/labels.py`：根据同一候选轨迹的 R=1/3/6 前缀计算结构、命中、稳定性指标和监督标签。
- `wfcllm/gate/data.py`：候选窗口枚举、六次重写编排、同轨迹多前文/多预算样本生成。
- `wfcllm/gate/feasibility.py`：按独立 window group 计算 pilot 准入、预算收益、覆盖和数据规模报告。
- `wfcllm/gate/sources.py`：当前 run、MBPP、开源 Python 项目、人工边界样本的 source adapter 与按任务/仓库分组切分。
- `wfcllm/gate/dataset.py`：group-aware PyTorch Dataset、collator 和 consistency sampler。
- `wfcllm/gate/model.py`：约 30M～80M 的编码器加 close/suitable 双 head。
- `wfcllm/gate/losses.py`：分类、危险假阳性、上下文一致性、batch 一致性、量化近似一致性损失。
- `wfcllm/gate/trainer.py`：训练循环、验证、早停、指标与 checkpoint 写入；不读取部署密钥。
- `wfcllm/gate/bundle.py`：不可变 bundle manifest、文件哈希、加载校验和 formal quantized model。
- `wfcllm/gate/validation.py`：single/batch、batch 排列、CPU/GPU、float/quantized、重复运行一致性矩阵。
- `wfcllm/gate/pipeline.py`：`gate-data`、`gate-train`、`gate-validate` 的低层入口。

### 新增或扩展语义层

- 修改 `wfcllm/semantic/keying.py`：保留旧 `derive(...)`，新增基于规范父节点描述的版本化派生函数。
- 新增 `wfcllm/semantic/window_lsh.py`：直接对一个完整语义窗口计算签名、允许区域、margin、命中和稳定状态。
- 修改 `wfcllm/semantic/__init__.py`：导出新窗口 API，不改变旧规则默认行为。

### 新增 gated 生成层

- 修改 `wfcllm/generation/generator.py`：增加从窗口起点 checkpoint 教师强制 replay 已选 token 的能力，支持失败后恢复候选 0。
- 新增 `wfcllm/generation/gated_state.py`：候选 0、重写预算、命中、结构拒绝、恢复原窗口的纯状态机。
- 新增 `wfcllm/generation/gated_generator.py`：正常生成、gate 关窗、完整窗口 rollback/resample/replay、继续后文生成。
- 新增 `wfcllm/generation/window_rewriter.py`：为 gate-data 提供不含密钥的完整窗口改写 adapter。
- 新增 `wfcllm/generation/gated_pipeline.py`：批量生成严格 final-code 与 audit-only sidecar。
- 修改 `wfcllm/generation/__init__.py`：导出 gated API，同时保留现有 SAWR API。

### 新增 gated 检测层

- 修改 `wfcllm/detection/config.py`：新增 gated detector 配置，不改变现有 proxy preset。
- 新增 `wfcllm/detection/gated_windows.py`：仅从最终代码运行共享 parser、gate bundle 与 partitioner。
- 新增 `wfcllm/detection/gated_pipeline.py`：gated 校准/检测，验证 gate bundle 和 parser 合同哈希。
- 修改 `wfcllm/detection/scoring.py`：抽取可由 proxy 和 gated window 共用的只读评分协议。
- 修改 `wfcllm/detection/__init__.py`：导出 gated pipeline。

### 配置、编排、审计与文档

- 修改 `wfcllm/method/config.py`、`wfcllm/method/presets.py`：新增 gated preset，旧 preset 原样保留。
- 修改 `wfcllm/method/artifacts.py`：新增 gate 数据、模型、验证、gated generation/detection 路径。
- 修改 `wfcllm/orchestration/state.py`、`pipeline.py`、`prereq.py`：支持由方法配置给出的阶段序列及 gate 前置条件。
- 修改 `wfcllm/cli/arguments.py`、`entry.py`、`runners.py`、`config_resolver.py`：注册三项 gate 阶段并派发 gated 方法。
- 新增 `wfcllm/common/secrets.py`：从本地文件或环境变量读取密钥，禁止将原文放入配置和 artifact。
- 新增 `scripts/wfcllm_gate.py`：`data`、`train`、`validate` 低层子命令；测试只用 fake backend，不运行真实训练。
- 新增 `configs/wfcllm/gated_semantic_window_v1.json`：不含任何 key 原文。
- 修改 `wfcllm/audit/forbidden.py`、`artifact_integrity.py`、`no_quality_gate.py`：覆盖 gate artifacts、bundle hash、secret 泄漏和 forbidden quality proxy。
- 修改 `docs/ARTIFACT_SCHEMA.md`、`docs/NO_QUALITY_GATE_PROTOCOL.md`、`README.md`、`CLAUDE.md`：记录新流程、边界和基线关系。

### 新增测试目录

- `tests/windowing/`：语句单元、父节点描述、规范化和窗口可复现性。
- `tests/gate/`：schema、标签、数据、模型、损失、bundle、量化与验证。
- `tests/generation/test_gated_*.py`：完整窗口回退、重写、恢复和 final-code sidecar 隔离。
- `tests/semantic/test_window_lsh.py`：父节点描述派生和完整窗口 LSH。
- `tests/detection/test_gated_*.py`：final-code-only 重建、校准和检测。
- `tests/audit/test_gate_artifacts.py`：密钥和质量代理审计。
- `tests/integration/test_gated_workflow.py`：fake gate/fake model 的八阶段离线端到端测试。

---

### 任务 0：记录基线并锁定执行边界

**文件：**

- 读取：`CLAUDE.md`
- 读取：`AGENTS.md`
- 读取：`docs/superpowers/specs/2026-07-15-gated-semantic-window-watermark-design.md`
- 读取：`docs/ARTIFACT_SCHEMA.md`
- 读取：`docs/NO_QUALITY_GATE_PROTOCOL.md`
- 不修改任何实现文件

- [ ] **步骤 1：确认分支和用户文件**

运行：

```bash
git branch --show-current
git status --short
```

预期：当前分支为 `codex/gated-semantic-window-watermark`；记录所有既有未跟踪文件，后续任务不得删除、移动或覆盖它们。

- [ ] **步骤 2：运行当前主线的快速基线**

运行：

```bash
conda run -n WFCLLM python run.py --status
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/method tests/generation tests/semantic tests/detection tests/audit tests/integration -q
```

预期：保存退出码与首个失败；若失败来自当前未改代码，记录为基线失败，不在本任务顺手修复。

- [ ] **步骤 3：确认没有执行被禁止操作**

不得运行训练、Git 暂存、commit、push 或 PR 命令。运行：

```bash
git diff --check
git status --short
```

预期：除已知用户文件和已批准 spec/plan 外，没有本任务产生的新修改。

---

### 任务 1：建立共享语句单元合同和 Python 提取器

**文件：**

- 创建：`wfcllm/windowing/__init__.py`
- 创建：`wfcllm/windowing/contracts.py`
- 创建：`wfcllm/windowing/python.py`
- 创建：`tests/windowing/test_python_units.py`
- 参考但不修改：`wfcllm/lang/python/parser.py`

- [ ] **步骤 1：先写语句单元失败测试**

测试必须覆盖以下固定事实：

```python
from __future__ import annotations

from wfcllm.windowing.python import PythonStatementUnitExtractor


def texts(source: str) -> list[str]:
    return [unit.text for unit in PythonStatementUnitExtractor().extract(source)]


def test_semicolon_and_inline_if_are_parser_units() -> None:
    assert texts("def f(c):\n    x = 1; y = 2\n    if c: a = 3; b = 4\n") == [
        "def f(c):",
        "x = 1",
        "y = 2",
        "if c:",
        "a = 3",
        "b = 4",
    ]


def test_parenthesized_expression_is_not_split() -> None:
    assert texts("def f(a, b):\n    x = (a +\n         b)\n    return x\n") == [
        "def f(a, b):",
        "x = (a +\n         b)",
        "return x",
    ]


def test_return_is_eligible_but_fixed_exclusions_are_not() -> None:
    units = PythonStatementUnitExtractor().extract(
        "def f(x):\n    import os\n    assert x\n    return x\n"
    )
    assert [(unit.node_type, unit.eligible) for unit in units] == [
        ("function_definition", False),
        ("import_statement", False),
        ("assert_statement", False),
        ("return_statement", True),
    ]
```

另加参数化测试覆盖 `if/elif/else`、`for/while`、`try/except/finally`、`with`、`match/case`、`async for/with`、函数调用、`yield/yield from`、`await`、函数头、类头、多层缩进和注释/空行忽略；函数头和类头必须 `eligible=False`，调用/yield/await/return 必须 `eligible=True`。

- [ ] **步骤 2：运行测试确认按预期失败**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/windowing/test_python_units.py -v
```

预期：FAIL，原因是 `wfcllm.windowing` 尚不存在，而不是 Tree-sitter 或环境缺失。

- [ ] **步骤 3：实现稳定合同**

`wfcllm/windowing/contracts.py` 至少定义：

```python
from __future__ import annotations

from dataclasses import dataclass

WINDOW_CONTRACT_VERSION = "python-statement-window/v1"


@dataclass(frozen=True)
class StatementUnit:
    unit_id: str
    node_type: str
    text: str
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    depth: int
    parent_path: tuple[str, ...]
    direct_parent_type: str
    direct_child_ordinal: int
    eligible: bool
    hard_boundary: bool
    compound_header: bool

    def __post_init__(self) -> None:
        if self.start_byte < 0 or self.end_byte <= self.start_byte:
            raise ValueError("statement unit must have a non-empty byte span")
```

`wfcllm/windowing/python.py` 实现 `PythonStatementUnitExtractor.extract(source, *, function_name=None)`，规则必须是：

1. 只依赖 Tree-sitter 的命名节点和字节区间，不按字符串分号切分；
2. 普通语句按 parser 的直接语句子节点产生单元；
3. 复合节点、函数定义和类定义从起点截到真实 header 冒号，body 中语句递归产生后续单元；函数/类 header 仍产生单元，但标记为不可承载硬边界；
4. 括号、下标、调用参数和表达式内部节点永不递归成独立单元；
5. `elif_clause`、`else_clause`、`except_clause`、`finally_clause`、`case_clause` 各自产生 header 单元；
6. 使用 byte span 从 UTF-8 源码取文本，保证中文字符串不会错位；
7. 计算直接父级路径、缩进深度和同父级 ordinal；
8. 仅把固定排除类型标为 `eligible=False`，仍保留在上下文序列中，并标记为窗口硬分隔点；复合语句 header 标记 `compound_header=True`，第一版只能单独形成一单元窗口。

- [ ] **步骤 4：运行目标测试和现有 parser 回归**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/windowing/test_python_units.py tests/lang -v
```

预期：新测试全通过；现有 `wfcllm.lang` 行为不变。

- [ ] **步骤 5：本地检查点**

```bash
git diff --check
git status --short
```

不得暂存或提交。

---

### 任务 2：规范父节点描述与 gate 输入序列化

**文件：**

- 修改：`wfcllm/windowing/contracts.py`
- 创建：`wfcllm/windowing/normalization.py`
- 创建：`wfcllm/gate/__init__.py`
- 创建：`wfcllm/gate/input.py`
- 创建：`tests/windowing/test_normalization.py`
- 创建：`tests/gate/test_input.py`

- [ ] **步骤 1：写规范化失败测试**

```python
from __future__ import annotations

from wfcllm.gate.input import GateInput, serialize_gate_input
from wfcllm.windowing.contracts import ParentDescriptor


def test_parent_descriptor_has_one_canonical_string() -> None:
    descriptor = ParentDescriptor(
        contract_version="python-statement-window/v1",
        ancestor_node_types=("module", "function_definition", "if_statement"),
        direct_parent_type="block",
        first_unit_ordinal=2,
        compound_header_role="body",
    )
    assert descriptor.canonical == (
        "python-statement-window/v1|module/function_definition/if_statement|"
        "parent=block|ordinal=2|role=body"
    )


def test_gate_input_contains_no_key_or_target_region() -> None:
    payload = serialize_gate_input(
        GateInput(
            normalization_version="wfcllm-window-normalization/v1",
            parent_descriptor="v1|function|parent=block|ordinal=0|role=body",
            depth=1,
            previous_units=("x = 1",),
            previous_unit_types=("assignment",),
            current_units=("return x",),
            current_unit_types=("return_statement",),
            current_unit_count=1,
            current_token_count=2,
        )
    )
    assert "[CURRENT]\n[S type=return_statement] return x" in payload
    assert "secret" not in payload.lower()
    assert "lsh" not in payload.lower()
    assert "retry" not in payload.lower()
```

增加测试确保 CRLF/LF、尾随空格和 UTF-8 字符在规范化后得到相同 canonical text，但字符串字面量内部空白不被改变。

- [ ] **步骤 2：运行并确认失败**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/windowing/test_normalization.py tests/gate/test_input.py -v
```

预期：FAIL，缺少 `ParentDescriptor`、`GateInput` 或序列化器。

- [ ] **步骤 3：实现规范化与输入合同**

`ParentDescriptor` 只包含最终代码可恢复字段，并用属性生成 canonical string：

```python
@dataclass(frozen=True)
class ParentDescriptor:
    contract_version: str
    ancestor_node_types: tuple[str, ...]
    direct_parent_type: str
    first_unit_ordinal: int
    compound_header_role: str

    @property
    def canonical(self) -> str:
        ancestors = "/".join(self.ancestor_node_types)
        return (
            f"{self.contract_version}|{ancestors}|parent={self.direct_parent_type}|"
            f"ordinal={self.first_unit_ordinal}|role={self.compound_header_role}"
        )
```

`GateInput` 固定为规范化版本、父节点描述、缩进深度、最多三个前文语句及其 parser 类型、当前开放窗口及其 parser 类型、当前单元数和当前窗口 token 数；`serialize_gate_input` 使用版本化纯文本标签。`current_token_count` 由 bundle tokenizer 只对规范化当前窗口计算，不能由生成模型 token 数替代：

```text
[CONTRACT] wfcllm-gate-input/v1
[NORMALIZATION] wfcllm-window-normalization/v1
[PARENT] ...
[DEPTH] 1
[CURRENT_UNIT_COUNT] 1
[CURRENT_TOKEN_COUNT] 2
[PREVIOUS]
[S type=assignment] x = 1
[CURRENT]
[S type=return_statement] return x
```

禁止加入 key、LSH bit、候选序号、重试次数、测试结果、batch 信息或设备信息。

- [ ] **步骤 4：运行目标测试**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/windowing/test_normalization.py tests/gate/test_input.py -v
```

预期：PASS。

- [ ] **步骤 5：本地检查点**

```bash
git diff --check
git status --short
```

不得暂存或提交。

---

### 任务 3：实现生成端与提取端共享的窗口划分器

**文件：**

- 修改：`wfcllm/windowing/contracts.py`
- 创建：`wfcllm/windowing/partitioner.py`
- 修改：`wfcllm/windowing/__init__.py`
- 创建：`tests/windowing/test_partitioner.py`

- [ ] **步骤 1：写确定性划分失败测试**

使用不依赖真实模型的 `FakeGatePredictor`，验证同一语句序列两次划分完全相同：

```python
from __future__ import annotations

from wfcllm.windowing.contracts import GateScores
from wfcllm.windowing.partitioner import GateThresholds, WindowPartitioner


class FakeGatePredictor:
    def __init__(self, scores: list[GateScores]) -> None:
        self._scores = iter(scores)

    def predict(self, _gate_input: str) -> GateScores:
        return next(self._scores)


def test_non_overlapping_windows_close_at_gate_or_three_units(units) -> None:
    thresholds = GateThresholds(close_low=0.45, close_high=0.70, suitable_accept=0.80)
    result = WindowPartitioner(
        predictor=FakeGatePredictor([
            GateScores(0.20, 0.90, stable=True, precision_delta=0.01),
            GateScores(0.85, 0.88, stable=True, precision_delta=0.01),
            GateScores(0.10, 0.99, stable=True, precision_delta=0.01),
        ]),
        thresholds=thresholds,
        tokenizer_counter=lambda _text: 32,
    ).partition(units[:5])

    assert [window.unit_count for window in result.windows] == [2, 3]
    assert result.windows[0].suitable is True
    assert result.windows[0].units[-1].end_byte < result.windows[1].units[0].start_byte
```

另加测试覆盖：不同父级强制关窗、复合 header 强制单单元关窗、排除语句作为硬分隔点且保留在后续上下文、512 token 溢出 close-and-skip、分数落在阈值不确定带时 skip、第三个单元强制关闭、输入中只保留前三个前文单元。

- [ ] **步骤 2：运行并确认失败**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/windowing/test_partitioner.py -v
```

预期：FAIL，缺少 partitioner。

- [ ] **步骤 3：实现纯状态机 API**

至少提供以下合同：

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol


class GatePredictor(Protocol):
    def predict(self, serialized_input: str) -> GateScores: ...


@dataclass(frozen=True)
class GateThresholds:
    close_low: float
    close_high: float
    suitable_accept: float
    max_units: int = 3
    max_input_tokens: int = 512


class WindowPartitioner:
    def __init__(
        self,
        predictor: GatePredictor,
        thresholds: GateThresholds,
        tokenizer_counter: Callable[[str], int],
    ) -> None: ...

    def partition(self, units: list[StatementUnit]) -> PartitionResult: ...
```

`GateScores` 固定包含 formal quantized 的 close/suitable 概率、与 bundle 内 float reference 的最大概率差、离散决策是否一致，以及最终 `stable`。划分器每读入一个新单元就序列化一次开放窗口；当跨父级、复合 header、达到三单元、`p_close >= close_high` 或输入溢出时关窗。`p_close <= close_low` 时继续，二者之间是边界不确定：立即关窗并 close-and-skip。遇到 `hard_boundary=True` 时，先在该单元之前关闭已有开放窗口，再把硬边界单元记录为 skipped context，绝不把它加入正式窗口，然后从下一单元重新开始；后续 gate 的最近三个前文仍可包含它。复合 header 无论 gate 分数如何都只能形成一单元窗口。`GateScores.stable=False` 时同样立即 close-and-skip。正式 `suitable=True` 还必须同时满足：所有单元 eligible、`p_suitable >= suitable_accept`、close 不是不确定状态、没有溢出。窗口一旦关闭，下一窗口从下一个单元开始，禁止滑动或重叠。

- [ ] **步骤 4：验证生成/提取共用同一对象路径**

测试中分别从“generation adapter”和“detection adapter”调用同一个 `WindowPartitioner.partition`，断言 window byte spans、parent descriptor 和 suitable 位完全相同；不得复制一份检测专用关窗逻辑。

- [ ] **步骤 5：运行目标测试**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/windowing -v
```

预期：PASS。

- [ ] **步骤 6：本地检查点**

```bash
git diff --check
git status --short
```

不得暂存或提交。

---

### 任务 4：为完整窗口增加版本化语义 LSH 评分

**文件：**

- 修改：`wfcllm/semantic/keying.py`
- 创建：`wfcllm/semantic/window_lsh.py`
- 修改：`wfcllm/semantic/__init__.py`
- 创建：`tests/semantic/test_window_lsh.py`
- 创建：`tests/semantic/test_keying.py`
- 回归：`tests/semantic/test_rules.py`

- [ ] **步骤 1：写新派生合同失败测试**

```python
from __future__ import annotations

from wfcllm.semantic.keying import WatermarkKeying


def test_descriptor_derivation_is_versioned_and_deterministic() -> None:
    keying = WatermarkKeying("deployment-secret", d=4)
    first = keying.derive_descriptor(
        contract_version="python-statement-window/v1",
        parent_descriptor="python-statement-window/v1|module/function_definition|parent=block|ordinal=1|role=body",
        k=4,
    )
    second = keying.derive_descriptor(
        contract_version="python-statement-window/v1",
        parent_descriptor="python-statement-window/v1|module/function_definition|parent=block|ordinal=1|role=body",
        k=4,
    )
    assert first == second
    assert first != keying.derive_descriptor(
        contract_version="python-statement-window/v1",
        parent_descriptor="python-statement-window/v1|module/function_definition|parent=block|ordinal=2|role=body",
        k=4,
    )
```

再用 fake verifier/LSH space 测试候选窗口文本整体进入 encoder，命中、margin、边界不确定三态输出正确。

- [ ] **步骤 2：运行并确认失败**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/semantic/test_window_lsh.py tests/semantic/test_keying.py tests/semantic/test_rules.py -v
```

预期：新测试 FAIL；旧测试继续 PASS。

- [ ] **步骤 3：保留旧 API 并新增窗口 API**

`WatermarkKeying.derive(...)` 的字符串输入、`k/ordinal` 语义和 `frozenset[tuple[int, ...]]` 输出保持不变。先把现有“由 HMAC digest 执行确定性 Fisher-Yates 并取前 k 个签名”的代码抽成私有 `_derive_from_message(message: bytes, k: int)`，旧方法与新增方法都调用它。新增：

```python
def derive_descriptor(
    self,
    *,
    contract_version: str,
    parent_descriptor: str,
    k: int,
) -> frozenset[tuple[int, ...]]:
    payload = f"window-descriptor\0{contract_version}\0{parent_descriptor}".encode("utf-8")
    return self._derive_from_message(payload, k)
```

`wfcllm/semantic/window_lsh.py` 定义：

```python
@dataclass(frozen=True)
class SemanticWindowEvidence:
    signature: tuple[int, ...]
    allowed_region_id: str
    hit: bool
    margin: float
    stable: bool


class SemanticWindowScorer:
    def score(self, *, window_text: str, parent_descriptor: str) -> SemanticWindowEvidence: ...
```

`allowed_region_id` 必须是不可逆标识，不得写入密钥或派生 key 原文。窗口文本由全部单元按规范换行连接，不单独给每个语句打 bit。

- [ ] **步骤 4：运行语义测试**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/semantic -v
```

预期：新旧测试全部 PASS，旧 `SemanticLshEmbeddingRule` 输出不变。

- [ ] **步骤 5：本地检查点**

```bash
git diff --check
git status --short
```

不得暂存或提交。
### 任务 5：定义 gate-data 的无泄漏 artifact 合同与 key bank

**文件：**

- 创建：`wfcllm/gate/config.py`
- 创建：`wfcllm/gate/schema.py`
- 创建：`wfcllm/gate/key_bank.py`
- 创建：`tests/gate/test_schema.py`
- 创建：`tests/gate/test_key_bank.py`
- 修改：`.gitignore`

- [ ] **步骤 1：写 JSON 合同和密钥泄漏失败测试**

```python
from __future__ import annotations

import json

import pytest

from wfcllm.gate.key_bank import TrainingKeyBank
from wfcllm.gate.schema import CandidateObservation, GateTrainingGroup


def test_candidate_observation_uses_allowed_fields_only() -> None:
    row = CandidateObservation(
        candidate_index=0,
        code="x += item",
        parse_status="ok",
        unit_count=1,
        same_parent_scope=True,
        boundary_span=(0, 9),
        stable_across_precision_modes=True,
        stable_across_batch_modes=True,
        lsh_by_key_id={"train-key-000": {"hit": True, "margin": 0.23, "stable": True}},
        generation_seed_id="seed-plan-v1:000",
        rewrite_config_id="rewrite-v1",
    ).to_dict()
    serialized = json.dumps(row, sort_keys=True)
    assert "syntax_valid" not in serialized
    assert "pass" not in serialized
    assert "correctness" not in serialized


def test_key_bank_never_serializes_raw_keys(tmp_path) -> None:
    key_file = tmp_path / "keys.json"
    key_file.write_text(json.dumps({"keys": ["alpha", "beta"]}), encoding="utf-8")
    bank = TrainingKeyBank.load(key_file, expected_count=2)
    manifest = json.dumps(bank.public_manifest(), sort_keys=True)
    assert "alpha" not in manifest
    assert "beta" not in manifest
    assert bank.key_ids == ("train-key-000", "train-key-001")


def test_deployment_key_cannot_be_loaded_as_training_key(tmp_path) -> None:
    with pytest.raises(ValueError, match="deployment"):
        TrainingKeyBank.from_records(
            [{"id": "deployment-key", "material": "forbidden"}], expected_count=1
        )
```

- [ ] **步骤 2：运行并确认失败**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/gate/test_schema.py tests/gate/test_key_bank.py -v
```

预期：FAIL，缺少 schema/config/key bank。

- [ ] **步骤 3：实现版本化 schema**

schema 至少包含以下不可变记录：

```python
@dataclass(frozen=True)
class CandidateObservation:
    candidate_index: int
    code: str
    parse_status: str
    unit_count: int
    same_parent_scope: bool
    boundary_span: tuple[int, int]
    stable_across_precision_modes: bool
    stable_across_batch_modes: bool
    lsh_by_key_id: dict[str, dict[str, bool | float]]
    generation_seed_id: str
    rewrite_config_id: str


@dataclass(frozen=True)
class GateTrainingGroup:
    schema_version: str
    group_id: str
    source_id: str
    source_family: str
    repository_group: str
    function_group: str
    language: str
    parser_contract_version: str
    split: str
    window_start_unit_id: str
    parent_descriptor: str
    candidate_window_lengths: tuple[int, ...]
    previous_units: tuple[str, ...]
    candidates_by_length: dict[str, tuple[CandidateObservation, ...]]
```

所有 `to_dict()` 都要调用现有 `reject_quality_proxy_fields`；不得使用已被 no-quality-gate 禁止的字段名。`parse_status` 只描述 parser/范围事实，取值固定为 `ok`、`parse_error`、`scope_changed`、`unit_count_out_of_range`。正式 writer 把 group 元数据写入 `window_groups.jsonl`，把每个 `window_length + candidate_index` 展平写入 `candidate_attempts.jsonl`，把派生监督写入 `labels.jsonl`；三者用 `group_id` 关联，不把七个候选嵌进一个不可流式读取的大 JSON 行。

- [ ] **步骤 4：实现配置与 key bank**

`GateDataConfig` 固定训练 key 数 32、holdout key 数 8、rewrite 数 6、预算 `(1, 3, 6)`；`GateTrainConfig` 固定 max tokens 512 和 base model 本地路径；`GateValidateConfig` 固定一致性阈值。

`TrainingKeyBank` 只在进程内保存原始字节，public manifest 只能包含：schema 版本、bank ID、key 数量、每个 key 的不可逆 ID 和 key 文件 SHA-256。把以下路径加入 `.gitignore`：

```gitignore
data/gate_keys/
data/gate_datasets/
data/gate_models/
data/runs/*/gate-data/
data/runs/*/gate-train/
data/runs/*/gate-validate/
```

- [ ] **步骤 5：运行测试并检查 JSON**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/gate/test_schema.py tests/gate/test_key_bank.py tests/method/test_no_quality_gate_contract.py -v
```

预期：PASS；序列化记录不含 raw key 和 forbidden quality 字段。

- [ ] **步骤 6：本地检查点**

```bash
git diff --check
git status --short
```

不得暂存或提交。

---

### 任务 6：用同一候选轨迹生成预算标签和 gate 监督目标

**文件：**

- 创建：`wfcllm/gate/labels.py`
- 创建：`tests/gate/test_labels.py`

- [ ] **步骤 1：写轨迹前缀和标签失败测试**

```python
from __future__ import annotations

from wfcllm.gate.labels import LabelThresholds, build_gate_labels


def test_r1_r3_r6_are_prefixes_of_one_candidate_trajectory(observation_factory) -> None:
    candidates = tuple(
        observation_factory(index=index, hit_keys=range(20 if index in {1, 3} else 4))
        for index in range(7)
    )
    labels = build_gate_labels(
        candidates,
        training_key_count=32,
        thresholds=LabelThresholds(
            r3_hit_rate=0.60,
            structural_valid_rate=2 / 3,
            unstable_rate=0.10,
        ),
    )
    assert labels.budgets[1].candidate_indices == (0, 1)
    assert labels.budgets[3].candidate_indices == (0, 1, 2, 3)
    assert labels.budgets[6].candidate_indices == (0, 1, 2, 3, 4, 5, 6)
    assert labels.suitable_target is True


def test_candidate_zero_is_checked_but_not_counted_as_rewrite(observation_factory) -> None:
    candidates = (observation_factory(index=0, hit_keys=range(32)),) + tuple(
        observation_factory(index=index, hit_keys=()) for index in range(1, 7)
    )
    labels = build_gate_labels(candidates, training_key_count=32)
    assert labels.budgets[1].rewrite_count == 1
    assert labels.budgets[1].success_rate == 1.0
```

增加测试：R3 命中率 0.59 时 suitable 为假；结构有效率不足 2/3 时为假；不稳定率高于 0.10 时为假；不同 key 的“首次命中候选序号”按 0～6 记录；没有命中时为 `None`；`close_target` 在硬结构边界、长度达到 3 或当前开放窗口已经满足 `suitable_target` 时为真，否则继续。close 标签只能使用当前已经生成的窗口，不得查看未来语句。

- [ ] **步骤 2：运行并确认失败**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/gate/test_labels.py -v
```

预期：FAIL，缺少标签实现。

- [ ] **步骤 3：实现可审计标签计算**

```python
@dataclass(frozen=True)
class BudgetOutcome:
    budget: int
    candidate_indices: tuple[int, ...]
    rewrite_count: int
    success_rate: float
    structural_valid_rate: float
    unstable_rate: float
    first_hit_by_key_id: dict[str, int | None]


@dataclass(frozen=True)
class GateLabels:
    close_target: bool
    suitable_target: bool
    budgets: dict[int, BudgetOutcome]
```

计算顺序固定为：先筛结构可用候选，再按每个训练 key 判断 `hit and stable and margin >= configured_margin`，最后对 key 取“候选 0 加给定预算前缀内至少一个候选命中”的比例。`structural_valid_rate` 和 `unstable_rate` 只统计候选 1～R，不能让天然可解析的候选 0 抬高重写能力；`success_rate` 才包含候选 0。绝不能把候选代码的测试结果、参考答案相似度或人工质量评分写进 formal label。

- [ ] **步骤 4：运行目标测试**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/gate/test_labels.py tests/gate/test_schema.py -v
```

预期：PASS。

- [ ] **步骤 5：本地检查点**

```bash
git diff --check
git status --short
```

不得暂存或提交。

---

### 任务 7：实现 gate-data 的候选生成、数据来源与防泄漏切分

**文件：**

- 创建：`wfcllm/gate/data.py`
- 创建：`wfcllm/gate/sources.py`
- 创建：`tests/gate/test_data.py`
- 创建：`tests/gate/test_sources.py`
- 参考：`wfcllm/datasets/`
- 参考：`wfcllm/generation/pipeline.py`

- [ ] **步骤 1：写同起点、同轨迹失败测试**

```python
from __future__ import annotations

from wfcllm.gate.data import GateDataBuilder


class RecordingRewriter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []

    def rewrite(self, request, *, candidate_index: int):
        self.calls.append((request.window_start_unit_id, request.window_length, candidate_index))
        return request.fake_candidate(candidate_index)


def test_w1_w2_w3_each_get_candidate_zero_plus_six_ordered_rewrites(sample_units) -> None:
    rewriter = RecordingRewriter()
    groups = GateDataBuilder(rewriter=rewriter, lsh_probe=FakeLshProbe()).build(sample_units)
    first = groups[0]
    assert first.candidate_window_lengths == (1, 2, 3)
    for length in (1, 2, 3):
        assert [item.candidate_index for item in first.candidates_by_length[str(length)]] == list(range(7))
    assert all(call[0] == first.window_start_unit_id for call in rewriter.calls[:18])
```

增加测试确保 rewrite request 不含 key、目标 region、LSH bit 或首次命中位置；前文 1/2/3 版本复用同一 `candidates_by_length`，不重新调用 rewriter；候选 0 是原窗口；改写后 1～3 单元且同父级才进入 LSH probe。

- [ ] **步骤 2：写切分失败测试**

```python
def test_all_context_and_budget_variants_stay_in_one_split(splitter, sample_group) -> None:
    variants = sample_group.expand_contexts_and_budgets()
    assert len({splitter.assign(item) for item in variants}) == 1


def test_same_function_and_repository_never_cross_splits(splitter, repository_groups) -> None:
    assignments = splitter.assign_all(repository_groups)
    for group_id, split_names in assignments.by_repository_group().items():
        assert len(split_names) == 1, group_id


def test_humaneval_is_rejected_from_formal_gate_data(source_loader) -> None:
    with pytest.raises(ValueError, match="HumanEval.*holdout"):
        source_loader.load(source_family="humaneval")
```

- [ ] **步骤 3：运行并确认失败**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/gate/test_data.py tests/gate/test_sources.py -v
```

预期：FAIL，缺少 data/source 实现。

- [ ] **步骤 4：实现接口和数据路径**

`data.py` 定义无密钥 rewriter 协议与 LSH probe 协议：

```python
class WindowRewriter(Protocol):
    def rewrite(self, request: RewriteRequest, *, candidate_index: int) -> RewriteCandidate: ...


class MultiKeyLshProbe(Protocol):
    def probe(
        self,
        *,
        window_text: str,
        parent_descriptor: str,
        key_ids: tuple[str, ...],
    ) -> dict[str, LshProbeResult]: ...
```

正式 `MultiKeyLshProbe` 对同一候选运行冻结语义 encoder 的单样本、不同 batch 邻居和批准的 float/量化模式，分别记录签名、margin 与一致性；只有规定模式下签名/命中一致且 margin 达标时，`reliable_hit_by_training_key_id` 才为真。该稳定性检查属于语义数值合同，不是代码质量检查。

`RewriteRequest` 只包含任务 prompt、已完成前缀、原窗口、规范父级和结构边界；其 `to_dict()` 遇到 `key`、`lsh`、`target_region` 字段立即报错。对每个合法起点依次构造 W1、W2、W3，遇到父级/缩进变化、硬分隔点或复合 header 单例限制就停止扩展；每个固定长度各自只生成一次候选 0～6 轨迹；之后从相同 group 派生前文长度 1/2/3 和预算 1/3/6 的训练视图。

`sources.py` 支持四类来源：现有 main 生成结果、MBPP train/validation、明确许可证的本地开源 Python 快照、人工 parser 边界样本。开源项目是完整数据的主要独立 group 来源；完整 manifest 必须覆盖至少三个不同代码生成模型，source model ID 只进入分层 metadata，不进入 gate 输入。人工样本标记为 `contract_or_hard_set=true`，只用于 parser 合同和困难集报告，不用于估计正式标签比例或拟合正式阈值。HumanEval、部署检测集和最终负样本校准集拒绝加载。

切分键优先级固定为：repository ID > task ID > function ID；使用 `sha256(seed + group_id)` 确定 train/validation/test，不能按行随机切分。

- [ ] **步骤 5：运行 gate-data 单元测试**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/gate/test_data.py tests/gate/test_sources.py tests/windowing -v
```

预期：PASS；fake rewriter 的调用数精确等于每个起点 `3 * 6`，上下文/预算展开不会增加调用。

- [ ] **步骤 6：本地检查点**

```bash
git diff --check
git status --short
```

不得暂存或提交。

---

### 任务 8：实现 group-aware 数据集和 close/suitable 双头门模型

**文件：**

- 创建：`wfcllm/gate/dataset.py`
- 创建：`wfcllm/gate/model.py`
- 创建：`tests/gate/test_dataset.py`
- 创建：`tests/gate/test_model.py`

- [ ] **步骤 1：写 dataset 和模型失败测试**

```python
from __future__ import annotations

import torch

from wfcllm.gate.model import GateModel, GateModelOutput


def test_gate_model_returns_two_logits(fake_encoder) -> None:
    model = GateModel(encoder=fake_encoder, hidden_size=16)
    output = model(
        input_ids=torch.ones((4, 12), dtype=torch.long),
        attention_mask=torch.ones((4, 12), dtype=torch.long),
    )
    assert isinstance(output, GateModelOutput)
    assert output.close_logits.shape == (4,)
    assert output.suitable_logits.shape == (4,)


def test_collator_keeps_group_and_variant_ids(gate_examples, fake_tokenizer) -> None:
    batch = GateCollator(fake_tokenizer, max_tokens=512)(gate_examples)
    assert batch["input_ids"].shape[0] == len(gate_examples)
    assert batch["group_ids"] == [example.group_id for example in gate_examples]
    assert batch["context_lengths"] == [example.context_length for example in gate_examples]
```

增加测试：超过 512 token 的样本标记为 `overflow=True` 且 target 强制 close/skip，不静默截断当前窗口；同 group 的不同上下文版本由 sampler 放进同一 batch；source family/model ID 不进入 tokenizer text。

- [ ] **步骤 2：运行并确认失败**

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/gate/test_dataset.py tests/gate/test_model.py -v
```

预期：FAIL，缺少 Dataset/Model。

- [ ] **步骤 3：实现最小双头模型**

```python
@dataclass(frozen=True)
class GateModelOutput:
    close_logits: torch.Tensor
    suitable_logits: torch.Tensor


class GateModel(nn.Module):
    def __init__(self, *, encoder: nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.close_head = nn.Linear(hidden_size, 1)
        self.suitable_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> GateModelOutput:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return GateModelOutput(
            close_logits=self.close_head(pooled).squeeze(-1),
            suitable_logits=self.suitable_head(pooled).squeeze(-1),
        )
```

正式 loader 使用 `AutoModel.from_pretrained(local_files_only=True)`；base model 路径由 config 提供，默认目标规模 30M～80M，不能在单元测试中下载模型。

- [ ] **步骤 4：实现 group-aware Dataset**

每条 `GateExample` 包含 `group_id`、`window_start_unit_id`、`context_length`、`budget`、serialized gate input、两个 target 和危险负例标志。`GroupConsistencyBatchSampler` 至少把同 group 的 1/2/3 前文变体放进同一 batch；R=1/3/6 只作为监督分析字段，不能告诉 gate 当前生成会给多少次重写。

- [ ] **步骤 5：运行目标测试**

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/gate/test_dataset.py tests/gate/test_model.py -v
```

预期：PASS，测试只使用 fake encoder/tokenizer。

- [ ] **步骤 6：本地检查点**

```bash
git diff --check
git status --short
```

不得暂存或提交。

---

### 任务 9：实现可计算的第一版损失与训练循环

**文件：**

- 创建：`wfcllm/gate/losses.py`
- 创建：`wfcllm/gate/trainer.py`
- 创建：`tests/gate/test_losses.py`
- 创建：`tests/gate/test_trainer.py`

- [ ] **步骤 1：写损失分量失败测试**

```python
from __future__ import annotations

import torch

from wfcllm.gate.losses import GateLoss, GateLossWeights


def test_false_positive_penalty_is_larger_for_dangerous_negative() -> None:
    loss_fn = GateLoss(GateLossWeights(suitable_false_positive=4.0))
    logits = torch.tensor([2.0, 2.0])
    targets = torch.tensor([0.0, 0.0])
    ordinary = loss_fn.suitable(logits[:1], targets[:1], torch.tensor([False]))
    dangerous = loss_fn.suitable(logits[1:], targets[1:], torch.tensor([True]))
    assert dangerous > ordinary


def test_context_consistency_penalizes_same_group_disagreement() -> None:
    loss_fn = GateLoss(GateLossWeights(context_consistency=1.0))
    same = loss_fn.consistency(torch.tensor([0.1, 0.1]), ["g", "g"])
    different = loss_fn.consistency(torch.tensor([0.1, 0.9]), ["g", "g"])
    assert same == 0
    assert different > same


def test_fake_quant_consistency_is_finite() -> None:
    value = GateLoss().quantization_consistency(torch.tensor([-2.3, 0.2, 4.7]))
    assert torch.isfinite(value)
```

- [ ] **步骤 2：写训练器失败测试**

用 8 条 fake examples 和 tiny fake encoder 运行一个 epoch，断言只写 `checkpoints/last.pt`、`checkpoints/best.pt`、`training_metrics.jsonl`、`development_summary.json`，其中不含 key 原文、训练代码全文或部署密钥。

- [ ] **步骤 3：运行并确认失败**

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/gate/test_losses.py tests/gate/test_trainer.py -v
```

预期：FAIL，缺少 losses/trainer。

- [ ] **步骤 4：实现第一版总损失**

第一版只实现可由现有正式数据直接计算的六项：

1. close 的加权 BCE；
2. suitable 的加权 BCE；
3. 危险负例 suitable 假阳性额外权重；
4. 同 group 不同前文长度的概率一致性；
5. 同一 serialized input 放入不同 batch 邻居和 padding 组合时的概率一致性；
6. logits 与其 fake-quantized 版本的一致性。

不加入“预测重试次数”“代码破坏风险”“候选排序”head。`GateLoss.forward` 返回总 loss 和命名分量字典，训练 metrics 必须逐项记录。

fake quantization 使用固定对称 int8 scale 与 straight-through estimator，只作为训练期近似；是否可以成为 formal bundle 仍由任务 10 的真实量化验证决定。

- [ ] **步骤 5：实现训练循环**

`GateTrainer.fit()` 必须：设置 Python/NumPy/Torch seed；使用 group sampler；为 batch-consistency 分支把同一 serialized input 放入两种固定邻居/padding 组合并比较输出；按 validation suitable false-positive 和决策一致性早停；保存 optimizer/scheduler state；每次加载 checkpoint 校验 config hash 和 dataset manifest hash；不把 key bank 内容写入 checkpoint。batch 一致性测试若发现 padding mask、pooling、train/eval mode 或未初始化状态错误，必须先修工程缺陷，不能用损失掩盖。

训练命令在本任务只通过 fake 单元测试调用。不得运行真实 `gate-train`。

- [ ] **步骤 6：运行目标测试**

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/gate/test_losses.py tests/gate/test_trainer.py -v
```

预期：PASS。

- [ ] **步骤 7：本地检查点**

```bash
git diff --check
git status --short
```

不得暂存或提交。

---

### 任务 10：打包 float/quantized gate 并执行可复现性验收

**文件：**

- 创建：`wfcllm/gate/bundle.py`
- 创建：`wfcllm/gate/validation.py`
- 创建：`tests/gate/test_bundle.py`
- 创建：`tests/gate/test_validation.py`

- [ ] **步骤 1：写 bundle 哈希失败测试**

```python
from __future__ import annotations

import pytest

from wfcllm.gate.bundle import GateBundle


def test_bundle_rejects_tampered_model_file(fake_bundle_dir) -> None:
    bundle = GateBundle.load(fake_bundle_dir)
    assert bundle.manifest.contract_version == "wfcllm-gate-bundle/v1"
    (fake_bundle_dir / "gate_float.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        GateBundle.load(fake_bundle_dir)


def test_bundle_manifest_binds_parser_tokenizer_and_thresholds(fake_bundle_dir) -> None:
    manifest = GateBundle.load(fake_bundle_dir).manifest
    assert manifest.window_contract_version == "python-statement-window/v1"
    assert manifest.gate_input_contract_version == "wfcllm-gate-input/v1"
    assert manifest.tokenizer_sha256
    assert manifest.float_model_sha256
    assert manifest.quantized_model_sha256
    assert manifest.close_low_threshold
    assert manifest.close_high_threshold
    assert manifest.suitable_accept_threshold
```

- [ ] **步骤 2：写验证矩阵失败测试**

用 deterministic fake predictors 构造 single、不同 batch size、batch 重排、CPU float、CPU quantized、两次重复运行的输出。断言验证器计算：全部决策一致率、accepted-window 一致率、suitable 假阳性率，并在任一硬阈值不达标时拒绝 `validated=true`。

- [ ] **步骤 3：运行并确认失败**

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/gate/test_bundle.py tests/gate/test_validation.py -v
```

预期：FAIL，缺少 bundle/validation。

- [ ] **步骤 4：实现不可变 bundle**

bundle 目录固定包含：

```text
manifest.json
gate_float.pt
gate_int8.pt
tokenizer/
validation_summary.json
```

`manifest.json` 只保存版本、文件哈希、模型架构、本地 base model ID、tokenizer hash、阈值、max tokens、训练数据 manifest hash、训练/holdout key bank 的不可逆 bank ID、验证摘要 hash。不得包含 raw key、数据样本、候选代码或部署密钥。

formal quantized 模型第一版使用 `torch.ao.quantization.quantize_dynamic` 对 `nn.Linear` 做 qint8 动态量化；若运行时不支持该编码器，`gate-validate` 必须失败并说明“不产生 formal bundle”，不能默默回退 float 后仍标记 quantized 已验证。

bundle loader 提供一个唯一 `StableGatePredictor`：每个输入先运行 formal int8，再运行 bundle 内 float reference；二者离散 close/suitable 不同或概率差超过 manifest 上限时返回 `GateScores.stable=False`。生成端和检测端都只能注入该 predictor，不能各自选择不同精度路径。

- [ ] **步骤 5：实现硬验收矩阵**

`GateValidator` 对同一 holdout group 至少运行：batch size 1/2/4/8、原序/逆序/固定随机序、CPU float、CPU int8、可用时 GPU float、每种模式两次独立进程加载。比较概率、离散 close/suitable、raw accepted sets、经过共识过滤后的 formal window spans。只有同时满足以下条件才写 `validated=true`：

```python
decision_agreement >= 0.999
float_quantized_accepted_set_agreement >= 0.999
formal_accepted_span_consensus == 1.0
suitable_false_positive_rate <= 0.05
```

正式阈值只能在 gate-validate 的 grouped threshold-fit 子集上冻结：`close_low`、`close_high`、`suitable_accept`、允许的 float/int8 概率偏差和 max tokens；随后在不重叠的 agreement 子集上执行上面硬验收。`gate-train`、最终 watermark 检测集和 detector negative calibration 都不能调这些 gate 阈值。阈值之间必须留出非空 close 不确定区，所有跨模式不一致样本强制进入该区或 suitable skip。

GPU 不可用时必须写 `gpu_status="not_available"`，不能假装通过 GPU 检验；该 bundle 只能声明 CPU 验证范围。

- [ ] **步骤 6：运行目标测试**

```bash
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/gate/test_bundle.py tests/gate/test_validation.py -v
```

预期：PASS，全部使用 tiny/fake 模型，不运行真实训练。

- [ ] **步骤 7：本地检查点**

```bash
git diff --check
git status --short
```

不得暂存或提交。

---

### 任务 11：新增 gated preset、运行目录和安全密钥入口

**文件：**

- 修改：`wfcllm/method/config.py`
- 修改：`wfcllm/method/presets.py`
- 修改：`wfcllm/method/artifacts.py`
- 创建：`wfcllm/common/secrets.py`
- 创建：`configs/wfcllm/gated_semantic_window_v1.json`
- 创建：`tests/method/test_gated_semantic_window_preset.py`
- 修改：`tests/method/test_artifacts.py`
- 创建：`tests/common/test_secrets.py`

- [ ] **步骤 1：写 preset 与 artifact 失败测试**

```python
from __future__ import annotations

from wfcllm.method.artifacts import build_run_paths
from wfcllm.method.presets import GATED_SEMANTIC_WINDOW_V1_NAME, load_method_preset


def test_gated_preset_has_full_phase_sequence_and_no_secret() -> None:
    preset = load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME)
    assert preset.method["name"] == "gated_semantic_window_v1"
    assert preset.runtime["default_phases"] == [
        "gate-data",
        "gate-train",
        "gate-validate",
        "generate",
        "calibrate",
        "detect",
        "report",
        "audit",
    ]
    assert preset.method["windowing"]["max_units"] == 3
    assert preset.method["gate"]["max_input_tokens"] == 512
    assert "secret" not in str(preset.to_dict()).lower()


def test_run_paths_include_gate_artifacts_without_changing_final_code_path(tmp_path) -> None:
    paths = build_run_paths(tmp_path, "run-1")
    assert paths.final_code_input == tmp_path / "run-1" / "inputs" / "final_code.jsonl"
    assert paths.gate_data_manifest == tmp_path / "run-1" / "gate-data" / "manifest.json"
    assert paths.gate_bundle_manifest == tmp_path / "run-1" / "gate-validate" / "gate_bundle_manifest.json"
    assert paths.gate_validation_summary == tmp_path / "run-1" / "gate-validate" / "validation_summary.json"
```

另加回归测试：`evidence_retry_seed7x3` 的序列仍为五个主阶段，原字段和默认值不变。

- [ ] **步骤 2：写安全密钥入口失败测试**

```python
def test_secret_loader_prefers_file_and_returns_bytes(tmp_path, monkeypatch) -> None:
    path = tmp_path / "deployment.key"
    path.write_text("0101\n", encoding="utf-8")
    monkeypatch.setenv("WFCLLM_DEPLOYMENT_KEY", "wrong")
    assert load_secret(secret_file=path, env_name=None) == b"0101"


def test_secret_loader_rejects_ambiguous_sources(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WFCLLM_DEPLOYMENT_KEY", "value")
    with pytest.raises(ValueError, match="exactly one"):
        load_secret(secret_file=tmp_path / "key", env_name="WFCLLM_DEPLOYMENT_KEY")
```

- [ ] **步骤 3：运行并确认失败**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/method/test_gated_semantic_window_preset.py tests/method/test_artifacts.py tests/common/test_secrets.py -v
```

预期：FAIL，缺少 gated preset/path/secret helper。

- [ ] **步骤 4：实现双 preset 配置合同**

`WFCLLMMethodPreset` 新增默认空字典字段 `gate_data`、`gate_train`、`gate_validate`。gated 方法的 `windowing`、`gate`、`rewrite`、`semantic` 保存在现有 `method` 字典下，与批准规格的 `method.windowing` 等路径一致；现有顶层 `generation`、`semantic_lsh`、`detector`、`calibration` 继续承载可复用低层参数。`__post_init__` 根据 method name 分别校验，不得把 gated 必填项强加给旧 preset。`load_method_preset` 返回深拷贝，未知名称报 `ValueError`。

新 config 必须写完整版本和阈值，但只允许 key 的“来源参数名”或本地文件参数，不得含 key 值。它还要提供两种合法运行方式：

1. 完整八阶段，从空 run 开始；
2. 引用一个已经 `validated=true` 的 bundle，从 `generate` 开始五阶段。

- [ ] **步骤 5：实现运行目录**

在现有 `data/runs/<run_id>/` 下增加：

```text
gate-data/manifest.json
gate-data/window_groups.jsonl
gate-data/candidate_attempts.jsonl
gate-data/labels.jsonl
gate-data/split_manifest.json
gate-data/training_key_bank_manifest.json
gate-data/feasibility_summary.json
gate-train/resolved_training_config.json
gate-train/training_metrics.jsonl
gate-train/development_summary.json
gate-train/checkpoints/
gate-train/candidate_bundle/
gate-train/candidate_bundle_manifest.json
gate-validate/validation_summary.json
gate-validate/agreement_details.jsonl
gate-validate/bundle/
gate-validate/gate_bundle_manifest.json
generation/audit.jsonl
generation/candidate_sidecar.jsonl
generation/raw_attempt_summary.jsonl
calibration/reference_calibration.json
detection/positive_details.jsonl
detection/reference_negative_details.jsonl
detection/model_negative_details.jsonl
```

这些目录都不是 detector input；`inputs/final_code.jsonl` 的字段和位置不变。

- [ ] **步骤 6：实现安全密钥加载**

`load_secret` 只接受 exactly one of `secret_file` 或 `env_name`，去除单个结尾换行后返回 bytes；不日志输出值；空值报错。CLI 新方法只接受 `--secret-key-file` 或 `--secret-key-env`，不得使用 config 内明文。旧 `--secret-key` 若必须保留，只能继续服务 legacy 路径并显示弃用警告。

- [ ] **步骤 7：运行目标与回归测试**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/method tests/common/test_secrets.py -v
```

预期：PASS；旧 preset 测试没有变化。

- [ ] **步骤 8：本地检查点**

```bash
git diff --check
git status --short
```

不得暂存或提交。

---

### 任务 12：实现 gate-data、gate-train、gate-validate 低层 pipeline

**文件：**

- 创建：`wfcllm/gate/feasibility.py`
- 创建：`wfcllm/gate/pipeline.py`
- 修改：`wfcllm/gate/__init__.py`
- 创建：`scripts/wfcllm_gate.py`
- 创建：`tests/gate/test_pipeline.py`
- 创建：`tests/gate/test_feasibility.py`
- 创建：`tests/integration/test_wfcllm_gate_cli.py`

- [ ] **步骤 1：写三阶段 pipeline 失败测试**

使用依赖注入的 fake source、rewriter、LSH probe、trainer 和 validator：

```python
def test_gate_data_writes_manifest_and_grouped_jsonl(tmp_path, fake_gate_dependencies) -> None:
    result = run_gate_data(fake_gate_dependencies.data_config(tmp_path), fake_gate_dependencies)
    assert result.group_count > 0
    assert result.manifest_path.exists()
    assert result.manifest["human_eval_included"] is False
    assert result.manifest["rewrite_count"] == 6


def test_gate_train_requires_data_manifest_hash(tmp_path, fake_gate_dependencies) -> None:
    with pytest.raises(ValueError, match="data manifest"):
        run_gate_train(fake_gate_dependencies.train_config(tmp_path), fake_gate_dependencies)


def test_gate_validate_only_publishes_validated_bundle(tmp_path, fake_gate_dependencies) -> None:
    result = run_gate_validate(fake_gate_dependencies.validate_config(tmp_path), fake_gate_dependencies)
    assert result.bundle.manifest.validated is True
```

另加 feasibility 测试：独立 group 而不是派生行计数；pilot 少于 2,000 或多于 5,000 个独立起点拒绝标记通过；正负例、W1/W2/W3、语句族、R3-R1 bootstrap 收益和 holdout-key 差距分别产生明确 admission 结果。CLI 测试要求 `python scripts/wfcllm_gate.py --help` 列出 `data/train/validate`；fake backend 子命令能离线完成；没有 key 来源时 `data` 失败；`validate` 失败时不留下冒充通过的 manifest。

- [ ] **步骤 2：运行并确认失败**

```bash
TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/gate/test_pipeline.py tests/gate/test_feasibility.py tests/integration/test_wfcllm_gate_cli.py -v
```

预期：FAIL，缺少 pipeline/script。

- [ ] **步骤 3：实现 gate-data pipeline**

执行顺序固定为：加载并哈希 source manifest；拒绝 HumanEval；加载 32 个训练 key 和 8 个 holdout key；解析语句单元；生成 W1/W2/W3 候选轨迹；运行多 key LSH probe；计算 labels；按 repository/task/function group 切分；写临时目录；审计通过后原子 rename 到 `gate-data/`。

`gate-data` 必须显式选择 `scale=pilot` 或 `scale=full`。pilot 只接受 2,000～5,000 个独立起点并生成 `feasibility_summary.json`；只有通过文末 `gate-data-feasibility/v1` 准入线才能运行 full。full 目标为 20,000～50,000 个独立起点，且 manifest 同时保存 5k/10k/20k/全量的确定性 group subset IDs，用于标签/覆盖稳定曲线和后续真正的模型学习曲线。派生的前文、预算、key 和候选行不计入这些数字。

manifest 至少写 source hashes、parser contract、rewriter config hash、semantic encoder hash、LSH config hash、不可逆 key bank IDs、group counts、split counts 和 schema version。

- [ ] **步骤 4：实现 gate-train pipeline**

pipeline 校验 full data manifest、已通过的 pilot feasibility contract 与 config hash，构造 dataset/model/loss/trainer，完成训练后只把 candidate bundle 写到 `gate-train/candidate_bundle/`，并写 `gate-train/candidate_bundle_manifest.json`。正式执行时会运行训练；本任务和 CI 只通过 injected fake trainer。若独立 group 少于 20,000、`suitable=True` 少于 2,000、负 group 少于 5,000，或 validation/test 各少于 200 个正 group和 500 个负 group，训练前直接失败并报告计数，不能训练一个全负分类器，也不能用派生行补足。正式实验获批后，trainer 还要按 manifest 冻结的 5k/10k/20k/全量 group subsets 训练同配置学习曲线，并把未见 group 的 close/suitable 指标写入 `development_summary.json`；这四个运行不能在当前仅写计划的阶段执行。

- [ ] **步骤 5：实现 gate-validate pipeline**

加载 candidate bundle 与 holdout split，运行任务 10 验证矩阵。只有通过硬阈值才原子发布 `gate-validate/bundle/` 和 `gate-validate/gate_bundle_manifest.json` 并标 `validated=true`；失败结果写到 `gate-validate/failed_validation_summary.json`，不覆盖旧有效 bundle。

- [ ] **步骤 6：实现低层脚本**

`scripts/wfcllm_gate.py` 只负责参数解析、config 加载和调用三个 pipeline；业务逻辑不得写进脚本。真实 backend 默认离线，不自动下载模型/数据。加 `--backend fake` 仅供测试，输出必须标记 `diagnostic_test_backend=true` 且不能被主编排接受为 formal bundle。

- [ ] **步骤 7：运行目标测试**

```bash
TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/gate tests/integration/test_wfcllm_gate_cli.py -v
```

预期：PASS；没有真实训练或网络访问。

- [ ] **步骤 8：本地检查点**

```bash
git diff --check
git status --short
```

不得暂存或提交。

---

### 任务 13：把三个 gate 阶段接入可配置主编排

**文件：**

- 修改：`wfcllm/orchestration/state.py`
- 修改：`wfcllm/orchestration/pipeline.py`
- 修改：`wfcllm/orchestration/prereq.py`
- 修改：`wfcllm/cli/arguments.py`
- 修改：`wfcllm/cli/entry.py`
- 修改：`wfcllm/cli/runners.py`
- 修改：`wfcllm/cli/config_resolver.py`
- 修改：`tests/integration/test_orchestration.py`
- 修改：`tests/integration/test_cli_resolution.py`

- [ ] **步骤 1：写阶段序列失败测试**

```python
def test_gated_method_uses_eight_phase_sequence(orchestrator_factory) -> None:
    orchestrator = orchestrator_factory(method_name="gated_semantic_window_v1")
    assert orchestrator.resolve_phase_sequence() == [
        "gate-data", "gate-train", "gate-validate", "generate",
        "calibrate", "detect", "report", "audit",
    ]


def test_existing_method_keeps_five_phase_sequence(orchestrator_factory) -> None:
    orchestrator = orchestrator_factory(method_name="evidence_retry_seed7x3")
    assert orchestrator.resolve_phase_sequence() == [
        "generate", "calibrate", "detect", "report", "audit"
    ]


def test_generate_requires_validated_gate_bundle_for_gated_method(orchestrator_factory) -> None:
    with pytest.raises(ValueError, match="validated gate bundle"):
        orchestrator_factory(method_name="gated_semantic_window_v1").run_phase("generate")
```

增加测试：显式 `--phase gate-data/gate-train/gate-validate` 可运行；引用外部已验证 bundle 时可从 generate 开始；bundle hash 不匹配拒绝；旧 phase status 文件仍可读。

- [ ] **步骤 2：运行并确认失败**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_orchestration.py tests/integration/test_cli_resolution.py -v
```

预期：FAIL，当前只有固定五阶段。

- [ ] **步骤 3：实现“允许阶段集合”与“方法阶段序列”分离**

`ALL_PHASES` 加入三项 gate phase，供 argparse 和 registry 使用；现有 `PHASES` 保持五阶段基线常量。`PhaseOrchestrator.resolve_phase_sequence()` 从已解析 preset 读取 `runtime.default_phases`，逐项校验属于 `ALL_PHASES`；未提供时使用 `PHASES`。这样不能把 gate 阶段强制加到旧 preset。

- [ ] **步骤 4：注册 runner 和前置条件**

新增 `run_gate_data`、`run_gate_train`、`run_gate_validate`，分别调用任务 12 pipeline。前置关系：`gate-train` 需要 data manifest；`gate-validate` 需要 candidate bundle；gated `generate` 需要 validated bundle；gated `detect` 需要与 generate/calibrate 相同的 bundle hash。旧方法的前置关系不变。

- [ ] **步骤 5：验证 skip/force 和 run-state**

run-state 每个 gate phase 保存 config hash、输入 artifact hash 和输出 manifest hash。输入 hash 变化时，即使旧 status 是 done 也不能跳过；`--force` 只重跑目标 phase，不删除用户 artifact。

- [ ] **步骤 6：运行编排测试**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_orchestration.py tests/integration/test_cli_resolution.py -v
conda run -n WFCLLM python run.py --status
```

预期：PASS；status 同时认识新 phase，旧 config 仍解析为五阶段。

- [ ] **步骤 7：本地检查点**

```bash
git diff --check
git status --short
```

不得暂存或提交。

---

### 任务 14：扩展 gate artifact、密钥和 no-quality-gate 审计

**文件：**

- 修改：`wfcllm/audit/forbidden.py`
- 修改：`wfcllm/audit/artifact_integrity.py`
- 修改：`wfcllm/audit/no_quality_gate.py`
- 创建：`tests/audit/test_gate_artifacts.py`
- 回归：`tests/audit/test_detector_input_integrity.py`

- [ ] **步骤 1：写泄漏和质量代理失败测试**

```python
@pytest.mark.parametrize(
    "field",
    ["raw_training_key", "deployment_key", "key_material", "target_lsh_region"],
)
def test_gate_artifacts_reject_secret_fields(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        audit_gate_artifact({"manifest": {field: "secret"}})


@pytest.mark.parametrize(
    "field",
    ["pass", "test_result", "correctness_score", "syntax_valid", "quality_rank"],
)
def test_formal_gate_data_rejects_quality_proxy_fields(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        audit_gate_artifact({"group": {field: True}})


def test_detector_still_accepts_only_four_final_code_fields(valid_final_code_row) -> None:
    validate_final_code_record_exact(valid_final_code_row)
    with pytest.raises(ValueError):
        validate_final_code_record_exact({**valid_final_code_row, "gate_decision": True})
```

- [ ] **步骤 2：运行并确认失败**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/audit/test_gate_artifacts.py tests/audit/test_detector_input_integrity.py -v
```

预期：新字段未被审计或 helper 不存在。

- [ ] **步骤 3：实现递归审计**

审计范围包含 gate-data JSONL、训练 metrics、checkpoint metadata、bundle manifest、validation summary、generation window audit、gated calibration/details。允许结构状态、LSH hit/margin/stability、概率、阈值、hash、retry budget；禁止 raw secrets 和 formal quality proxies。

注意：`parse_status`、`same_parent_scope`、`unit_count` 是结构事实，可以保留；不要为了避开 forbidden 名称而把 correctness proxy 改名伪装。

- [ ] **步骤 4：验证 detector sidecar 隔离**

在 audit 测试中把 `gate_decision`、candidate trajectory、retry count 和 generation audit 注入 official detector input，断言全部拒绝。gated detection 可以在运行时从 final code 重新计算 gate，但不能读取生成 sidecar。

- [ ] **步骤 5：运行审计测试**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/audit tests/method/test_no_quality_gate_contract.py -v
```

预期：PASS。

- [ ] **步骤 6：本地检查点**

```bash
git diff --check
git status --short
```

不得暂存或提交。

---

### 任务 15：让模型上下文能够回放完整替换窗口并恢复候选 0

**文件：**

- 修改：`wfcllm/generation/generator.py`
- 创建：`tests/generation/test_gated_replay.py`
- 创建：`tests/generation/test_generator.py`
- 回归：`tests/generation/test_boundary.py`

- [ ] **步骤 1：写 checkpoint replay 失败测试**

使用当前测试体系中的 fake causal model/tokenizer，先正常生成前缀和候选 0，再 rollback 到窗口起点并 replay 替换 token：

```python
def test_replay_from_checkpoint_rebuilds_kv_text_and_boundary(fake_model_context) -> None:
    checkpoint = fake_model_context.checkpoint()
    fake_model_context.replay_from_checkpoint(
        checkpoint,
        replacement_steps=[
            GeneratedToken(token_id=21, token_text="x = 2"),
            GeneratedToken(token_id=22, token_text="\n"),
            GeneratedToken(token_id=23, token_text="y = 3\n"),
        ],
    )
    assert fake_model_context.generated_text == checkpoint.generated_text + "x = 2\ny = 3\n"
    assert fake_model_context.generated_ids == checkpoint.generated_ids + [21, 22, 23]
    assert fake_model_context.checkpoint().kv_snapshot.seq_len == checkpoint.kv_snapshot.seq_len + 3
    assert fake_model_context.boundary.checkpoint().generated_text == fake_model_context.generated_text


def test_failed_rewrites_can_restore_exact_candidate_zero(fake_model_context) -> None:
    start = fake_model_context.checkpoint()
    original = [GeneratedToken(31, "total += item"), GeneratedToken(32, "\n")]
    fake_model_context.replay_from_checkpoint(start, replacement_steps=original)
    fake_model_context.replay_from_checkpoint(
        start,
        replacement_steps=[GeneratedToken(41, "total = total + item\n")],
    )
    fake_model_context.replay_from_checkpoint(start, replacement_steps=original)
    assert fake_model_context.generated_text.endswith("total += item\n")
    assert fake_model_context.generated_ids[-2:] == [31, 32]
```

增加测试：replay 后下一 token logits 与从 prompt+已接受 token 一次性 prefill 的 logits 在容差内一致；每个 token 的 boundary checkpoint 数量与正常生成一致；重复 replay 不积累 retry penalty；异常中断恢复到 start checkpoint。

另加 `checkpoint_for_generated_byte(start_byte)` 测试：当语句起点正好落在 token 边界时返回空 `protected_prefix`；当一个 token 同时包含换行/缩进和语句开头时，返回该 token 之前的 checkpoint 以及从 checkpoint 文本末尾到精确语句起点的 `protected_prefix`。后续重写只有逐字节复现该前缀才可接受，从而不能改到前一条语句或语句间 trivia。

- [ ] **步骤 2：运行并确认失败**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/generation/test_gated_replay.py tests/generation/test_generator.py tests/generation/test_boundary.py -v
```

预期：新 replay API 缺失，旧测试 PASS。

- [ ] **步骤 3：实现教师强制 replay**

把当前私有 `_GeneratedStep` 升格为不可变 `GeneratedToken(token_id: int, token_text: str)`，正常采样与 replay 共用该记录，避免用“整段文本 + token IDs”重建时丢失 token 边界。

`SawrModelContext.replay_from_checkpoint` 固定顺序：

1. 校验 replacement steps 非空、文本连接结果非空且 checkpoint 属于当前 context；
2. rollback 到 start checkpoint；
3. 对每个 `GeneratedToken` 执行教师强制 model forward，消费该 token 并更新 past KV 和 next logits，不调用 sampler；
4. 像正常 `forward_and_sample` 一样逐 token 更新 `generated_ids`、`generated_text`、`_step_history` 和 `_boundary_checkpoints`；
5. 对每个 token text 调用现有 `boundary.feed_text`，因此 parser 状态和 token 边界与正常生成完全相同，不把 prompt 当生成代码；
6. 清除从该 checkpoint 开始积累的 retry penalty，只保留已提交前缀所需历史；
7. 返回新的提交后 checkpoint。

新增私有 `_forward_forced_token(token_id)`，它必须使用 checkpoint 的 `past_kv` 消费指定 token 并保存该 token 之后的 next logits。若任何 token forward 失败，rollback 到输入的 start checkpoint 并重新抛出，不能留下半提交状态。

同时新增 `GeneratedByteCheckpoint(checkpoint, protected_prefix, exact_start_byte)` 和 `checkpoint_for_generated_byte`。它从 `_step_history` 中按每个 checkpoint 的 UTF-8 `generated_text` 长度找到不晚于目标 byte 的最近 token checkpoint，禁止把 byte offset 当 Python character index。该 API 是 gated 整窗回退唯一允许的起点映射。

- [ ] **步骤 4：运行生成回归**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/generation/test_gated_replay.py tests/generation/test_generator.py tests/generation/test_boundary.py -v
```

预期：PASS；现有 SAWR rollback 行为不变。

- [ ] **步骤 5：本地检查点**

```bash
git diff --check
git status --short
```

不得暂存或提交。

---

### 任务 16：实现完整窗口候选 0 与重写预算状态机

**文件：**

- 创建：`wfcllm/generation/gated_state.py`
- 创建：`tests/generation/test_gated_state.py`

- [ ] **步骤 1：写纯状态机失败测试**

```python
from __future__ import annotations

from wfcllm.generation.gated_state import GatedWindowRetryController, RetryAction


def test_original_hit_is_accepted_without_rewrite(window_attempt) -> None:
    controller = GatedWindowRetryController(max_rewrites=3)
    controller.begin(window_attempt(index=0, structure_ok=True, hit=True, stable=True))
    assert controller.next_action() is RetryAction.ACCEPT_ORIGINAL
    assert controller.rewrite_count == 0


def test_suitable_miss_rewrites_whole_window_then_accepts_hit(window_attempt) -> None:
    controller = GatedWindowRetryController(max_rewrites=3)
    controller.begin(window_attempt(index=0, suitable=True, structure_ok=True, hit=False, stable=True))
    assert controller.next_action() is RetryAction.REWRITE
    controller.observe(window_attempt(index=1, structure_ok=True, hit=True, stable=True))
    assert controller.next_action() is RetryAction.ACCEPT_REWRITE


def test_budget_exhaustion_restores_candidate_zero(window_attempt) -> None:
    controller = GatedWindowRetryController(max_rewrites=3)
    controller.begin(window_attempt(index=0, suitable=True, structure_ok=True, hit=False, stable=True))
    for index in (1, 2, 3):
        controller.observe(window_attempt(index=index, structure_ok=True, hit=False, stable=True))
    assert controller.next_action() is RetryAction.RESTORE_ORIGINAL
    assert controller.selected_candidate_index == 0
```

增加测试：`suitable=False` 直接保留原窗口；结构错误和数值不稳定候选消耗一次 rewrite 预算但不能接受；最大预算只计 1～R，不计候选 0；状态机不接收 correctness/test/quality 字段；开始新窗口前必须 reset。

- [ ] **步骤 2：运行并确认失败**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/generation/test_gated_state.py -v
```

预期：FAIL，缺少 gated state。

- [ ] **步骤 3：实现显式动作和记录**

```python
class RetryAction(str, Enum):
    ACCEPT_ORIGINAL = "accept_original"
    SKIP_ORIGINAL = "skip_original"
    REWRITE = "rewrite"
    ACCEPT_REWRITE = "accept_rewrite"
    RESTORE_ORIGINAL = "restore_original"


@dataclass(frozen=True)
class WindowAttempt:
    candidate_index: int
    suitable: bool
    structure_ok: bool
    semantic_hit: bool
    semantic_stable: bool
    semantic_margin: float
```

状态转移只允许读取上述结构/gate/semantic evidence 和预算；不允许传入测试执行、pass、静态质量或参考答案相似度。sidecar 可记录每次动作，但该记录不得进入检测器。

- [ ] **步骤 4：运行目标测试**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/generation/test_gated_state.py -v
```

预期：PASS。

- [ ] **步骤 5：本地检查点**

```bash
git diff --check
git status --short
```

不得暂存或提交。

---

### 任务 17：实现在线 gate 关窗和完整窗口 rollback/resample

**文件：**

- 创建：`wfcllm/generation/gated_generator.py`
- 创建：`wfcllm/generation/window_rewriter.py`
- 修改：`wfcllm/generation/__init__.py`
- 创建：`tests/generation/test_gated_generator.py`
- 创建：`tests/generation/test_window_rewriter.py`

- [ ] **步骤 1：写完整窗口重写失败测试**

```python
def test_rewrite_rolls_back_to_first_unit_not_last_unit(gated_harness) -> None:
    result = gated_harness.generate(
        original="a = 1\nb = 2\nc = 3\n",
        rewrites=["a = 10\nb = 20\n"],
        gate_close_after=3,
        original_hit=False,
        rewrite_hits=[True],
    )
    assert result.final_code == "a = 10\nb = 20\n"
    assert result.audit[0].original_unit_count == 3
    assert result.audit[0].selected_unit_count == 2
    assert result.audit[0].rollback_anchor <= result.audit[0].original_span.start
    assert result.audit[0].previous_statement_text_unchanged is True


def test_all_failed_rewrites_restore_original_and_continue(gated_harness) -> None:
    result = gated_harness.generate(
        original="x = 1\ny = 2\n",
        rewrites=["bad one\n", "bad two\n", "bad three\n"],
        original_hit=False,
        rewrite_hits=[False, False, False],
        suffix="return x + y\n",
    )
    assert result.final_code == "x = 1\ny = 2\nreturn x + y\n"
    assert result.audit[0].selected_candidate_index == 0
```

增加测试：候选可从 3 单元变为 1/2 单元；不能跨父级；0 或 4 单元拒绝；rollback token 横跨语句起点时候选必须逐字节保留 `protected_prefix`；改写后的 gate 必须稳定恢复为一个完整窗口；gate 输入看不到 key；rewriter request 看不到 key/LSH/目标 region；复合 header 重写不能改变控制角色（例如 if 不能变 for），且后续 body 从已接受 header 继续生成；输入溢出不重写。

- [ ] **步骤 2：运行并确认失败**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/generation/test_gated_generator.py tests/generation/test_window_rewriter.py -v
```

预期：FAIL，缺少 gated generator/rewriter。

- [ ] **步骤 3：实现在线窗口生命周期**

`GatedGenerator` 对每个新语句单元执行：

1. 用共享 parser 取得刚完成单元；
2. 若这是开放窗口第一个单元，用 `checkpoint_for_generated_byte` 保存 rollback anchor 与必须原样保留的 `protected_prefix`；
3. 用共享 `WindowPartitioner` 更新开放窗口；
4. 关窗时保存 candidate 0 的精确 token IDs/text；
5. 若 close-and-skip 或 unsuitable，提交 candidate 0；
6. 若 suitable，使用部署 key 的 `SemanticWindowScorer` 检查 candidate 0；
7. miss 时 rollback 到 anchor，按预算重采样 `protected_prefix + 整个窗口`；
8. 每个重写必须先逐字节匹配 protected prefix，再经 parser、同父级、1～3 单元和共享 gate 关窗检查，并从候选最终结构重新计算父节点描述后才做 semantic stability 检查；
9. 首个稳定 hit 用任务 15 replay 提交；预算耗尽则 replay candidate 0；
10. 对提交后的完整 prefix 重新解析，更新后续单元 ordinal 和父节点描述，reset window cursor 后继续生成。

不能“只重写最后一条语句”，不能用后续已生成 body 修补 header，也不能在两个候选之间做 quality 排名。

- [ ] **步骤 4：实现 gate-data rewriter adapter**

`CausalWindowRewriter` 从给定已完成 prefix 开始重新生成完整窗口，停止条件使用共享 parser 和固定最大 3 单元，不读取 gate 的 suitable 预测来偏置候选，不读取任何 key 或 LSH 结果。输出 token IDs、文本、parser spans 和父级描述；正式 gate-data 总是收集完整六次，不因提前命中停止。

- [ ] **步骤 5：验证候选选择信息隔离**

在测试 fake gate 和 fake rewriter 的调用参数，递归断言不存在：`secret_key`、`training_key`、`deployment_key`、`lsh`、`target_region`、`candidate_hit`、`retry_result`、`test_result`。

- [ ] **步骤 6：运行目标与生成回归测试**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/generation/test_gated_generator.py tests/generation/test_window_rewriter.py tests/generation/test_gated_replay.py tests/generation/test_gated_state.py -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/generation -q
```

预期：PASS；现有 generation suite 无回归。

- [ ] **步骤 7：本地检查点**

```bash
git diff --check
git status --short
```

不得暂存或提交。

---

### 任务 18：建立 gated 批量生成和严格 artifact 输出

**文件：**

- 创建：`wfcllm/generation/gated_pipeline.py`
- 修改：`wfcllm/generation/outputs.py`
- 修改：`wfcllm/cli/runners.py`
- 创建：`tests/generation/test_gated_pipeline.py`
- 修改：`tests/integration/test_wfcllm_generate_cli.py`

- [ ] **步骤 1：写输出隔离失败测试**

```python
def test_gated_pipeline_keeps_official_final_code_strict(tmp_path, fake_gated_pipeline) -> None:
    fake_gated_pipeline.run(tmp_path)
    rows = read_jsonl(tmp_path / "inputs" / "final_code.jsonl")
    assert set(rows[0]) == {"id", "dataset", "prompt", "final_code"}
    assert "gate" not in rows[0]
    assert "candidate" not in rows[0]


def test_generation_audit_is_sidecar_only(tmp_path, fake_gated_pipeline) -> None:
    fake_gated_pipeline.run(tmp_path)
    audit = read_jsonl(tmp_path / "generation" / "audit.jsonl")
    assert audit[0]["audit_only"] is True
    assert audit[0]["not_detector_input"] is True
    assert "selected_candidate_index" in audit[0]
```

增加测试：bundle 未 validated 拒绝；parser/gate/tokenizer hash 不匹配拒绝；候选失败保留原窗口；existing preset 继续走现有 pipeline；gated 方法不能再叠加外层 evidence-retry 选择多个完整程序。

- [ ] **步骤 2：运行并确认失败**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/generation/test_gated_pipeline.py tests/integration/test_wfcllm_generate_cli.py -v
```

预期：FAIL，缺少 gated pipeline/dispatch。

- [ ] **步骤 3：实现 gated pipeline**

`GatedGenerationPipeline` 加载一个 validated gate bundle、semantic encoder/LSH、部署 key（只在内存）、基础代码模型和数据 adapter。每个样本只输出一个最终程序；内部窗口候选记录写 sidecar。run manifest 必须绑定 gate bundle hash、parser contract、semantic encoder hash、LSH config hash、generation config hash 和 secret source 类型，但不写 secret 值。

如果重写候选全部失败，正式代码必须包含 candidate 0，不重新生成“普通替代语句”。如果单个样本发生不可恢复 parser/context 错误，按 config 明确 fail-fast 或标记 sample generation failure，不能把半截代码送入 detector。

- [ ] **步骤 4：在 runner 中按 method dispatch**

`run_generate` 读取 method name：旧 preset 调现有生成路径；gated preset 调 `GatedGenerationPipeline`。不得用 gated 代码改变旧 preset 的候选选择、重试次数或 artifact schema。

- [ ] **步骤 5：运行生成测试**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/generation tests/integration/test_wfcllm_generate_cli.py -v
```

预期：PASS；测试使用 fake model/bundle，不运行真实生成。

- [ ] **步骤 6：本地检查点**

```bash
git diff --check
git status --short
```

不得暂存或提交。

---

### 任务 19：从最终代码独立恢复 gated 窗口

**文件：**

- 修改：`wfcllm/detection/config.py`
- 创建：`wfcllm/detection/gated_windows.py`
- 修改：`wfcllm/detection/__init__.py`
- 创建：`tests/detection/test_gated_windows.py`
- 回归：`tests/detection/test_proxy_windows.py`
- 回归：`tests/detection/test_code_only_contract.py`

- [ ] **步骤 1：写 final-code-only 窗口恢复失败测试**

```python
from __future__ import annotations

from wfcllm.detection.gated_windows import GatedWindowExtractor


def test_extractor_recovers_exact_generation_spans_from_final_code(
    generated_fixture, validated_fake_gate_bundle
) -> None:
    extracted = GatedWindowExtractor(validated_fake_gate_bundle).extract(
        generated_fixture.final_code
    )
    assert [window.byte_span for window in extracted.windows] == (
        generated_fixture.accepted_window_spans
    )
    assert [window.parent_descriptor for window in extracted.windows] == (
        generated_fixture.accepted_parent_descriptors
    )


def test_generation_sidecar_cannot_change_extraction(
    generated_fixture, validated_fake_gate_bundle
) -> None:
    extractor = GatedWindowExtractor(validated_fake_gate_bundle)
    before = extractor.extract(generated_fixture.final_code)
    generated_fixture.sidecar["accepted_window_spans"] = [[999, 1000]]
    after = extractor.extract(generated_fixture.final_code)
    assert before == after
```

增加测试覆盖分号、单行 if、嵌套 if/for、try/except/finally、match/case、async、return、排除语句、多行括号表达式、UTF-8 字符；prompt 内容改变不能改变窗口；bundle/parser/tokenizer hash 不匹配拒绝；溢出窗口 skip。

- [ ] **步骤 2：运行并确认失败**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/detection/test_gated_windows.py tests/detection/test_proxy_windows.py tests/detection/test_code_only_contract.py -v
```

预期：新 extractor 不存在；旧 proxy/code-only 测试 PASS。

- [ ] **步骤 3：实现 gated detector 配置**

保留现有 `DETECTOR_MODE` 与 `WFCLLMDetectionConfig` 默认值。新增独立 `GATED_DETECTOR_MODE = "wfcllm-gated-semantic-window/v1"` 和 `GatedDetectionConfig`，必填 gate bundle path/hash、window/input contract、semantic encoder/LSH hash、minimum reliable windows、calibration artifact path。

- [ ] **步骤 4：实现只读提取器**

`GatedWindowExtractor.extract(final_code)` 固定执行：

1. 校验 gate bundle 为 validated 且哈希匹配；
2. 调用任务 1 的共享 Python extractor；
3. 按最终代码重新计算规范父节点描述；
4. 调用同一个 `WindowPartitioner` 和 bundle 内 `StableGatePredictor`（formal int8 决策 + float reference 一致性检查）；
5. 返回全部关闭窗口、suitable/uncertain/overflow 状态和 byte spans；
6. 不接受 prompt、generation result、sidecar、原始候选、重试次数、hidden states 或 logits 文件参数。

API 层面让构造函数无法接收 generation sidecar，而不是仅靠调用方自律。

- [ ] **步骤 5：增加生成—提取合同夹具**

在测试 fixture 中由同一个 final code 分别走 generation 的提交后 partition 和 detection extractor；比较的是最终 byte spans、unit texts、parent descriptors、suitable/skip，不比较生成日志。对所有 parser 边界夹具要求 100% 一致。

- [ ] **步骤 6：运行检测与窗口测试**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/windowing tests/detection/test_gated_windows.py tests/detection/test_proxy_windows.py tests/detection/test_code_only_contract.py -v
```

预期：PASS；现有 proxy extractor 不变。

- [ ] **步骤 7：本地检查点**

```bash
git diff --check
git status --short
```

不得暂存或提交。

---

### 任务 20：实现 gated 校准、评分、检测和 runner 派发

**文件：**

- 修改：`wfcllm/detection/scoring.py`
- 创建：`wfcllm/detection/gated_pipeline.py`
- 修改：`wfcllm/cli/runners.py`
- 创建：`tests/detection/test_gated_pipeline.py`
- 创建：`tests/detection/test_gated_calibration.py`
- 修改：`tests/integration/test_wfcllm_detect_cli.py`

- [ ] **步骤 1：写 hit/miss/abstain 失败测试**

```python
def test_gated_scorer_uses_only_stable_suitable_windows(fake_gated_scorer) -> None:
    score = fake_gated_scorer.score(
        windows=[
            fake_window(suitable=True, semantic="hit"),
            fake_window(suitable=True, semantic="miss"),
            fake_window(suitable=True, semantic="unstable"),
            fake_window(suitable=False, semantic="hit"),
        ]
    )
    assert score.hit_count == 1
    assert score.miss_count == 1
    assert score.abstain_count == 1
    assert score.reliable_window_count == 2
    assert score.hit_rate == 0.5


def test_insufficient_reliable_windows_abstains(fake_gated_scorer) -> None:
    result = fake_gated_scorer.detect([fake_window(suitable=True, semantic="hit")])
    assert result.decision == "insufficient_evidence"
```

- [ ] **步骤 2：写校准隔离失败测试**

```python
def test_calibration_binds_gate_and_semantic_hashes(tmp_path, fake_negative_rows) -> None:
    artifact = calibrate_gated_detector(fake_negative_rows, output_dir=tmp_path)
    assert artifact.detector_mode == "wfcllm-gated-semantic-window/v1"
    assert artifact.gate_bundle_sha256
    assert artifact.semantic_encoder_sha256
    assert artifact.window_contract_version == "python-statement-window/v1"
    assert artifact.reliable_window_count_buckets
    assert artifact.empirical_p_value_rule == "right_tail_plus_one/v1"


def test_detector_rejects_generation_audit_as_input(gated_detector, audit_sidecar_path) -> None:
    with pytest.raises(ValueError, match="final_code"):
        gated_detector.detect_jsonl(audit_sidecar_path)
```

- [ ] **步骤 3：运行并确认失败**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/detection/test_gated_pipeline.py tests/detection/test_gated_calibration.py tests/integration/test_wfcllm_detect_cli.py -v
```

预期：FAIL，缺少 gated scoring/pipeline。

- [ ] **步骤 4：抽取共用评分协议但保留 proxy 统计量**

在 `scoring.py` 增加只读 `ScorableWindow` protocol 或小 dataclass，使 proxy 与 gated 路径复用 semantic encoder 调用和数值稳定性判断。现有 proxy penalty、context mean 和 detector mode 不能改名或改公式。

gated 路径对每个 `suitable=True` 窗口输出：稳定命中 `hit`、稳定未命中 `miss`、margin/精度不稳定 `abstain`。样本统计量固定为 `hit_count / (hit_count + miss_count)`；可靠窗口数不足配置下限时返回 `insufficient_evidence`，不能把 abstain 当 miss。

- [ ] **步骤 5：实现 final-code-only 经验校准**

校准输入是没有水印的严格四字段代码，运行与正样本完全相同的 parser、formal quantized gate、window partition、semantic encoder、部署 key 派生和稳定性过滤。按可靠窗口数量分桶保存背景 hit-rate 分布，使用加一修正的经验右尾 p-value，按冻结目标 FPR（默认 5%）选择 sample-level 阈值，并同时冻结 minimum reliable windows。不能用名义 `gamma` 代替经验背景分布。

calibration artifact 绑定 gate bundle、parser、gate input、semantic encoder、LSH config、key identifier hash、negative corpus manifest 和 detector mode；不写 raw key。任何 hash 不匹配都拒绝检测。

- [ ] **步骤 6：实现 gated detection pipeline**

`GatedDetectionPipeline.calibrate` 与 `.detect` 只调用严格 final-code loader。details 可以记录从 final code 派生的 span、parent descriptor、gate probabilities、hit/miss/abstain 和 margin；不得记录或读取 generation candidate/retry sidecar。

- [ ] **步骤 7：在 runner 中按 method dispatch**

`run_calibrate`/`run_detect` 对 gated preset 调 gated pipeline，对旧 preset 调现有 proxy pipeline。报告中必须同时写 method name 和 detector mode，禁止把 gated 与 proxy 数字混成一个方法。

- [ ] **步骤 8：运行检测回归**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/detection tests/integration/test_wfcllm_detect_cli.py -v
```

预期：PASS；旧 detection suite 无回归。

- [ ] **步骤 9：本地检查点**

```bash
git diff --check
git status --short
```

不得暂存或提交。

---

### 任务 21：打通八阶段 fake 工作流并更新正式文档

**文件：**

- 创建：`tests/integration/test_gated_workflow.py`
- 修改：`tests/integration/test_orchestration.py`
- 修改：`docs/ARTIFACT_SCHEMA.md`
- 修改：`docs/NO_QUALITY_GATE_PROTOCOL.md`
- 修改：`README.md`
- 修改：`CLAUDE.md`
- 修改：`configs/wfcllm/gated_semantic_window_v1.json`
- 修改：`wfcllm/cli/runners.py`

- [ ] **步骤 1：写八阶段端到端失败测试**

```python
def test_gated_workflow_runs_all_phases_with_fake_backends(tmp_path, cli_runner) -> None:
    result = cli_runner.run_gated(tmp_path, backend="fake")
    assert result.phase_order == [
        "gate-data", "gate-train", "gate-validate", "generate",
        "calibrate", "detect", "report", "audit",
    ]
    assert result.run_paths.final_code_input.exists()
    assert result.run_paths.gate_bundle_manifest.exists()
    assert result.audit_summary["detector_input_integrity"] == "pass"


def test_detector_opens_no_generation_sidecars(tmp_path, tracing_open, cli_runner) -> None:
    cli_runner.run_gated(tmp_path, backend="fake")
    detector_reads = tracing_open.paths_for_phase("detect")
    assert all("generation/audit.jsonl" not in str(path) for path in detector_reads)
    assert all("generation/candidate_sidecar.jsonl" not in str(path) for path in detector_reads)
    assert all("gate-data" not in str(path) for path in detector_reads)


def test_external_validated_bundle_skips_gate_phases(tmp_path, cli_runner, fake_bundle) -> None:
    result = cli_runner.run_gated(tmp_path, bundle=fake_bundle, start_phase="generate")
    assert result.phase_order == ["generate", "calibrate", "detect", "report", "audit"]
```

另加旧 preset 五阶段回归、run-state resume、bundle hash mismatch、failed validation 不能 generate、gated/proxy 报告分离。

- [ ] **步骤 2：运行并确认失败**

```bash
TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_gated_workflow.py tests/integration/test_orchestration.py -v
```

预期：FAIL，直到 runner/config/docs 合同全部接通。

- [ ] **步骤 3：完成配置和 report/audit 派发**

`gated_semantic_window_v1.json` 明确写：完整 phase sequence、所有 contract/version、max 3、candidate 0、rewrite budgets、gate data规模、32/8 key 数、512 tokens、标签阈值、正式量化模式、验证阈值、detector FPR、posthoc 非劣界。配置不含任何 key 或本地绝对路径。

`run_report` 为 gated 方法报告 gate coverage、hit/miss/abstain、重写成本、检测曲线和 calibration；pass@1 只能从已有 posthoc artifact 合并，并保留 `posthoc_only=true`、`not_used_for_generation=true`。`run_audit` 加入任务 14 检查。

- [ ] **步骤 4：更新 artifact 与协议文档**

`docs/ARTIFACT_SCHEMA.md` 增加 `gate-data/`、`gate-train/`、`gate-validate/`、gated generation sidecar、gated calibration/details 的字段表和 producer/consumer；再次明确 detector 正样本只读 `inputs/final_code.jsonl`。

`docs/NO_QUALITY_GATE_PROTOCOL.md` 增加 gate 正式允许信号（结构、key-independent 多训练 key LSH 汇总、margin/stability、预算）和禁止信号；解释 parser 结构有效不等于代码质量，posthoc correctness 不得回流。

`README.md` 和 `CLAUDE.md` 写清：旧 preset 仍是正式基线；gated 方法实验性启用；三个 gate 阶段；本地模型/数据/密钥要求；不自动下载；不提交 artifacts。

- [ ] **步骤 5：运行 fake 端到端和配置测试**

```bash
TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_gated_workflow.py tests/integration/test_orchestration.py tests/integration/test_cli_resolution.py -v
conda run -n WFCLLM python run.py --status
```

预期：PASS；fake workflow 只验证接线和合同，不声称真实模型效果。

- [ ] **步骤 6：本地检查点**

```bash
git diff --check
git status --short
```

不得暂存或提交。

---

### 任务 22：分层验证、结果边界和实施交付检查

**文件：**

- 验证：上述全部实现与测试文件
- 不新增功能文件
- 不修改用户既有未跟踪文件

- [ ] **步骤 1：静态编译和测试收集**

```bash
conda run -n WFCLLM python -m compileall wfcllm run.py scripts tools
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest --collect-only -q
```

预期：compileall 成功；pytest 能收集新旧测试，无 import error。

- [ ] **步骤 2：按层运行主动测试**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/windowing -v
TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/gate -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/semantic -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/generation -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/detection -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/audit -v
TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration -v
```

预期：全部 PASS。资源缺失失败必须报告首个命令、首个错误行和影响文件，不能写成“全套通过”。

- [ ] **步骤 3：运行完整回归**

```bash
TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -v
```

预期：PASS；若因本地模型/数据缺失失败，保留分层测试证据并明确完整套件未通过。

- [ ] **步骤 4：执行静态合同审计**

```bash
rg -n 'raw_training_key|deployment_key|key_material|target_lsh_region' wfcllm configs docs/ARTIFACT_SCHEMA.md docs/NO_QUALITY_GATE_PROTOCOL.md
rg -n 'pass|test_result|correctness_score|quality_rank|syntax_valid' wfcllm/gate wfcllm/generation/gated_state.py wfcllm/generation/gated_generator.py
git diff --check
```

预期：第一条只命中禁止字段集合、审计测试或说明文字；第二条只命中审计/拒绝逻辑、posthoc 说明或测试，formal 决策路径无质量代理。

- [ ] **步骤 5：核对用户文件与 Git 边界**

```bash
git status --short
git diff --name-only
```

预期：没有修改或删除 `.aris/`、`MANIFEST.md`、`docs/design/semantic_atomic_unit_gate_final*.md`、`review-stage/`；没有 `data/`、模型、checkpoint、日志、大型 JSONL 被跟踪；没有暂存区、commit、push 或 PR。

- [ ] **步骤 6：按证据声明完成范围**

只有 fake/offline 测试通过时，只能声明“实现合同和离线接线完成”，不能声明真实 gate 有效、检测提升或数据充足。真实结论必须在用户另行批准后依次取得：

1. 2,000～5,000 个独立起点的 gate-data 可行性批次；
2. 通过无门模型 R=1/3/6 和 LSH 稳定性检查；
3. 扩展至 20,000～50,000 个独立起点；
4. 获得明确的数据学习曲线和 split 外泛化；
5. 运行 gate-train 与 gate-validate；
6. 运行真实生成、负样本校准和 final-code-only 检测；
7. 最后运行 posthoc pass@1 和鲁棒性比较。

任何一层未通过都按下面“阶段门”回退，不能跳过。

- [ ] **步骤 7：停在用户验收点**

总结修改文件、测试命令、通过/失败证据、未运行的真实训练/实验、剩余风险。等待用户明确批准后再决定是否运行真实 gate-data 或训练；不得自行训练、暂存、提交、push 或创建 PR。

---

## 阶段门与失败回退

| 阶段 | 进入下一阶段的硬条件 | 失败时回退 |
|---|---|---|
| A：共享解析与窗口合同 | 合同夹具及 generation/detection adapter 的基础语句 span 100% 相同 | 只修 `wfcllm.windowing`，不收数据 |
| B：gate-data pilot | 2,000～5,000 个独立起点；无 split/key/quality 泄漏；W1/W2/W3 和候选 0～6 可恢复 | 回到 parser、source 或 rewriter 合同 |
| C：无门模型可行性 | R=3/R=6 有足够结构有效且稳定的命中，预算增长有可测收益，正负标签不退化为单一类别 | 调整语义 LSH、允许区域或 key-blind 重写；不训练 gate |
| D：完整 gate-data | 20,000～50,000 个独立起点；按 repository/task/function 分组；学习曲线和未见来源覆盖可计算 | 增加独立来源，不用派生变体伪装数据量 |
| E：gate-train/validate | 解析一致、离散决策 >=99.9%、accepted spans 100%、量化 accepted set >=99.9%、suitable FP <=5% | 扩数据、加不确定区或缩小正式语句范围；bundle 不发布 |
| F：生成接入 | 整窗回退、1～3 单元变化、候选 0 精确恢复、继续生成全部通过 | 暂时限制保持单元数的 diagnostic 模式，正式 preset 不启用 |
| G：final-code-only 检测 | 不读 sidecar 可恢复相同 accepted windows；负样本经验校准可用 | 保留旧 proxy baseline，不声称复现 gate 窗口 |
| H：正式比较 | 固定 5% FPR 下检测不低于基线、重写成本下降、posthoc pass@1 下降不超过 2pp | gated 方法保持 experimental/diagnostic，不替代基线 |

## 数据是否足够的判定

“数据够不够”只按独立 window group 计算，前文 1/2/3、预算 1/3/6、32 个训练 key 和 7 个候选产生的行都不增加独立样本数。

第一轮 pilot 使用 2,000～5,000 个独立起点，仅回答机制是否值得训练。pilot 报告至少同时展示：

- suitable 正/负 group 数与比例；
- W1/W2/W3、主要语句类型、来源仓库、任务和代码生成模型的覆盖；
- R1、R3、R6 可靠成功率和 bootstrap 置信区间；
- 结构无效率、数值不稳定率、首次命中候选位置；
- 训练 key 与 8 个 holdout key 的差距；
- 按 group 数 500/1,000/2,000/5,000 的标签比例稳定曲线。

pilot 满足以下默认准入线才扩展数据：至少 200 个 suitable 正 group、500 个 suitable 负 group；W1/W2/W3 各至少 200 个独立 group；至少四类主要语句族各 100 个独立 group；R3 相对 R1 的可靠成功率提升 95% bootstrap 区间下界大于 0；holdout-key 成功率相对训练 key 的绝对下降不超过 10 个百分点。任一条件不满足，不启动 gate-train。

完整训练集目标是 20,000～50,000 个独立起点。正式训练前至少需要 2,000 个 suitable 正 group、5,000 个负 group，且 validation/test 各自至少含 200 个正 group和 500 个负 group；如果天然正例率使这些条件无法满足，应调整语义/重写机制或扩大独立来源，而不是重复派生同一窗口。最终是否继续扩数据由 5k/10k/20k/全量的 group-level 学习曲线决定：若 suitable 假阳性和 accepted-window 一致性仍持续改善，继续收集；若已经平台且未见 repository/model 泛化达到 gate-validate 门槛，优先改方法而不是盲目堆数据。

这些准入值属于 `gate-data-feasibility/v1` 合同，必须写入 config、manifest 和报告；改变阈值会产生新的 feasibility contract ID，不能用新阈值覆盖旧结论。

## 实现完成后的正式比较矩阵

最终报告必须把以下方法明确分开，不能只报告新方法的最佳配置：

1. 当前 `evidence_retry_seed7x3` + proxy detector 基线；
2. 不用 gate、固定窗口长度 1；
3. 不用 gate、固定窗口长度 2；
4. 不用 gate、最大 3 单元的规则窗口；
5. 只在生成端使用 gate、检测端旧 proxy（诊断对照，不是主张的最终方法）；
6. 生成端和检测端使用同一 formal gate bundle 的完整方法；
7. float gate 与 formal quantized gate 的稳定性对照；
8. rewrite budget R=1、R=3、R=6 消融。

每项报告独立窗口数、候选 0 命中率、嵌入成功率、实际重写次数、hit/miss/abstain、insufficient evidence、TPR@5%FPR、运行成本、跨 batch/设备/量化一致性，以及只在最后计算的 posthoc pass@1 和轻微格式修改保留率。

## 计划完成定义

计划的代码实施只有在以下条件全部满足时才能称为“实现完成”：

- 新旧 preset 测试全部通过，旧基线语义未被改变；
- generation 与 detection 只依赖同一个共享 parser/partitioner；
- candidate 0、完整窗口重写、可变单元数和精确恢复均有回归测试；
- gate-data、gate-train、gate-validate 已进入主编排并有 artifact/hash/prerequisite 合同；
- formal detector 仍只读取严格 final code；
- bundle、训练数据和 sidecar 通过密钥/no-quality-gate 审计；
- fake 八阶段端到端测试和完整主动测试通过；
- 没有声称未经真实数据、训练、校准和检测证明的效果；
- 没有修改用户未跟踪文件，没有 commit、push 或 PR。

完成代码实施后仍需单独经过真实数据与实验阶段门，才可以把 `gated_semantic_window_v1` 从 experimental 晋升为正式方法并考虑替代当前基线。
