# HumanEval Pass@1 提升设计

**日期：** 2026-07-18

**范围：** `/root/autodl-tmp/WFCLLM-gated` 与固定实验目录中的独立
`full164-attempt2`。保留且不修改 `full164-attempt1`。

## 目标与验收标准

在不使用 HumanEval 隐藏测试选择候选、不读取 canonical solution、不增加多候选质量门的前提下，提高简化单区域水印最终代码的功能正确率。

- HumanEval Pass@1：从 15.85% 提升到至少 30%。
- Python 语法有效率：从 35.37% 提升到至少 90%。
- 水印样本检出率：保持至少 85%。
- `target_fpr=0.05` 与 `minimum_reliable_windows=2` 不变。
- 每题只调用一次基础生成模型；无隐藏测试反馈，无远程模型 API。
- 所有 164 个 ID 都必须进入最终评估；任何失败都保留并计为失败，不丢分母。

## 根因证据

`full164-attempt1` 的 164 条最终代码中，106 条无法通过 `ast.parse`，58 条语法有效；全部 26 个 HumanEval 通过样本都来自语法有效集合。119 条包含额外顶层 `print`，84 条包含模型自生成测试，21 条包含 Markdown 围栏。当前 `max_new_tokens=256`，失败尾部反复出现未闭合括号、字符串、测试用例和下一个函数。

因此主要断点在基础生成输出的边界，而不是内部空白水印变换。

## 方案

### 1. 生成参数

保持本地 DeepSeek Coder 1.3B、单次生成和 seed 7，修改为：

- `max_new_tokens=512`
- `temperature=0.0`
- `top_p=0.95` 保留（greedy 时不参与采样）
- `load_in_4bit=true` 保留

不修改 HumanEval 原始 prompt，不添加隐藏测试或参考答案。

### 2. 目标函数收口器

新增纯函数组件 `wfcllm/generation/completion_finalizer.py`。输入为 HumanEval prompt 与模型返回的完整源码，输出 `ProgramFinalizationResult`：

1. 从 prompt 的 AST 中确定最后一个顶层函数名，即 HumanEval 目标函数。
2. 从完整输出开始逐行回退，寻找包含目标函数完整实现的最长可解析前缀。
3. 目标函数必须包含 prompt 原有 docstring 之外的新可执行语句；仅恢复到 prompt stub 不算成功。
4. 在可解析前缀 AST 中只保留目标函数及其之前的顶层节点，从而删除目标函数后的测试、Markdown、解释和截断续写。
5. 使用原始源码位置切片，不使用 canonical solution，也不按 HumanEval 测试结果做选择。
6. 如果无法找到完整目标函数，返回原始模型输出并记录 `applied=false`；样本仍进入 164 条分母。

收口发生在水印嵌入之前，因此最终水印仍绑定收口后的实际代码，检测不会读取收口前文本。

### 3. 管线集成和可追溯性

`GatedGenerationPipeline` 接收可选 `program_finalizer`。每个样本基础生成后、进入 `GatedGenerator` 前调用。新增：

- `generation/finalizer.jsonl`：每题记录 `id`、`applied`、`reason`、收口前后 SHA-256 和字符数；不保存额外 detector input。
- generation manifest 字段：`program_finalizer`、`finalizer_applied_count`、`finalizer_fallback_count`。

正式 detector input 仍严格只有 `id`、`dataset`、`prompt`、`final_code`。

### 4. 实验流程

1. 先在单元测试中验证截断测试代码、Markdown、完整输出、仅 prompt stub 和不匹配 prompt。
2. 运行 generation/integration 相关测试。
3. 创建隔离 `full164-attempt2`，复用既有 10-epoch gate bundle，不重训。
4. 先跑 16 题预检，仅检查语法有效率、样本齐全和管线错误；不得根据 HumanEval 测试结果选择候选。
5. 预检通过后强制生成 164 条，运行真实 calibration、detect、report。
6. 使用 HumanEval 官方测试执行一次 Pass@1，并报告 164 固定分母。
7. 比较 attempt1/attempt2 的 Pass@1、语法有效率、检出率和联合“通过且检出”比例。

## 错误处理

- 收口器永不以 prompt stub 冒充完成答案。
- 收口失败时保留原始输出并明确记录 fallback，不静默删除样本。
- 完整输出若已符合目标函数边界，收口结果保持语义等价。
- 实验日志出现 Traceback、OOM、样本缺失或 ID 不完整则 attempt2 验收失败。

## 非目标

- 不下载或调用更强远程模型。
- 不生成 3–5 个候选并按质量选择。
- 不运行项目 `audit` 或 `gate-validate`。
- 不把简化 keyed-text 空白水印称为原始四区域 semantic-LSH。
- 不使用 attempt1 的 HumanEval 测试结果对单题进行补丁或挑选。

