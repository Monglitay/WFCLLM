# Reference Negative Corpus Design

**Goal:** 让 `generate-negative` 支持两种负样本来源：数据集原生参考解和 LLM 无水印生成，并默认使用参考解作为校准负样本。

**Why:** 当前 `negative_corpus.jsonl` 来自同模型无水印生成，本质上是 AI-negative。它适合衡量 “watermarked AI vs unwatermarked AI”，但不适合近似人类代码上的 FPR。为了让 `extract` 的 `fpr` 更接近 human-negative 语义，需要直接使用 HumanEval / MBPP 的原生参考解。

## Scope

- 在 `generate-negative` 配置中新增 `source_mode`
- 支持 `reference` 和 `llm` 两种模式
- `reference` 模式下不加载 LLM，直接从本地数据集读取参考解并写出 JSONL
- `llm` 模式保持现有行为
- HumanEval 和 MBPP 都统一支持该模式

## Data Model

`negative_corpus.jsonl` 继续保持现有格式，不改提取端消费接口：

```json
{
  "id": "HumanEval/0",
  "dataset": "humaneval",
  "prompt": "def foo(...):\n",
  "generated_code": "..."
}
```

这里只改变 `generated_code` 的来源：

- `reference`: 数据集原生参考解
- `llm`: 当前无水印 LLM 输出

## Dataset Mapping

### HumanEval

- `id` -> `task_id`
- `prompt` -> `prompt`
- `generated_code` -> `canonical_solution`

### MBPP

- `id` -> `mbpp/{task_id}`
- `prompt` -> `text`
- `generated_code` -> `code`

## Architecture

### `wfcllm/common/dataset_loader.py`

新增一个加载 “prompt + reference solution” 对的入口，避免把参考解逻辑散落在 `negative_corpus.py`。

建议接口：

```python
def load_reference_solutions(dataset: str, dataset_path: str) -> list[dict]:
    ...
```

返回统一结构：

```python
{
  "id": "...",
  "prompt": "...",
  "generated_code": "...",
}
```

### `wfcllm/extract/negative_corpus.py`

新增：

- `NegativeCorpusConfig.source_mode: Literal["reference", "llm"] = "reference"`

运行行为：

- `reference` 模式：`run()` 调用 `load_reference_solutions()`，直接写 JSONL；`__init__` 不加载模型
- `llm` 模式：保持现有 `load_prompts() -> _generate()` 路径

### `run.py`

`run_generate_negative()` 改成：

- 读取 `generate_negative.source_mode`
- 只有在 `source_mode == "llm"` 时才要求 `lm_model_path`
- `reference` 模式下允许不传 `--lm-model-path`

## Compatibility

- 默认切到 `reference`
- `llm` 路径保持兼容，便于后续做 AI-negative 对照实验
- `extract` 侧不需要改输入格式和读取逻辑

## Risks

### 1. HumanEval / MBPP 参考解口径不同

HumanEval 的 `canonical_solution` 与 prompt continuation 相容；MBPP 的 `code` 是完整解。当前提取器只看 `generated_code`，所以可以统一消费，但 block 统计分布会和 LLM-generated negatives 不同。这是预期行为，不是兼容性问题。

### 2. 旧测试假设总会加载 LLM

`NegativeCorpusGenerator.__init__` 会变成条件加载，测试需要拆成 reference / llm 两种路径。

## Success Criteria

- `generate-negative` 默认生成 reference-negative corpus
- 不传 `--lm-model-path` 也能成功跑 `reference` 模式
- 显式切换 `source_mode="llm"` 时维持当前行为
- `negative_corpus.jsonl` 输出格式不变
