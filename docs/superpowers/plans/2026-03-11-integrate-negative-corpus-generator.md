# Integrate Negative Corpus Generator Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `scripts/generate_negative_corpus.py` 的逻辑迁移进 `wfcllm/extract/` 包，并同步更新 `run.py`、`configs/base_config.json` 和 `README.md`，使负样本生成可以通过 `run.py --phase generate-negative` 触发，数据集加载逻辑与现有 `WatermarkPipeline` 复用同一实现。

**Architecture:** 在 `wfcllm/extract/` 中新增 `negative_corpus.py` 模块（含 `NegativeCorpusConfig` dataclass 和 `NegativeCorpusGenerator` 类），`run.py` 新增 `generate-negative` phase，`scripts/generate_negative_corpus.py` 改为薄包装器（调用包内实现）。数据集加载逻辑从 `WatermarkPipeline._load_prompts()` 中提取为公共函数放入 `wfcllm/common/dataset_loader.py`，消除重复（包括 `pipeline.py` 中的 `SUPPORTED_DATASETS` 常量）。

**Tech Stack:** Python 3.11, transformers (AutoModelForCausalLM / AutoTokenizer / BitsAndBytesConfig), datasets (HuggingFace), torch, argparse, pytest

---

## Chunk 1: 提取公共数据集加载逻辑

### Task 1: 创建 `wfcllm/common/dataset_loader.py`

**Files:**
- Create: `wfcllm/common/dataset_loader.py`
- Modify: `wfcllm/common/__init__.py`
- Modify: `wfcllm/watermark/pipeline.py`
- Create: `tests/common/__init__.py`
- Create: `tests/common/test_dataset_loader.py`

当前 `WatermarkPipeline._load_prompts()` 和 `scripts/generate_negative_corpus.py` 中的 `_load_prompts()` 逻辑几乎相同，都调用 `load_dataset(...)` 并返回 `list[dict]`（含 `id` 和 `prompt` 字段）。`pipeline.py` 还有自己的 `SUPPORTED_DATASETS` 常量，将被统一替换。

- [ ] **Step 1: 写失败测试**

如果 `tests/common/` 目录不存在，先创建：

```bash
mkdir -p tests/common
touch tests/common/__init__.py
```

创建 `tests/common/test_dataset_loader.py`：

```python
"""Tests for wfcllm.common.dataset_loader."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from wfcllm.common.dataset_loader import SUPPORTED_DATASETS, load_prompts


class TestLoadPrompts:
    def test_supported_datasets_constant(self):
        assert "humaneval" in SUPPORTED_DATASETS
        assert "mbpp" in SUPPORTED_DATASETS

    def test_unsupported_dataset_raises(self):
        with pytest.raises(ValueError, match="dataset must be one of"):
            load_prompts("unknown", "data/datasets")

    @patch("wfcllm.common.dataset_loader.load_dataset")
    def test_humaneval_returns_id_and_prompt(self, mock_load):
        fake_split = [{"task_id": "HumanEval/0", "prompt": "def foo():"}]
        mock_ds = {"test": fake_split}
        mock_load.return_value = mock_ds

        prompts = load_prompts("humaneval", "data/datasets")

        assert len(prompts) == 1
        assert prompts[0]["id"] == "HumanEval/0"
        assert prompts[0]["prompt"] == "def foo():"

    @patch("wfcllm.common.dataset_loader.load_dataset")
    def test_mbpp_returns_id_and_prompt(self, mock_load):
        fake_split = [{"task_id": 1, "text": "Write a function"}]
        mock_ds = {"train": fake_split}
        mock_load.return_value = mock_ds

        prompts = load_prompts("mbpp", "data/datasets")

        assert len(prompts) == 1
        assert prompts[0]["id"] == "mbpp/1"
        assert prompts[0]["prompt"] == "Write a function"
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/common/test_dataset_loader.py -v
```

期望：`ModuleNotFoundError: No module named 'wfcllm.common.dataset_loader'`

- [ ] **Step 3: 实现 `wfcllm/common/dataset_loader.py`**

```python
"""Shared dataset loading utility for HumanEval and MBPP."""
from __future__ import annotations

from pathlib import Path

from datasets import load_dataset

SUPPORTED_DATASETS = ("humaneval", "mbpp")


def load_prompts(dataset: str, dataset_path: str) -> list[dict]:
    """Load prompts from local HumanEval or MBPP dataset.

    Args:
        dataset: One of "humaneval" or "mbpp".
        dataset_path: Root directory containing local dataset caches.

    Returns:
        List of dicts with keys "id" (str) and "prompt" (str).

    Raises:
        ValueError: If dataset is not in SUPPORTED_DATASETS.
    """
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"dataset must be one of {SUPPORTED_DATASETS}, got '{dataset}'"
        )

    path = str(Path(dataset_path) / dataset)

    if dataset == "humaneval":
        ds = load_dataset(
            "openai/openai_humaneval",
            cache_dir=path,
            download_mode="reuse_cache_if_exists",
        )
        prompts = []
        for split in ds:
            for item in ds[split]:
                prompts.append({"id": item["task_id"], "prompt": item["prompt"]})
        return prompts

    # mbpp
    ds = load_dataset(
        "google-research-datasets/mbpp",
        "full",
        cache_dir=path,
        download_mode="reuse_cache_if_exists",
    )
    prompts = []
    for split in ds:
        for item in ds[split]:
            prompts.append({"id": f"mbpp/{item['task_id']}", "prompt": item["text"]})
    return prompts
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/common/test_dataset_loader.py -v
```

期望：全部 PASS

- [ ] **Step 5: 更新 `wfcllm/common/__init__.py`**

当前文件内容为空（1行）。完整替换为：

```python
from wfcllm.common.dataset_loader import SUPPORTED_DATASETS, load_prompts

__all__ = [
    "SUPPORTED_DATASETS",
    "load_prompts",
]
```

- [ ] **Step 6: 将 `wfcllm/watermark/pipeline.py` 改为使用共享实现**

打开 `wfcllm/watermark/pipeline.py`，做以下三处修改：

**（a）** 删除这两行：

```python
from datasets import load_dataset

SUPPORTED_DATASETS = ("humaneval", "mbpp")
```

**（b）** 在现有 import 区块中添加：

```python
from wfcllm.common.dataset_loader import SUPPORTED_DATASETS, load_prompts
```

**（c）** 将 `_load_prompts` 方法体替换为委托调用：

```python
def _load_prompts(self) -> list[dict]:
    """Load prompts from local dataset. Returns list of {"id", "prompt"}."""
    return load_prompts(self._config.dataset, self._config.dataset_path)
```

注意：`WatermarkPipelineConfig.__post_init__` 中 `if self.dataset not in SUPPORTED_DATASETS:` 这行现在引用的是从 `dataset_loader` 导入的 `SUPPORTED_DATASETS`，行为不变。

- [ ] **Step 7: 运行 watermark 及 common 相关测试，确认不回归**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -v -k "watermark or common"
```

期望：全部 PASS，无新失败

- [ ] **Step 8: Commit**

```bash
git add wfcllm/common/dataset_loader.py wfcllm/common/__init__.py tests/common/__init__.py tests/common/test_dataset_loader.py wfcllm/watermark/pipeline.py
git commit -m "refactor: extract shared dataset loading into wfcllm.common.dataset_loader"
```

---

## Chunk 2: 在 `wfcllm/extract/` 中实现负样本生成器

### Task 2: 创建 `wfcllm/extract/negative_corpus.py`

**Files:**
- Create: `wfcllm/extract/negative_corpus.py`
- Modify: `wfcllm/extract/__init__.py`
- Create: `tests/extract/test_negative_corpus.py`

将 `scripts/generate_negative_corpus.py` 的核心逻辑封装为 `NegativeCorpusGenerator` 类，与 `WatermarkPipeline` 结构对称。

- [ ] **Step 1: 写失败测试**

创建 `tests/extract/test_negative_corpus.py`：

```python
"""Tests for wfcllm.extract.negative_corpus."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from wfcllm.extract.negative_corpus import NegativeCorpusConfig, NegativeCorpusGenerator


class TestNegativeCorpusConfig:
    def test_default_values(self):
        cfg = NegativeCorpusConfig(
            lm_model_path="data/models/my-model",
            output_path="data/negative_corpus.jsonl",
        )
        assert cfg.dataset == "humaneval"
        assert cfg.dataset_path == "data/datasets"
        assert cfg.max_new_tokens == 512
        assert cfg.temperature == 0.8
        assert cfg.top_p == 0.95
        assert cfg.top_k == 50
        assert cfg.device == "cuda"
        assert cfg.limit is None

    def test_custom_values(self):
        cfg = NegativeCorpusConfig(
            lm_model_path="data/models/my-model",
            output_path="data/out.jsonl",
            dataset="mbpp",
            temperature=1.0,
            limit=5,
        )
        assert cfg.dataset == "mbpp"
        assert cfg.temperature == 1.0
        assert cfg.limit == 5

    def test_unsupported_dataset_raises(self):
        with pytest.raises(ValueError, match="dataset must be one of"):
            NegativeCorpusConfig(
                lm_model_path="x",
                output_path="y",
                dataset="unknown",
            )


class TestNegativeCorpusGeneratorGenerate:
    def test_generate_returns_string(self):
        """NegativeCorpusGenerator._generate() strips prompt tokens and decodes."""
        cfg = NegativeCorpusConfig(
            lm_model_path="data/models/my-model",
            output_path="data/out.jsonl",
            device="cpu",
        )
        gen = NegativeCorpusGenerator.__new__(NegativeCorpusGenerator)
        gen._config = cfg

        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 0
        # tokenizer(prompt) returns dict with input_ids of shape [1, 3]
        fake_inputs = {"input_ids": torch.zeros(1, 3, dtype=torch.long)}
        mock_tokenizer.return_value = fake_inputs
        mock_tokenizer.decode.return_value = "def foo(): pass"

        mock_model = MagicMock()
        # model.generate returns shape [1, 10]; first 3 tokens are prompt
        mock_model.generate.return_value = torch.zeros(1, 10, dtype=torch.long)

        gen._model = mock_model
        gen._tokenizer = mock_tokenizer

        result = gen._generate("def foo():", device="cpu")

        assert isinstance(result, str)
        assert result == "def foo(): pass"
        # Verify decode was called with the generated-only tokens (shape [7])
        decoded_arg = mock_tokenizer.decode.call_args[0][0]
        assert decoded_arg.shape == (7,)


class TestNegativeCorpusGeneratorRun:
    def test_run_writes_jsonl(self, tmp_path):
        """run() writes one JSONL record per prompt with correct fields."""
        cfg = NegativeCorpusConfig(
            lm_model_path="data/models/my-model",
            output_path=str(tmp_path / "out.jsonl"),
            device="cpu",
        )
        gen = NegativeCorpusGenerator.__new__(NegativeCorpusGenerator)
        gen._config = cfg
        gen._device = "cpu"
        gen._generate = MagicMock(side_effect=["def foo(): pass", "def bar(): pass"])

        prompts = [
            {"id": "HumanEval/0", "prompt": "def foo():"},
            {"id": "HumanEval/1", "prompt": "def bar():"},
        ]

        with patch("wfcllm.extract.negative_corpus.load_prompts", return_value=prompts):
            out_path = gen.run()

        assert Path(out_path).exists()
        lines = Path(out_path).read_text().strip().splitlines()
        assert len(lines) == 2

        record = json.loads(lines[0])
        assert record["id"] == "HumanEval/0"
        assert record["generated_code"] == "def foo(): pass"
        assert record["dataset"] == "humaneval"
        assert "prompt" in record

    def test_run_respects_limit(self, tmp_path):
        """run() processes only first `limit` prompts when limit is set."""
        cfg = NegativeCorpusConfig(
            lm_model_path="data/models/my-model",
            output_path=str(tmp_path / "out.jsonl"),
            device="cpu",
            limit=1,
        )
        gen = NegativeCorpusGenerator.__new__(NegativeCorpusGenerator)
        gen._config = cfg
        gen._device = "cpu"
        gen._generate = MagicMock(return_value="def foo(): pass")

        prompts = [
            {"id": "HumanEval/0", "prompt": "def foo():"},
            {"id": "HumanEval/1", "prompt": "def bar():"},
        ]

        with patch("wfcllm.extract.negative_corpus.load_prompts", return_value=prompts):
            gen.run()

        assert gen._generate.call_count == 1
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_negative_corpus.py -v
```

期望：`ModuleNotFoundError: No module named 'wfcllm.extract.negative_corpus'`

- [ ] **Step 3: 实现 `wfcllm/extract/negative_corpus.py`**

```python
"""Negative corpus generation for FPR threshold calibration.

Generates unwatermarked code using a code LLM directly, writing JSONL output
compatible with ThresholdCalibrator's corpus format.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from wfcllm.common.dataset_loader import SUPPORTED_DATASETS, load_prompts


@dataclass
class NegativeCorpusConfig:
    """Configuration for negative corpus generation."""

    lm_model_path: str
    """Path to the code-generation LLM (same as --lm-model-path in run.py)."""

    output_path: str
    """Output JSONL file path (e.g. data/negative_corpus.jsonl)."""

    dataset: str = "humaneval"
    """Dataset to use for prompts ("humaneval" or "mbpp")."""

    dataset_path: str = "data/datasets"
    """Local dataset root directory."""

    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    device: str = "cuda"
    limit: int | None = None
    """Process only first N prompts (for debugging). None = all."""

    def __post_init__(self):
        if self.dataset not in SUPPORTED_DATASETS:
            raise ValueError(
                f"dataset must be one of {SUPPORTED_DATASETS}, got '{self.dataset}'"
            )


class NegativeCorpusGenerator:
    """Generate unwatermarked code samples for negative corpus.

    Loads a code LLM and generates code for each prompt in the dataset
    without any watermarking, writing results to JSONL for use with
    ThresholdCalibrator.
    """

    def __init__(self, config: NegativeCorpusConfig):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self._config = config
        device = config.device
        if device == "cuda" and not torch.cuda.is_available():
            print("[警告] CUDA 不可用，回退到 CPU", file=sys.stderr)
            device = "cpu"
        self._device = device

        print(f"加载模型 {config.lm_model_path} ...", file=sys.stderr)
        self._tokenizer = AutoTokenizer.from_pretrained(config.lm_model_path)

        if device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                config.lm_model_path,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                config.lm_model_path,
                torch_dtype=torch.float32,
            ).to(device)
        self._model.eval()

    def _generate(self, prompt: str, device: str) -> str:
        """Generate code for a single prompt without watermarking."""
        import torch

        cfg = self._config
        inputs = self._tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True)

    def run(self) -> str:
        """Generate negative corpus and write to JSONL.

        Returns:
            Path to output JSONL file.
        """
        import torch

        cfg = self._config
        prompts = load_prompts(cfg.dataset, cfg.dataset_path)
        if cfg.limit is not None:
            prompts = prompts[: cfg.limit]
        print(f"  共 {len(prompts)} 条 prompt", file=sys.stderr)

        out_path = Path(cfg.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            for i, item in enumerate(prompts):
                try:
                    code = self._generate(item["prompt"], self._device)
                except Exception as e:
                    print(f"[警告] {item['id']} 生成失败：{e}", file=sys.stderr)
                    code = ""

                record = {
                    "id": item["id"],
                    "dataset": cfg.dataset,
                    "prompt": item["prompt"],
                    "generated_code": code,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                print(
                    f"  [{i + 1}/{len(prompts)}] {item['id']} | "
                    f"tokens: {len(code.split())}",
                    file=sys.stderr,
                )

        print(f"\n完成，输出至: {out_path}（{len(prompts)} 条）", file=sys.stderr)
        return str(out_path)
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_negative_corpus.py -v
```

期望：全部 PASS

- [ ] **Step 5: 更新 `wfcllm/extract/__init__.py`**

在文件末尾（现有最后一行 `from wfcllm.extract.pipeline import ...` 之后）追加以下两行：

```python
from wfcllm.extract.negative_corpus import NegativeCorpusConfig, NegativeCorpusGenerator
```

在 `__all__` 列表中追加（列表已有的内容保持不动，仅在末尾添加）：

```python
    "NegativeCorpusConfig",
    "NegativeCorpusGenerator",
```

具体来说，现有 `__init__.py` 的 `__all__` 是：

```python
__all__ = [
    "ExtractConfig",
    "DetectionResult",
    "BlockScore",
    "WatermarkDetector",
    "BlockScorer",
    "DPSelector",
    "HypothesisTester",
    "ThresholdCalibrator",
]
```

替换为：

```python
__all__ = [
    "ExtractConfig",
    "DetectionResult",
    "BlockScore",
    "WatermarkDetector",
    "BlockScorer",
    "DPSelector",
    "HypothesisTester",
    "ThresholdCalibrator",
    "NegativeCorpusConfig",
    "NegativeCorpusGenerator",
]
```

- [ ] **Step 6: 运行全部 extract 测试**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/ -v
```

期望：全部 PASS

- [ ] **Step 7: Commit**

```bash
git add wfcllm/extract/negative_corpus.py wfcllm/extract/__init__.py tests/extract/test_negative_corpus.py
git commit -m "feat: add NegativeCorpusGenerator to wfcllm.extract"
```

---

## Chunk 3: 更新 `run.py` — 新增 `generate-negative` phase

### Task 3: 在 `run.py` 中添加 `generate-negative` 阶段

**Files:**
- Modify: `run.py`
- Modify: `tests/test_run.py`

新增一个独立阶段 `generate-negative`，不依赖 encoder/watermark 阶段，可单独运行。

- [ ] **Step 1: 写失败测试**

在 `tests/test_run.py` 中，在 `TestRunState` 类末尾追加新测试，并新增 `TestRunGenerateNegative` 类。

找到 `TestRunState.test_phases_order` 方法，将其改为：

```python
def test_phases_order(self):
    assert PHASES == ["encoder", "watermark", "extract", "generate-negative"]
```

在 `TestCLI` 类之前，追加以下测试类：

```python
class TestRunGenerateNegative:
    def test_run_generate_negative_missing_lm_model_path(self, tmp_path):
        """run_generate_negative returns 1 when lm_model_path is missing."""
        from run import run_generate_negative, RunState

        state = RunState(tmp_path / "state.json")
        args = argparse.Namespace(
            lm_model_path=None,
            dataset=None,
            dataset_path=None,
            negative_output=None,
            negative_limit=None,
            config=Path("configs/base_config.json"),
        )
        # Temporarily patch base_config.json to have no lm_model_path
        import json as _json
        cfg_data = _json.loads(Path("configs/base_config.json").read_text())
        cfg_data["generate_negative"]["lm_model_path"] = ""
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(_json.dumps(cfg_data))
        args.config = cfg_path

        rc = run_generate_negative(args, state)
        assert rc == 1

    def test_run_generate_negative_calls_generator(self, tmp_path):
        """run_generate_negative calls NegativeCorpusGenerator.run() and marks done."""
        from unittest.mock import patch, MagicMock
        from run import run_generate_negative, RunState
        import json as _json

        state = RunState(tmp_path / "state.json")
        out_jsonl = str(tmp_path / "neg.jsonl")

        args = argparse.Namespace(
            lm_model_path="data/models/my-model",
            dataset="humaneval",
            dataset_path="data/datasets",
            negative_output=out_jsonl,
            negative_limit=None,
            config=Path("configs/base_config.json"),
        )

        mock_gen = MagicMock()
        mock_gen.run.return_value = out_jsonl

        with patch("run.NegativeCorpusGenerator", return_value=mock_gen) as MockCls:
            rc = run_generate_negative(args, state)

        assert rc == 0
        mock_gen.run.assert_called_once()
        assert state.is_done("generate-negative")
```

在文件顶部已有 `import sys, from pathlib import Path` 等，补充缺少的 import：

```python
import argparse
```

（若已存在则跳过）

- [ ] **Step 2: 运行测试，确认失败**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/test_run.py -v
```

期望：`test_phases_order` 和 `TestRunGenerateNegative` 相关测试失败

- [ ] **Step 3: 在 `run.py` 中更新 `PHASES`**

```python
PHASES = ["encoder", "watermark", "extract", "generate-negative"]
```

- [ ] **Step 4: 在 `run.py` 的 `build_parser()` 中追加参数**

在 Extract 参数区块之后追加：

```python
    # generate-negative 参数
    parser.add_argument(
        "--negative-output", default=None,
        help="负样本语料输出 JSONL 路径（默认从配置文件读取，或 data/negative_corpus.jsonl）",
    )
    parser.add_argument(
        "--negative-limit", type=int, default=None,
        help="只处理前 N 条 prompt（调试用，默认: 全量）",
    )
```

注意：`--lm-model-path`、`--dataset`、`--dataset-path` 已在 watermark 参数区块定义，直接复用。

- [ ] **Step 5: 在 `run.py` 的 `run_phase()` 中注册 runner**

在 `runners` dict 中追加：

```python
        "generate-negative": run_generate_negative,
```

- [ ] **Step 6: 在 `run.py` 中实现 `run_generate_negative()` 函数**

在 `run_extract()` 函数结尾后、`if __name__ == "__main__":` 前插入：

```python
def run_generate_negative(args: argparse.Namespace, state: RunState) -> int:
    """生成负样本语料：用 LLM 直接生成代码（不加水印）。

    输出 JSONL 格式与阶段二水印数据集相同（含 generated_code 字段），
    可直接作为 --calibration-corpus 传给 run.py --phase extract。
    """
    from wfcllm.extract.negative_corpus import NegativeCorpusConfig, NegativeCorpusGenerator

    print("=== 生成负样本语料 ===")

    cfg = load_config(args.config)
    neg_cfg = cfg.get("generate_negative", {})

    lm_model_path = args.lm_model_path or neg_cfg.get("lm_model_path", "")
    if not lm_model_path:
        print("[错误] --lm-model-path 为必填参数", file=sys.stderr)
        return 1

    dataset = args.dataset or neg_cfg.get("dataset", "humaneval")
    dataset_path = args.dataset_path or neg_cfg.get("dataset_path", "data/datasets")
    output_path = (
        getattr(args, "negative_output", None)
        or neg_cfg.get("output_path", "data/negative_corpus.jsonl")
    )
    limit = getattr(args, "negative_limit", None) or neg_cfg.get("limit", None)

    config = NegativeCorpusConfig(
        lm_model_path=lm_model_path,
        output_path=output_path,
        dataset=dataset,
        dataset_path=dataset_path,
        max_new_tokens=neg_cfg.get("max_new_tokens", 512),
        temperature=neg_cfg.get("temperature", 0.8),
        top_p=neg_cfg.get("top_p", 0.95),
        top_k=neg_cfg.get("top_k", 50),
        device=neg_cfg.get("device", "cuda"),
        limit=limit,
    )

    try:
        generator = NegativeCorpusGenerator(config)
        out_path = generator.run()
    except Exception as e:
        print(f"[错误] 负样本生成失败：{e}", file=sys.stderr)
        return 1

    state.mark_done("generate-negative", output_file=out_path, dataset=dataset)
    print(f"[完成] 负样本语料已保存至 {out_path}")
    return 0
```

- [ ] **Step 7: 运行测试，确认通过**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/test_run.py -v
```

期望：全部 PASS

- [ ] **Step 8: 验证 `--status` 和 `--help`**

```bash
conda run -n WFCLLM python run.py --status
conda run -n WFCLLM python run.py --help
```

期望：`--status` 显示四个阶段（含 `generate-negative`），`--help` 显示 `--negative-output` 和 `--negative-limit`

- [ ] **Step 9: Commit**

```bash
git add run.py tests/test_run.py
git commit -m "feat: add generate-negative phase to run.py"
```

---

## Chunk 4: 更新配置文件与 `scripts/generate_negative_corpus.py`

### Task 4: 更新 `configs/base_config.json`

**Files:**
- Modify: `configs/base_config.json`

在 `extract` section 之后新增 `generate_negative` section。注意 JSON 不允许尾随逗号，`extract` block 结尾需加逗号。

- [ ] **Step 1: 编辑配置文件**

当前文件最后几行（`extract` 是最后一个 section，无后续逗号）：

```json
  "extract": {
    ...
  }
}
```

替换为：

```json
  "extract": {
    "secret_key": "1010",
    "embed_dim": 128,
    "fpr_threshold": 3.0,
    "lsh_d": 3,
    "lsh_gamma": 0.5,
    "calibration_corpus": null,
    "fpr": 0.01,
    "input_file": "data/watermarked/humaneval_20260310_223459.jsonl",
    "output_dir": "data/results"
  },
  "generate_negative": {
    "lm_model_path": "data/models/deepseek-coder-7b-base",
    "dataset": "humaneval",
    "dataset_path": "data/datasets",
    "output_path": "data/negative_corpus.jsonl",
    "max_new_tokens": 512,
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 50,
    "device": "cuda",
    "limit": null
  }
}
```

- [ ] **Step 2: 验证 JSON 语法**

```bash
conda run -n WFCLLM python -c "import json; json.load(open('configs/base_config.json')); print('OK')"
```

期望：输出 `OK`

- [ ] **Step 3: Commit**

```bash
git add configs/base_config.json
git commit -m "chore: add generate_negative section to base_config.json"
```

### Task 5: 将 `scripts/generate_negative_corpus.py` 改为薄包装器

**Files:**
- Modify: `scripts/generate_negative_corpus.py`

原脚本逻辑已迁移至 `wfcllm/extract/negative_corpus.py`，脚本改为调用包内实现。

- [ ] **Step 1: 重写 `scripts/generate_negative_corpus.py`**

用以下内容完整替换原文件：

```python
#!/usr/bin/env python
"""生成负样本语料：用 LLM 直接生成代码（不加水印），用于 ThresholdCalibrator 校准。

输出 JSONL 格式与阶段二水印数据集相同（含 generated_code 字段），
可直接作为 --calibration-corpus 传给 run.py --phase extract。

推荐使用 run.py 入口：
    python run.py --phase generate-negative \\
        --lm-model-path data/models/deepseek-coder-7b-base \\
        --dataset humaneval \\
        --negative-output data/negative_corpus.jsonl

也可直接调用本脚本（仅保留作向后兼容）：
    python scripts/generate_negative_corpus.py \\
        --lm-model-path data/models/deepseek-coder-7b-base \\
        --dataset humaneval \\
        --dataset-path data/datasets \\
        --output data/negative_corpus.jsonl
"""
from __future__ import annotations

import argparse


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成负样本语料（LLM 直接生成，无水印）"
    )
    parser.add_argument("--lm-model-path", required=True, help="代码生成 LLM 路径")
    parser.add_argument(
        "--dataset", default="humaneval", choices=["humaneval", "mbpp"],
        help="数据集（默认: humaneval）",
    )
    parser.add_argument(
        "--dataset-path", default="data/datasets",
        help="本地数据集根目录（默认: data/datasets）",
    )
    parser.add_argument("--output", required=True, help="输出 JSONL 文件路径")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    from wfcllm.extract.negative_corpus import NegativeCorpusConfig, NegativeCorpusGenerator

    config = NegativeCorpusConfig(
        lm_model_path=args.lm_model_path,
        output_path=args.output,
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        device=args.device,
        limit=args.limit,
    )
    generator = NegativeCorpusGenerator(config)
    generator.run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证脚本 `--help` 正常工作**

```bash
conda run -n WFCLLM python scripts/generate_negative_corpus.py --help
```

期望：打印 argparse help，无 import 错误

- [ ] **Step 3: Commit**

```bash
git add scripts/generate_negative_corpus.py
git commit -m "refactor: scripts/generate_negative_corpus.py delegates to wfcllm.extract"
```

---

## Chunk 5: 更新 README.md

### Task 6: 在 README.md 中添加 `generate-negative` phase 说明

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 替换「生成负样本语料」命令示例**

找到 README 中如下内容（大约在「快速开始 — 单独运行某阶段」区块）：

```markdown
# 生成负样本语料（用同一 LLM 直接生成，不加水印）
python scripts/generate_negative_corpus.py \
    --lm-model-path data/models/deepseek-coder-7b \
    --dataset humaneval \
    --dataset-path data/datasets \
    --output data/negative_corpus.jsonl
```

替换为：

```markdown
# 生成负样本语料（方式一：通过 run.py，推荐）
python run.py --phase generate-negative \
    --lm-model-path data/models/deepseek-coder-7b-base \
    --dataset humaneval \
    --negative-output data/negative_corpus.jsonl

# 生成负样本语料（方式二：直接调用脚本，向后兼容）
python scripts/generate_negative_corpus.py \
    --lm-model-path data/models/deepseek-coder-7b-base \
    --dataset humaneval \
    --dataset-path data/datasets \
    --output data/negative_corpus.jsonl
```

- [ ] **Step 2: 在「Phase 3 — `wfcllm.extract`」的关键 API 列表末尾追加**

找到如下内容：

```markdown
- `ThresholdCalibrator.calibrate(corpus, fpr)` → 离线校准 FPR 阈值 M_r
```

在其后追加：

```markdown
- `NegativeCorpusConfig` — 负样本生成配置（lm_model_path, output_path, dataset, limit 等）
- `NegativeCorpusGenerator.run()` → 负样本 JSONL 路径（每行含 id/dataset/prompt/generated_code）
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document generate-negative phase and NegativeCorpusGenerator in README"
```

---

## Chunk 6: 运行全部测试并最终验证

### Task 7: 全量测试与端到端验证

**Files:** 无新文件

- [ ] **Step 1: 运行全量测试套件**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -v
```

期望：全部 PASS，无回归

- [ ] **Step 2: 验证 `run.py --status` 包含新 phase**

```bash
conda run -n WFCLLM python run.py --status
```

期望输出包含：

```
encoder     ○ 未完成
watermark   ○ 未完成
extract     ○ 未完成
generate-negative ○ 未完成
```

- [ ] **Step 3: 验证配置文件包含新 section**

```bash
conda run -n WFCLLM python -c "
import json
cfg = json.load(open('configs/base_config.json'))
assert 'generate_negative' in cfg, 'Missing generate_negative section'
assert cfg['generate_negative']['device'] == 'cuda'
print('generate_negative section OK:', cfg['generate_negative'])
"
```

期望：无报错，打印出 `generate_negative` section 内容

- [ ] **Step 4: 验证 `wfcllm.extract` 导出**

```bash
conda run -n WFCLLM python -c "
from wfcllm.extract import NegativeCorpusConfig, NegativeCorpusGenerator
cfg = NegativeCorpusConfig(lm_model_path='x', output_path='y')
print('OK:', cfg.dataset, cfg.temperature)
"
```

期望：打印 `OK: humaneval 0.8`

- [ ] **Step 5: 验证 `wfcllm.common` 导出**

```bash
conda run -n WFCLLM python -c "
from wfcllm.common import SUPPORTED_DATASETS, load_prompts
print('SUPPORTED_DATASETS:', SUPPORTED_DATASETS)
"
```

期望：打印 `SUPPORTED_DATASETS: ('humaneval', 'mbpp')`

- [ ] **Step 6: 验证脚本 `--help` 正常**

```bash
conda run -n WFCLLM python scripts/generate_negative_corpus.py --help
```

期望：打印 argparse help，无 import 错误
