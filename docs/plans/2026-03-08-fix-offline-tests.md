# Fix Offline Test Failures Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 修复 `pytest tests/` 中所有与离线资源加载不符的测试，使其在 `HF_HUB_OFFLINE=1` 环境下全部通过。

**Architecture:** 问题根因是测试直接用 HF Hub ID `"Salesforce/codet5-base"` 创建 `SemanticEncoder`，而离线环境下 HF Hub 不可用。本地模型已存于 `data/models/codet5-base/`。修复策略：给测试的 `EncoderConfig` 传入本地路径，或在 fixture 中直接使用本地路径初始化模型。不修改生产代码。

**Tech Stack:** pytest, transformers (T5EncoderModel, AutoTokenizer), peft (LoRA), torch

---

## 当前失败汇总（HF_HUB_OFFLINE=1）

| 文件 | 问题 | 数量 |
|------|------|------|
| `tests/encoder/test_model.py` | `T5EncoderModel.from_pretrained("Salesforce/codet5-base")` 用 HF ID，离线报错 | 10 errors + 1 failure |
| `tests/encoder/test_trainer.py` | `SemanticEncoder(config=config)` 默认 `model_name` 是 HF ID | 2 errors |
| `tests/encoder/test_config.py` | `batch_size` 期望 32 但实际是 64；`lora_r` 期望 8 但实际是 16 | 2 failures |

**注意：**
- `tests/test_integration.py` 的 skip 是预期行为（`mbpp_blocks.json not found`），不需要修复
- `tests/encoder/test_dataset.py` 中的 `AutoTokenizer.from_pretrained("Salesforce/codet5-base")` 在离线下**不**报错（因为 HF_HUB_OFFLINE=1 时仍会从已缓存的全局 cache 读取），所以不需要修改

**本地资源路径：**
- 模型：`data/models/codet5-base/`（含 `config.json`、`pytorch_model.bin` 等）
- 实际 `EncoderConfig` 默认值（以实际代码为准）：
  - `batch_size = 64`（测试期望 32 —— 测试过时）
  - `lora_r = 16`（测试期望 8 —— 测试过时）
  - `lora_alpha = 32`（测试期望 16 —— 测试过时）

---

## Task 1: 修复 `tests/encoder/test_config.py` 中过时的默认值断言

**Files:**
- Modify: `tests/encoder/test_config.py`

**Step 1: 确认实际默认值**

```bash
grep -n "batch_size\|lora_r\|lora_alpha\|lora_dropout" wfcllm/encoder/config.py
```

Expected output 包含：`batch_size: int = 64`, `lora_r: int = 16`, `lora_alpha: int = 32`, `lora_dropout: float = 0.1`

**Step 2: 修改测试中的过时断言**

将 `tests/encoder/test_config.py` 的 `test_default_values` 和 `test_lora_defaults` 改为与实际代码一致：

```python
def test_default_values(self):
    cfg = EncoderConfig()
    assert cfg.model_name == "Salesforce/codet5-base"
    assert cfg.embed_dim == 128
    assert cfg.lr == 2e-5
    assert cfg.batch_size == 64          # 改：32 → 64
    assert cfg.epochs == 10
    assert cfg.margin == 0.3
    assert cfg.max_seq_length == 256
    assert cfg.warmup_ratio == 0.1
    assert cfg.early_stopping_patience == 3
    assert cfg.negative_ratio == 0.5

def test_lora_defaults(self):
    cfg = EncoderConfig()
    assert cfg.use_lora is True
    assert cfg.lora_r == 16             # 改：8 → 16
    assert cfg.lora_alpha == 32         # 改：16 → 32
    assert cfg.lora_dropout == 0.1
```

**Step 3: 运行测试确认通过**

```bash
conda run -n WFCLLM pytest tests/encoder/test_config.py -v
```

Expected: `9 passed`

**Step 4: Commit**

```bash
git add tests/encoder/test_config.py
git commit -m "fix: update test_config to match actual EncoderConfig defaults"
```

---

## Task 2: 修复 `tests/encoder/test_model.py` — 使用本地模型路径

**Files:**
- Modify: `tests/encoder/test_model.py`

**根因分析：**

所有 `SemanticEncoder(config=config)` 调用都用默认 `model_name="Salesforce/codet5-base"`（HF Hub ID）。离线时 `T5EncoderModel.from_pretrained` 找不到文件。

修复方案：在 module 级别定义 `LOCAL_MODEL_PATH` 常量并在所有 fixture 中使用。

**Step 1: 修改 `tests/encoder/test_model.py`**

在文件顶部增加常量，并修改所有 `EncoderConfig` 和 tokenizer fixture：

```python
"""Tests for wfcllm.encoder.model."""

import pytest
import torch
from transformers import AutoTokenizer

from wfcllm.encoder.model import SemanticEncoder
from wfcllm.encoder.config import EncoderConfig

# 使用本地离线模型，避免 HF Hub 访问
LOCAL_MODEL = "data/models/codet5-base"


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained(LOCAL_MODEL)


class TestSemanticEncoderFullFinetune:
    """Tests with LoRA disabled (full finetune, FP32)."""

    @pytest.fixture
    def model(self):
        config = EncoderConfig(model_name=LOCAL_MODEL, use_lora=False, use_bf16=False, embed_dim=128)
        return SemanticEncoder(config=config)

    def test_output_shape(self, model, tokenizer):
        inputs = tokenizer("x = 1", return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            output = model(inputs["input_ids"], inputs["attention_mask"])
        assert output.shape == (1, 128)

    def test_output_normalized(self, model, tokenizer):
        inputs = tokenizer("x = 1", return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            output = model(inputs["input_ids"], inputs["attention_mask"])
        norms = torch.norm(output, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_batch_input(self, model, tokenizer):
        texts = ["x = 1", "y = 2", "for i in range(10):\n    print(i)"]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            output = model(inputs["input_ids"], inputs["attention_mask"])
        assert output.shape == (3, 128)

    def test_all_params_trainable(self, model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable == total


class TestSemanticEncoderLoRA:
    """Tests with LoRA enabled (default config)."""

    @pytest.fixture
    def model(self):
        config = EncoderConfig(model_name=LOCAL_MODEL, use_lora=True, use_bf16=False, embed_dim=128)
        return SemanticEncoder(config=config)

    def test_output_shape(self, model, tokenizer):
        inputs = tokenizer("x = 1", return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            output = model(inputs["input_ids"], inputs["attention_mask"])
        assert output.shape == (1, 128)

    def test_output_normalized(self, model, tokenizer):
        inputs = tokenizer("x = 1", return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            output = model(inputs["input_ids"], inputs["attention_mask"])
        norms = torch.norm(output, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_fewer_trainable_params(self, model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable < total, "LoRA should freeze most parameters"
        ratio = trainable / total
        assert ratio < 0.10, f"Expected <10% trainable params, got {ratio:.2%}"

    def test_deterministic(self, model, tokenizer):
        inputs = tokenizer("x = 1", return_tensors="pt", padding=True, truncation=True, max_length=256)
        model.eval()
        with torch.no_grad():
            out1 = model(inputs["input_ids"], inputs["attention_mask"])
            out2 = model(inputs["input_ids"], inputs["attention_mask"])
        assert torch.allclose(out1, out2)


class TestSemanticEncoderBF16:
    """Tests with BF16 enabled."""

    @pytest.fixture
    def model(self):
        config = EncoderConfig(model_name=LOCAL_MODEL, use_lora=False, use_bf16=True, embed_dim=64)
        return SemanticEncoder(config=config)

    def test_encoder_dtype(self, model):
        param = next(model.encoder.parameters())
        assert param.dtype == torch.bfloat16

    def test_output_is_float32(self, model, tokenizer):
        inputs = tokenizer("x = 1", return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            output = model(inputs["input_ids"], inputs["attention_mask"])
        assert output.dtype == torch.float32

    def test_different_embed_dim(self, tokenizer):
        config = EncoderConfig(model_name=LOCAL_MODEL, use_lora=False, use_bf16=False, embed_dim=64)
        model = SemanticEncoder(config=config)
        inputs = tokenizer("x = 1", return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            output = model(inputs["input_ids"], inputs["attention_mask"])
        assert output.shape == (1, 64)
```

**Step 2: 运行测试确认通过**

```bash
conda run -n WFCLLM pytest tests/encoder/test_model.py -v
```

Expected: `11 passed`（可能较慢，每个 fixture 需加载模型）

**Step 3: Commit**

```bash
git add tests/encoder/test_model.py
git commit -m "fix: use local codet5-base path in test_model to support offline mode"
```

---

## Task 3: 修复 `tests/encoder/test_trainer.py` — 使用本地模型路径

**Files:**
- Modify: `tests/encoder/test_trainer.py`

**根因：** `dummy_setup` fixture 中 `SemanticEncoder(config=config)` 使用了默认 `model_name="Salesforce/codet5-base"`。

**Step 1: 修改 `dummy_setup` fixture**

在文件顶部加常量，修改 `EncoderConfig` 调用：

```python
# 文件顶部（imports 之后）加入：
LOCAL_MODEL = "data/models/codet5-base"
```

```python
@pytest.fixture
def dummy_setup(self):
    """Create minimal trainer with dummy data for smoke testing."""
    from wfcllm.encoder.model import SemanticEncoder
    config = EncoderConfig(
        model_name=LOCAL_MODEL,          # 新增：使用本地路径
        embed_dim=32, epochs=1, batch_size=2, lr=1e-4,
        use_lora=False, use_bf16=False,
        checkpoint_dir="/tmp/wfcllm_test_ckpt",
        results_dir="/tmp/wfcllm_test_results",
    )
    model = SemanticEncoder(config=config)
    # ... 其余不变
```

**Step 2: 运行测试确认通过**

```bash
conda run -n WFCLLM pytest tests/encoder/test_trainer.py -v
```

Expected: `5 passed`

**Step 3: Commit**

```bash
git add tests/encoder/test_trainer.py
git commit -m "fix: use local codet5-base path in test_trainer to support offline mode"
```

---

## Task 4: 全量验证

**Step 1: 以离线模式运行全套测试**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -v --tb=short 2>&1 | tail -20
```

Expected:
- 无 FAILED，无 ERROR
- `tests/test_integration.py::test_integration_first_5_samples` 仍为 SKIPPED（正常）
- 通过数约 256（原来 242 passed + 修复的 14 个）

**Step 2: 如有意外失败，定位并修复后再次运行**

**Step 3: Commit（如 Task 4 中有额外小修改）**

```bash
git add -p   # 只 stage 确认修改
git commit -m "fix: resolve remaining offline test issues"
```

---

## 注意事项

1. **`data/models/codet5-base/` 路径是相对路径**，测试必须从项目根目录 `/root/autodl-tmp/WFCLLM` 运行（`pytest tests/` 默认如此）
2. **不修改生产代码**（`wfcllm/encoder/model.py` 等），只修改测试文件
3. `tests/encoder/test_dataset.py` 的 `AutoTokenizer.from_pretrained("Salesforce/codet5-base")` 在离线下已能正常工作（使用全局 HF cache），无需修改
