# Encoder Best Model Export Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Automatically export the best encoder model to `data/models/encoder/best_model.pt` during training, and have watermark/extract/eval-only stages load from this fixed path by default.

**Architecture:** Add `output_model_dir` to `EncoderConfig`; extend `ContrastiveTrainer.train()` to call `_export_best_model()` whenever a new best val_loss is found; update `run.py` watermark/extract/eval-only loading logic to prefer `best_model.pt`; migrate the existing best checkpoint.

**Tech Stack:** Python, PyTorch, dataclasses

---

### Task 1: Add `output_model_dir` to `EncoderConfig`

**Files:**
- Modify: `wfcllm/encoder/config.py:43-47`
- Test: `tests/encoder/test_config.py` (create if not exists)

**Step 1: Write the failing test**

```python
# tests/encoder/test_config.py
from wfcllm.encoder.config import EncoderConfig

def test_output_model_dir_default():
    config = EncoderConfig()
    assert config.output_model_dir == "data/models/encoder"

def test_output_model_dir_custom():
    config = EncoderConfig(output_model_dir="custom/path")
    assert config.output_model_dir == "custom/path"
```

**Step 2: Run test to verify it fails**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/encoder/test_config.py -v
```
Expected: FAIL with `AttributeError: 'EncoderConfig' object has no attribute 'output_model_dir'`

**Step 3: Add the field to `EncoderConfig`**

In `wfcllm/encoder/config.py`, after the existing paths block (line 47), add:

```python
    output_model_dir: str = "data/models/encoder"  # 最优模型导出目录
```

The paths block should look like:
```python
    # Paths
    checkpoint_dir: str = "data/checkpoints/encoder"
    results_dir: str = "data/results"
    local_model_dir: str = "data/models"      # 本地模型根目录（离线部署）
    local_dataset_dir: str = "data/datasets"  # 本地数据集根目录（离线部署）
    output_model_dir: str = "data/models/encoder"  # 最优模型导出目录
```

**Step 4: Run test to verify it passes**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/encoder/test_config.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add wfcllm/encoder/config.py tests/encoder/test_config.py
git commit -m "feat: add output_model_dir to EncoderConfig"
```

---

### Task 2: Export best model in `ContrastiveTrainer`

**Files:**
- Modify: `wfcllm/encoder/trainer.py:134-175`
- Test: `tests/encoder/test_trainer.py` (create if not exists)

**Step 1: Write the failing test**

```python
# tests/encoder/test_trainer.py
import tempfile
from pathlib import Path
import torch
from unittest.mock import MagicMock, patch
from torch.utils.data import DataLoader, TensorDataset

from wfcllm.encoder.config import EncoderConfig
from wfcllm.encoder.trainer import ContrastiveTrainer


def _make_dummy_loader():
    """3 batches, each with anchor/positive/negative input_ids + attention_mask."""
    B, L = 4, 16
    data = TensorDataset(
        torch.ones(12, L, dtype=torch.long),   # anchor_input_ids
        torch.ones(12, L, dtype=torch.long),   # anchor_attention_mask
        torch.ones(12, L, dtype=torch.long),   # positive_input_ids
        torch.ones(12, L, dtype=torch.long),   # positive_attention_mask
        torch.ones(12, L, dtype=torch.long),   # negative_input_ids
        torch.ones(12, L, dtype=torch.long),   # negative_attention_mask
    )
    # Return raw tensors as batches manually
    return [
        {
            "anchor_input_ids": torch.ones(B, L, dtype=torch.long),
            "anchor_attention_mask": torch.ones(B, L, dtype=torch.long),
            "positive_input_ids": torch.ones(B, L, dtype=torch.long),
            "positive_attention_mask": torch.ones(B, L, dtype=torch.long),
            "negative_input_ids": torch.ones(B, L, dtype=torch.long),
            "negative_attention_mask": torch.ones(B, L, dtype=torch.long),
        }
        for _ in range(3)
    ]


def test_export_best_model_creates_file():
    """ContrastiveTrainer should export best_model.pt to output_model_dir."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = EncoderConfig(
            model_name="data/models/codet5-base",
            epochs=1,
            use_lora=False,
            use_bf16=False,
            checkpoint_dir=str(Path(tmpdir) / "checkpoints"),
            output_model_dir=str(Path(tmpdir) / "output"),
        )
        from wfcllm.encoder.model import SemanticEncoder
        model = SemanticEncoder(config=config)

        batches = _make_dummy_loader()
        trainer = ContrastiveTrainer(model, batches, batches, config)
        trainer.train()

        best_model_path = Path(tmpdir) / "output" / "best_model.pt"
        assert best_model_path.exists(), "best_model.pt should be created"
        ckpt = torch.load(best_model_path, map_location="cpu")
        assert "model_state_dict" in ckpt
        assert "config" in ckpt
        assert "best_metric" in ckpt
        assert "epoch" in ckpt
```

**Step 2: Run test to verify it fails**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/encoder/test_trainer.py::test_export_best_model_creates_file -v
```
Expected: FAIL — `best_model.pt` not found

**Step 3: Add `_export_best_model()` and call it in `train()`**

In `wfcllm/encoder/trainer.py`:

After `save_checkpoint()` (line 145), add:

```python
    def _export_best_model(self, epoch: int, best_val_loss: float) -> Path:
        """Export best model weights to output_model_dir/best_model.pt."""
        import dataclasses
        output_dir = Path(self.config.output_model_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "best_model.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": dataclasses.asdict(self.config),
                "best_metric": best_val_loss,
                "epoch": epoch,
            },
            path,
        )
        return path
```

In the `train()` method, after the line `self.save_checkpoint(epoch, metrics)` (line 168), add:

```python
                export_path = self._export_best_model(epoch, best_val_loss)
                print(f"[导出] 最优模型已保存至 {export_path}")
```

The updated `train()` block should look like:
```python
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                patience_counter = 0
                best_metrics = metrics
                self.save_checkpoint(epoch, metrics)
                export_path = self._export_best_model(epoch, best_val_loss)
                print(f"[导出] 最优模型已保存至 {export_path}")
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
```

**Step 4: Run test to verify it passes**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/encoder/test_trainer.py::test_export_best_model_creates_file -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add wfcllm/encoder/trainer.py tests/encoder/test_trainer.py
git commit -m "feat: export best model to output_model_dir in ContrastiveTrainer"
```

---

### Task 3: Update `configs/base_config.json`

**Files:**
- Modify: `configs/base_config.json:22-25`

**Step 1: Add `output_model_dir` to encoder section**

In `configs/base_config.json`, inside the `"encoder"` object after `"local_dataset_dir"`, add:

```json
    "output_model_dir": "data/models/encoder"
```

The end of the encoder section should look like:
```json
    "checkpoint_dir": "data/checkpoints/encoder",
    "results_dir": "data/results",
    "local_model_dir": "data/models",
    "local_dataset_dir": "data/datasets",
    "output_model_dir": "data/models/encoder"
  },
```

**Step 2: Verify JSON is valid**

```bash
python -c "import json; json.load(open('configs/base_config.json'))" && echo "OK"
```
Expected: `OK`

**Step 3: Commit**

```bash
git add configs/base_config.json
git commit -m "chore: add output_model_dir to base_config.json encoder section"
```

---

### Task 4: Update `run.py` — encoder phase records `best_model_path`

**Files:**
- Modify: `run.py:286-292`

**Step 1: After training completes, record `best_model_path` in `run_state`**

In `run_encoder()`, replace the `state.mark_done` call (lines 291-292) with:

```python
    # best_model.pt 固定路径
    best_model_path = str(Path(config.output_model_dir) / "best_model.pt")

    # 找到最新的 checkpoint 文件（向后兼容）
    ckpt_pattern = str(Path(config.checkpoint_dir) / "encoder_epoch*.pt")
    checkpoints = sorted(glob.glob(ckpt_pattern))
    checkpoint_path = checkpoints[-1] if checkpoints else config.checkpoint_dir

    state.mark_done("encoder", checkpoint=checkpoint_path, best_model_path=best_model_path)
    print(f"[完成] 编码器训练完毕，最优模型: {best_model_path}")
```

**Step 2: Verify syntax**

```bash
python -c "import run" && echo "OK"
```
Expected: `OK`

**Step 3: Commit**

```bash
git add run.py
git commit -m "feat: record best_model_path in run_state after encoder training"
```

---

### Task 5: Update `run.py` — watermark phase loads from `best_model.pt`

**Files:**
- Modify: `run.py:334-347`

**Step 1: Replace checkpoint loading logic in `run_watermark()`**

Find this block in `run_watermark()` (lines 334-347):
```python
    encoder_checkpoint = state.get("encoder", "checkpoint")

    # 加载编码器
    enc_config = EncoderConfig(embed_dim=embed_dim)
    local_codet5 = Path(enc_config.local_model_dir) / "codet5-base"
    if local_codet5.exists() and (local_codet5 / "config.json").exists():
        enc_config.model_name = str(local_codet5)
        print(f"[自动] 编码器使用本地模型: {enc_config.model_name}")
    else:
        print(f"[回退] 编码器使用 HF Hub: {enc_config.model_name}")
    encoder = SemanticEncoder(config=enc_config)
    if encoder_checkpoint and Path(encoder_checkpoint).exists():
        ckpt = torch.load(encoder_checkpoint, map_location="cpu")
        encoder.load_state_dict(ckpt["model_state_dict"])
```

Replace with:
```python
    # 加载编码器：优先 best_model.pt，回退 checkpoint
    enc_config = EncoderConfig(embed_dim=embed_dim)
    local_codet5 = Path(enc_config.local_model_dir) / "codet5-base"
    if local_codet5.exists() and (local_codet5 / "config.json").exists():
        enc_config.model_name = str(local_codet5)
        print(f"[自动] 编码器使用本地模型: {enc_config.model_name}")
    else:
        print(f"[回退] 编码器使用 HF Hub: {enc_config.model_name}")
    encoder = SemanticEncoder(config=enc_config)

    best_model_path = state.get("encoder", "best_model_path") or str(
        Path(enc_config.output_model_dir) / "best_model.pt"
    )
    encoder_checkpoint = state.get("encoder", "checkpoint")
    if Path(best_model_path).exists():
        ckpt = torch.load(best_model_path, map_location="cpu")
        encoder.load_state_dict(ckpt["model_state_dict"])
        print(f"[加载] 编码器权重来自: {best_model_path}")
    elif encoder_checkpoint and Path(encoder_checkpoint).exists():
        ckpt = torch.load(encoder_checkpoint, map_location="cpu")
        encoder.load_state_dict(ckpt["model_state_dict"])
        print(f"[加载] 编码器权重来自 checkpoint（fallback）: {encoder_checkpoint}")
    else:
        print("[警告] 未找到微调权重，使用预训练模型")
```

**Step 2: Verify syntax**

```bash
python -c "import run" && echo "OK"
```
Expected: `OK`

**Step 3: Commit**

```bash
git add run.py
git commit -m "feat: load encoder best_model.pt in watermark phase"
```

---

### Task 6: Update `run.py` — extract phase loads from `best_model.pt`

**Files:**
- Modify: `run.py:424-437`

**Step 1: Replace checkpoint loading logic in `run_extract()`**

Find this block in `run_extract()` (lines 424-440):
```python
    encoder_checkpoint = state.get("encoder", "checkpoint")

    # 加载编码器
    enc_config = EncoderConfig(embed_dim=embed_dim)
    local_codet5 = Path(enc_config.local_model_dir) / "codet5-base"
    if local_codet5.exists() and (local_codet5 / "config.json").exists():
        enc_config.model_name = str(local_codet5)
        print(f"[自动] 编码器使用本地模型: {enc_config.model_name}")
    else:
        print(f"[回退] 编码器使用 HF Hub: {enc_config.model_name}")
    encoder = SemanticEncoder(config=enc_config)
    if encoder_checkpoint and Path(encoder_checkpoint).exists():
        ckpt = torch.load(encoder_checkpoint, map_location="cpu")
        encoder.load_state_dict(ckpt["model_state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = encoder.to(device)
    tokenizer = AutoTokenizer.from_pretrained(enc_config.model_name)
```

Replace with:
```python
    # 加载编码器：优先 best_model.pt，回退 checkpoint
    enc_config = EncoderConfig(embed_dim=embed_dim)
    local_codet5 = Path(enc_config.local_model_dir) / "codet5-base"
    if local_codet5.exists() and (local_codet5 / "config.json").exists():
        enc_config.model_name = str(local_codet5)
        print(f"[自动] 编码器使用本地模型: {enc_config.model_name}")
    else:
        print(f"[回退] 编码器使用 HF Hub: {enc_config.model_name}")
    encoder = SemanticEncoder(config=enc_config)

    best_model_path = state.get("encoder", "best_model_path") or str(
        Path(enc_config.output_model_dir) / "best_model.pt"
    )
    encoder_checkpoint = state.get("encoder", "checkpoint")
    if Path(best_model_path).exists():
        ckpt = torch.load(best_model_path, map_location="cpu")
        encoder.load_state_dict(ckpt["model_state_dict"])
        print(f"[加载] 编码器权重来自: {best_model_path}")
    elif encoder_checkpoint and Path(encoder_checkpoint).exists():
        ckpt = torch.load(encoder_checkpoint, map_location="cpu")
        encoder.load_state_dict(ckpt["model_state_dict"])
        print(f"[加载] 编码器权重来自 checkpoint（fallback）: {encoder_checkpoint}")
    else:
        print("[警告] 未找到微调权重，使用预训练模型")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = encoder.to(device)
    tokenizer = AutoTokenizer.from_pretrained(enc_config.model_name)
```

**Step 2: Verify syntax**

```bash
python -c "import run" && echo "OK"
```
Expected: `OK`

**Step 3: Commit**

```bash
git add run.py
git commit -m "feat: load encoder best_model.pt in extract phase"
```

---

### Task 7: Update `run.py` — eval-only loads from `best_model.pt`

**Files:**
- Modify: `run.py:225-251`

**Step 1: Update eval-only checkpoint resolution in `run_encoder()`**

Find this block (lines 228-234):
```python
        checkpoint = args.checkpoint or state.get("encoder", "checkpoint")
        if not checkpoint:
            print("[错误] 未找到 checkpoint，请用 --checkpoint 指定路径", file=sys.stderr)
            return 1
        if not Path(checkpoint).exists():
            print(f"[错误] checkpoint 不存在：{checkpoint}", file=sys.stderr)
            return 1
```

Replace with:
```python
        # 优先顺序：CLI --checkpoint > best_model.pt > run_state checkpoint
        default_best = str(Path(EncoderConfig().output_model_dir) / "best_model.pt")
        checkpoint = (
            args.checkpoint
            or (default_best if Path(default_best).exists() else None)
            or state.get("encoder", "checkpoint")
        )
        if not checkpoint:
            print("[错误] 未找到 checkpoint，请用 --checkpoint 指定路径", file=sys.stderr)
            return 1
        if not Path(checkpoint).exists():
            print(f"[错误] checkpoint 不存在：{checkpoint}", file=sys.stderr)
            return 1
        print(f"[评测] 使用模型: {checkpoint}")
```

**Step 2: Verify syntax**

```bash
python -c "import run" && echo "OK"
```
Expected: `OK`

**Step 3: Commit**

```bash
git add run.py
git commit -m "feat: eval-only prefers best_model.pt over checkpoint"
```

---

### Task 8: Migrate existing best model to `data/models/encoder/best_model.pt`

**Files:**
- Run one-off migration script (no permanent file created)

**Step 1: Run migration**

```bash
conda run -n WFCLLM python - <<'EOF'
import dataclasses, torch
from pathlib import Path
from wfcllm.encoder.config import EncoderConfig

src = Path("data/checkpoints/encoder/encoder_epoch10.pt")
config = EncoderConfig()
dest = Path(config.output_model_dir) / "best_model.pt"
dest.parent.mkdir(parents=True, exist_ok=True)

ckpt = torch.load(src, map_location="cpu")
torch.save({
    "model_state_dict": ckpt["model_state_dict"],
    "config": dataclasses.asdict(config),
    "best_metric": ckpt["metrics"]["val_loss"],
    "epoch": ckpt["metrics"]["epoch"],
}, dest)
print(f"迁移完成: {dest}")
print(f"  epoch={ckpt['metrics']['epoch']}, val_loss={ckpt['metrics']['val_loss']:.6f}")
EOF
```
Expected: `迁移完成: data/models/encoder/best_model.pt`

**Step 2: Verify the exported file loads correctly**

```bash
conda run -n WFCLLM python - <<'EOF'
import torch
from pathlib import Path
from wfcllm.encoder.config import EncoderConfig
from wfcllm.encoder.model import SemanticEncoder

dest = Path("data/models/encoder/best_model.pt")
ckpt = torch.load(dest, map_location="cpu")
print("keys:", list(ckpt.keys()))
print("epoch:", ckpt["epoch"])
print("best_metric (val_loss):", ckpt["best_metric"])

config = EncoderConfig()
model = SemanticEncoder(config=config)
model.load_state_dict(ckpt["model_state_dict"])
print("模型加载成功 ✓")
EOF
```
Expected: `模型加载成功 ✓`

**Step 3: Update run_state.json to include best_model_path**

```bash
conda run -n WFCLLM python - <<'EOF'
import json
from pathlib import Path

state_path = Path("data/run_state.json")
if state_path.exists():
    data = json.loads(state_path.read_text())
    if data.get("encoder", {}).get("done"):
        data["encoder"]["best_model_path"] = "data/models/encoder/best_model.pt"
        state_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print("run_state.json 已更新")
    else:
        print("encoder 阶段尚未标记为完成，跳过")
else:
    print("run_state.json 不存在，跳过")
EOF
```

**Step 4: Commit**

```bash
git add data/models/encoder/best_model.pt data/run_state.json 2>/dev/null || true
# best_model.pt is large — check if it's tracked or gitignored
git status data/models/encoder/
```

If `data/models/` is gitignored (large binary), only commit `run_state.json`:
```bash
git add data/run_state.json
git commit -m "chore: migrate best encoder model to data/models/encoder/best_model.pt"
```

---

### Task 9: Run full test suite

**Step 1: Run all tests**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -v
```
Expected: all tests pass

**Step 2: Commit if any test files were modified**

```bash
git add tests/
git commit -m "test: update encoder tests for output_model_dir"
```
