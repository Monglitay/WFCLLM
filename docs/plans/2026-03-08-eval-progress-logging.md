# Eval Progress Logging Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add batch-level progress logging to `evaluate_only` so users see progress output during the embedding collection phase instead of a silent wait.

**Architecture:** The `evaluate_only` function in `wfcllm/encoder/train.py` iterates over `test_loader` batches with no feedback. We add a simple print-based progress counter (every N batches and at completion) — no tqdm dependency needed, just `print(..., flush=True)`.

**Tech Stack:** Python stdlib `print` with `flush=True`; pytest for tests.

---

### Task 1: Add progress logging to the `evaluate_only` batch loop

**Files:**
- Modify: `wfcllm/encoder/train.py:299-308`

The batch loop currently looks like:

```python
    all_anchor, all_positive, all_negative = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            a = model(batch["anchor_input_ids"].to(device), batch["anchor_attention_mask"].to(device))
            p = model(batch["positive_input_ids"].to(device), batch["positive_attention_mask"].to(device))
            neg = model(batch["negative_input_ids"].to(device), batch["negative_attention_mask"].to(device))
            all_anchor.append(a.cpu())
            all_positive.append(p.cpu())
            all_negative.append(neg.cpu())
```

**Step 1: Write the failing test**

File: `tests/encoder/test_train.py`

Add a test that captures stdout from `evaluate_only` and verifies progress lines are printed.
The test monkey-patches the model and data loader so it runs without GPU or real data.

```python
def test_evaluate_only_prints_progress(tmp_path, monkeypatch):
    """evaluate_only should print batch progress during embedding collection."""
    import io, sys
    import torch
    from unittest.mock import MagicMock, patch
    from wfcllm.encoder.train import evaluate_only
    from wfcllm.encoder.config import EncoderConfig

    # Minimal fake batch
    def fake_batch():
        ids = torch.zeros(4, 16, dtype=torch.long)
        mask = torch.ones(4, 16, dtype=torch.long)
        return {
            "anchor_input_ids": ids,
            "anchor_attention_mask": mask,
            "positive_input_ids": ids,
            "positive_attention_mask": mask,
            "negative_input_ids": ids,
            "negative_attention_mask": mask,
        }

    fake_loader = [fake_batch() for _ in range(6)]  # 6 batches

    # Fake model: returns normalized random embeddings
    fake_emb = torch.randn(4, 128)
    fake_emb = torch.nn.functional.normalize(fake_emb, dim=1)
    fake_model = MagicMock(return_value=fake_emb)
    fake_model.eval = MagicMock(return_value=fake_model)

    config = EncoderConfig(
        model_name="data/models/codet5-base",
        results_dir=str(tmp_path),
    )

    captured = io.StringIO()
    with (
        patch("wfcllm.encoder.train.load_code_samples", return_value=[]),
        patch("wfcllm.encoder.train.prepare_blocks_with_variants", return_value=[]),
        patch("wfcllm.encoder.train.build_triplets_from_blocks", return_value=[]),
        patch("wfcllm.encoder.train.TripletCodeDataset", return_value=MagicMock(__len__=lambda s: 6)),
        patch("wfcllm.encoder.train.random_split", return_value=(None, None, fake_loader)),
        patch("wfcllm.encoder.train.DataLoader", return_value=fake_loader),
        patch("wfcllm.encoder.train.AutoTokenizer.from_pretrained", return_value=MagicMock()),
        patch("wfcllm.encoder.train.SemanticEncoder", return_value=fake_model),
        patch("torch.load", return_value={"model_state_dict": {}}),
        patch.object(fake_model, "load_state_dict"),
        patch.object(fake_model, "to", return_value=fake_model),
        patch("sys.stdout", captured),
    ):
        try:
            evaluate_only("data/checkpoints/encoder/encoder_epoch9.pt", config)
        except Exception:
            pass  # metrics computation may fail on mock data; we only care about progress output

    output = captured.getvalue()
    assert "batch" in output.lower() or "/" in output, (
        "Expected progress output during batch loop, got:\n" + output
    )
```

**Step 2: Run test to verify it fails**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/encoder/test_train.py::test_evaluate_only_prints_progress -v
```

Expected: FAIL — the batch loop currently emits no progress output.

**Step 3: Add progress logging to the batch loop in `evaluate_only`**

In `wfcllm/encoder/train.py`, replace the batch loop (lines ~299–308) with:

```python
    # Collect embeddings
    all_anchor, all_positive, all_negative = [], [], []
    total_batches = len(test_loader)
    log_interval = max(1, total_batches // 10)  # log ~10 times
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            a = model(batch["anchor_input_ids"].to(device), batch["anchor_attention_mask"].to(device))
            p = model(batch["positive_input_ids"].to(device), batch["positive_attention_mask"].to(device))
            neg = model(batch["negative_input_ids"].to(device), batch["negative_attention_mask"].to(device))
            all_anchor.append(a.cpu())
            all_positive.append(p.cpu())
            all_negative.append(neg.cpu())
            if (i + 1) % log_interval == 0 or (i + 1) == total_batches:
                print(f"  Evaluating batch {i + 1}/{total_batches}", flush=True)
```

**Step 4: Run test to verify it passes**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/encoder/test_train.py::test_evaluate_only_prints_progress -v
```

Expected: PASS

**Step 5: Run full test suite to check no regressions**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -v
```

Expected: all tests pass.

**Step 6: Commit**

```bash
git add wfcllm/encoder/train.py tests/encoder/test_train.py
git commit -m "feat: add batch progress logging to evaluate_only"
```
