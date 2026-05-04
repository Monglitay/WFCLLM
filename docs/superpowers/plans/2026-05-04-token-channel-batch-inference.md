# Token Channel Batch Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Optimize teacher model inference from 67% GPU utilization to 90%+ through two-level parallel batch inference

**Architecture:** Two-level parallelism - (1) intra-text: single forward pass for all token positions using causal mask, (2) inter-text: batch multiple texts together with dynamic batch sizing based on sequence length

**Tech Stack:** PyTorch, Transformers, DeepSeek-Coder-7B, tqdm

---

## File Structure

**New files:**
- `tests/watermark/token_channel/test_teacher_batch.py` - Batch inference tests

**Modified files:**
- `wfcllm/watermark/token_channel/teacher.py` - Add batch inference functions
- `wfcllm/watermark/token_channel/train_corpus.py` - Use batch inference
- `wfcllm/watermark/token_channel/train_workflow.py` - Add batch_size config

---

## Task 1: Intra-Text Parallel Inference (Single Text All Positions)

**Files:**
- Modify: `wfcllm/watermark/token_channel/teacher.py`
- Test: `tests/watermark/token_channel/test_teacher_batch.py`

- [ ] **Step 1: Write failing test for single text all positions**

Create `tests/watermark/token_channel/test_teacher_batch.py`:

```python
"""Tests for batch teacher inference."""

import torch
import pytest

from wfcllm.watermark.token_channel.teacher import (
    _extract_single_text_all_positions,
    extract_teacher_rows,
)


def test_extract_single_text_all_positions_shape(mock_tokenizer, mock_model):
    """Test that single text extraction returns correct shape."""
    token_ids = [1, 2, 3, 4, 5]

    logits = _extract_single_text_all_positions(
        model=mock_model,
        token_ids=token_ids,
        tokenizer=mock_tokenizer,
    )

    # Should return [seq_len, vocab_size]
    assert logits.shape == (len(token_ids), mock_model.config.vocab_size)
    assert logits.dtype == torch.float32


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing."""
    class MockTokenizer:
        bos_token_id = 1
        pad_token_id = 0
        eos_token_id = 2
        vocab_size = 32000
    return MockTokenizer()


@pytest.fixture
def mock_model():
    """Mock model for testing."""
    class MockConfig:
        vocab_size = 32000

    class MockModel:
        config = MockConfig()
        training = False

        def __call__(self, input_ids, attention_mask=None):
            batch_size, seq_len = input_ids.shape
            logits = torch.randn(batch_size, seq_len, self.config.vocab_size)

            class Output:
                pass
            output = Output()
            output.logits = logits
            return output

        def parameters(self):
            return iter([torch.tensor([1.0])])

        def eval(self):
            self.training = False

    return MockModel()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/token_channel/test_teacher_batch.py::test_extract_single_text_all_positions_shape -v`

Expected: FAIL with "_extract_single_text_all_positions not found"

- [ ] **Step 3: Implement single text all positions extraction**

Add to `wfcllm/watermark/token_channel/teacher.py`:

```python
def _extract_single_text_all_positions(
    model: object,
    token_ids: list[int],
    tokenizer: object,
) -> torch.Tensor:
    """Extract logits for all token positions in a single text (intra-text parallel).

    Uses causal language model property: one forward pass gives logits for all positions.

    Args:
        model: Causal language model
        token_ids: Token IDs for the text
        tokenizer: Tokenizer (for bos_token_id if needed)

    Returns:
        Tensor of shape [seq_len, vocab_size] with logits for each position
    """
    if not token_ids:
        return torch.empty((0, _get_vocab_size(model)), dtype=torch.float32)

    # Prepare input: prepend bos_token if model expects it
    input_tokens = token_ids
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is not None and isinstance(bos_token_id, int):
        input_tokens = [bos_token_id] + token_ids

    # Single forward pass
    input_ids = torch.tensor(
        [input_tokens],
        dtype=torch.long,
        device=_resolve_module_device(model),
    )

    was_training = getattr(model, "training", None)
    eval_method = getattr(model, "eval", None)
    train_method = getattr(model, "train", None)

    if callable(eval_method):
        eval_method()

    try:
        with torch.no_grad():
            output = model(input_ids)
    finally:
        if was_training is not None and callable(train_method):
            train_method(was_training)

    # Extract logits
    logits = getattr(output, "logits", None)
    if logits is None and isinstance(output, dict):
        logits = output.get("logits")
    if logits is None:
        logits = output if isinstance(output, torch.Tensor) else None
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise ValueError("model must return 3D logits [batch, seq, vocab]")

    # Extract logits for each position: logits[0, i] predicts token at position i+1
    # If we prepended bos, logits[0, 0] predicts token_ids[0], logits[0, 1] predicts token_ids[1], etc.
    if bos_token_id is not None and isinstance(bos_token_id, int):
        position_logits = logits[0, :len(token_ids)]  # Skip the last prediction
    else:
        position_logits = logits[0, :-1]  # Skip the last prediction (no next token)

    return position_logits.detach().cpu().to(dtype=torch.float32)


def _get_vocab_size(model: object) -> int:
    """Get vocabulary size from model."""
    config = getattr(model, "config", None)
    if config is not None:
        vocab_size = getattr(config, "vocab_size", None)
        if isinstance(vocab_size, int) and vocab_size > 0:
            return vocab_size
    raise ValueError("Cannot determine model vocab_size")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/token_channel/test_teacher_batch.py::test_extract_single_text_all_positions_shape -v`

Expected: PASS

- [ ] **Step 5: Commit intra-text parallel inference**

```bash
git add wfcllm/watermark/token_channel/teacher.py tests/watermark/token_channel/test_teacher_batch.py
git commit -m "feat: add intra-text parallel inference for teacher model

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 2: Inter-Text Parallel Inference (Batch Multiple Texts)

**Files:**
- Modify: `wfcllm/watermark/token_channel/teacher.py`
- Test: `tests/watermark/token_channel/test_teacher_batch.py`

- [ ] **Step 1: Write failing test for batch forward**

Add to `tests/watermark/token_channel/test_teacher_batch.py`:

```python
def test_batch_forward_all_positions_shape(mock_tokenizer, mock_model):
    """Test batch forward returns correct shape."""
    batch_token_ids = [
        [1, 2, 3],
        [4, 5, 6, 7],
        [8, 9],
    ]

    from wfcllm.watermark.token_channel.teacher import _batch_forward_all_positions

    logits = _batch_forward_all_positions(
        model=mock_model,
        batch_token_ids=batch_token_ids,
        tokenizer=mock_tokenizer,
    )

    # Should return list of [seq_len, vocab_size] tensors
    assert len(logits) == 3
    assert logits[0].shape == (3, mock_model.config.vocab_size)
    assert logits[1].shape == (4, mock_model.config.vocab_size)
    assert logits[2].shape == (2, mock_model.config.vocab_size)


def test_batch_forward_with_padding(mock_tokenizer, mock_model):
    """Test that padding doesn't affect results."""
    token_ids = [1, 2, 3]

    from wfcllm.watermark.token_channel.teacher import (
        _extract_single_text_all_positions,
        _batch_forward_all_positions,
    )

    # Single inference
    single_logits = _extract_single_text_all_positions(
        model=mock_model,
        token_ids=token_ids,
        tokenizer=mock_tokenizer,
    )

    # Batch inference with padding
    batch_logits = _batch_forward_all_positions(
        model=mock_model,
        batch_token_ids=[token_ids, [4, 5, 6, 7, 8]],  # Different lengths
        tokenizer=mock_tokenizer,
    )

    # Results should be close (within float16 precision)
    assert torch.allclose(single_logits, batch_logits[0], atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/token_channel/test_teacher_batch.py::test_batch_forward_all_positions_shape -v`

Expected: FAIL with "_batch_forward_all_positions not found"

- [ ] **Step 3: Implement batch forward with padding**

Add to `wfcllm/watermark/token_channel/teacher.py`:

```python
def _batch_forward_all_positions(
    model: object,
    batch_token_ids: list[list[int]],
    tokenizer: object,
) -> list[torch.Tensor]:
    """Batch forward pass for multiple texts (inter-text parallel).

    Args:
        model: Causal language model
        batch_token_ids: List of token ID lists (variable length)
        tokenizer: Tokenizer (for padding and bos_token_id)

    Returns:
        List of tensors, each [seq_len, vocab_size] for corresponding input
    """
    if not batch_token_ids:
        return []

    # Get padding token
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", 0)

    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    use_bos = bos_token_id is not None and isinstance(bos_token_id, int)

    # Prepare inputs with optional bos prepending
    original_lengths = [len(ids) for ids in batch_token_ids]
    if use_bos:
        prepared_inputs = [[bos_token_id] + ids for ids in batch_token_ids]
    else:
        prepared_inputs = [list(ids) for ids in batch_token_ids]

    # Left padding (preserves causal structure)
    max_len = max(len(ids) for ids in prepared_inputs)
    input_ids = []
    attention_mask = []

    for ids in prepared_inputs:
        padding_length = max_len - len(ids)
        input_ids.append([pad_token_id] * padding_length + ids)
        attention_mask.append([0] * padding_length + [1] * len(ids))

    # Convert to tensors
    input_ids_tensor = torch.tensor(
        input_ids,
        dtype=torch.long,
        device=_resolve_module_device(model),
    )
    attention_mask_tensor = torch.tensor(
        attention_mask,
        dtype=torch.long,
        device=_resolve_module_device(model),
    )

    # Forward pass
    was_training = getattr(model, "training", None)
    eval_method = getattr(model, "eval", None)
    train_method = getattr(model, "train", None)

    if callable(eval_method):
        eval_method()

    try:
        with torch.no_grad():
            output = model(input_ids_tensor, attention_mask=attention_mask_tensor)
    finally:
        if was_training is not None and callable(train_method):
            train_method(was_training)

    # Extract logits
    logits = getattr(output, "logits", None)
    if logits is None and isinstance(output, dict):
        logits = output.get("logits")
    if logits is None:
        logits = output if isinstance(output, torch.Tensor) else None
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise ValueError("model must return 3D logits [batch, seq, vocab]")

    # Extract per-sample logits (remove padding and bos offset)
    results = []
    for i, orig_len in enumerate(original_lengths):
        # Find where actual tokens start (after left padding)
        padding_length = max_len - len(prepared_inputs[i])

        if use_bos:
            # logits[i, padding_length] predicts first real token
            # logits[i, padding_length + orig_len - 1] predicts last real token
            sample_logits = logits[i, padding_length : padding_length + orig_len]
        else:
            # Without bos, we need to skip the last prediction
            sample_logits = logits[i, padding_length : padding_length + orig_len]

        results.append(sample_logits.detach().cpu().to(dtype=torch.float32))

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/token_channel/test_teacher_batch.py::test_batch_forward_all_positions_shape tests/watermark/token_channel/test_teacher_batch.py::test_batch_forward_with_padding -v`

Expected: PASS

- [ ] **Step 5: Commit inter-text parallel inference**

```bash
git add wfcllm/watermark/token_channel/teacher.py tests/watermark/token_channel/test_teacher_batch.py
git commit -m "feat: add inter-text parallel batch inference with padding

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Dynamic Batch Sizing and Length Grouping

**Files:**
- Modify: `wfcllm/watermark/token_channel/teacher.py`
- Test: `tests/watermark/token_channel/test_teacher_batch.py`

- [ ] **Step 1: Write failing test for length grouping**

Add to `tests/watermark/token_channel/test_teacher_batch.py`:

```python
def test_group_texts_by_length(mock_tokenizer):
    """Test that texts are grouped by similar length."""
    texts = [
        "short",
        "a bit longer text",
        "tiny",
        "another longer text here",
        "mid",
    ]

    from wfcllm.watermark.token_channel.teacher import _group_texts_by_length

    groups = _group_texts_by_length(
        texts=texts,
        tokenizer=mock_tokenizer,
        tolerance=0.3,
    )

    # Should have multiple groups
    assert len(groups) >= 2

    # Each group should have texts with similar lengths
    for group in groups:
        lengths = [len(text) for idx, text in group]
        if len(lengths) > 1:
            max_len = max(lengths)
            min_len = min(lengths)
            assert (max_len - min_len) / max_len <= 0.3


def test_compute_dynamic_batch_size():
    """Test dynamic batch size computation."""
    from wfcllm.watermark.token_channel.teacher import _compute_dynamic_batch_size

    # Short sequences should allow larger batches
    batch_size_short = _compute_dynamic_batch_size(
        max_length=50,
        base_batch_size=16,
        available_memory_gb=20.0,
    )
    assert batch_size_short >= 16

    # Long sequences should use smaller batches
    batch_size_long = _compute_dynamic_batch_size(
        max_length=500,
        base_batch_size=16,
        available_memory_gb=20.0,
    )
    assert batch_size_long <= 16
    assert batch_size_short > batch_size_long
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/token_channel/test_teacher_batch.py::test_group_texts_by_length -v`

Expected: FAIL with "_group_texts_by_length not found"

- [ ] **Step 3: Implement length grouping and dynamic batch sizing**

Add to `wfcllm/watermark/token_channel/teacher.py`:

```python
def _group_texts_by_length(
    texts: list[str],
    tokenizer: object,
    tolerance: float = 0.2,
) -> list[list[tuple[int, str]]]:
    """Group texts by similar length to reduce padding waste.

    Args:
        texts: List of text strings
        tokenizer: Tokenizer for encoding
        tolerance: Max relative length difference within a group (0.2 = 20%)

    Returns:
        List of groups, each group is list of (index, text) tuples
    """
    if not texts:
        return []

    # Encode and get lengths
    indexed_texts = []
    for i, text in enumerate(texts):
        token_ids = list(_encode_text(tokenizer, text))
        indexed_texts.append((i, text, len(token_ids)))

    # Sort by length
    indexed_texts.sort(key=lambda x: x[2])

    # Group by tolerance
    groups = []
    current_group = []
    current_min_len = None

    for idx, text, length in indexed_texts:
        if current_min_len is None:
            current_min_len = length
            current_group.append((idx, text))
        else:
            # Check if this text fits in current group
            if length <= current_min_len * (1 + tolerance):
                current_group.append((idx, text))
            else:
                # Start new group
                groups.append(current_group)
                current_group = [(idx, text)]
                current_min_len = length

    if current_group:
        groups.append(current_group)

    return groups


def _compute_dynamic_batch_size(
    max_length: int,
    base_batch_size: int,
    available_memory_gb: float,
) -> int:
    """Compute dynamic batch size based on sequence length and available memory.

    Args:
        max_length: Maximum sequence length in the batch
        base_batch_size: Base batch size (for reference)
        available_memory_gb: Available GPU memory in GB

    Returns:
        Adjusted batch size
    """
    # Empirical formula: memory_per_sample ≈ seq_len * 0.5 MB
    # This is a rough estimate and should be tuned based on actual model
    memory_per_token_mb = 0.5
    memory_per_sample_gb = (max_length * memory_per_token_mb) / 1024

    if memory_per_sample_gb <= 0:
        return base_batch_size

    # Calculate max batch size that fits in memory
    max_batch_size = int(available_memory_gb / memory_per_sample_gb)

    # Apply constraints
    # - Minimum: 4 (too small is inefficient)
    # - Maximum: base_batch_size * 2 (don't go too large)
    adjusted = max(4, min(max_batch_size, base_batch_size * 2))

    return adjusted


def _get_available_memory_gb(model: object) -> float:
    """Get available GPU memory in GB.

    Args:
        model: Model to check device

    Returns:
        Available memory in GB, or 20.0 if cannot determine
    """
    device = _resolve_module_device(model)
    if device.type == "cuda":
        try:
            import torch.cuda
            # Get free memory on the device
            free_memory = torch.cuda.mem_get_info(device.index or 0)[0]
            return free_memory / (1024 ** 3)  # Convert to GB
        except Exception:
            pass
    # Default conservative estimate
    return 20.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/token_channel/test_teacher_batch.py::test_group_texts_by_length tests/watermark/token_channel/test_teacher_batch.py::test_compute_dynamic_batch_size -v`

Expected: PASS

- [ ] **Step 5: Commit dynamic optimization helpers**

```bash
git add wfcllm/watermark/token_channel/teacher.py tests/watermark/token_channel/test_teacher_batch.py
git commit -m "feat: add dynamic batch sizing and length grouping

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 4: Main Batch Extract Function

**Files:**
- Modify: `wfcllm/watermark/token_channel/teacher.py`
- Test: `tests/watermark/token_channel/test_teacher_batch.py`

- [ ] **Step 1: Write failing test for batch_extract_teacher_rows**

Add to `tests/watermark/token_channel/test_teacher_batch.py`:

```python
def test_batch_extract_teacher_rows_basic(mock_tokenizer, mock_model):
    """Test basic batch extraction."""
    texts = [
        "def foo():\n    return 42",
        "x = 1 + 2",
    ]

    from wfcllm.watermark.token_channel.teacher import batch_extract_teacher_rows

    results = batch_extract_teacher_rows(
        tokenizer=mock_tokenizer,
        model=mock_model,
        texts=texts,
        context_width=64,
        batch_size=2,
        show_progress=False,
    )

    # Should return list of lists (one per text)
    assert len(results) == 2
    assert isinstance(results[0], list)
    assert isinstance(results[1], list)

    # Each row should have required fields
    for text_rows in results:
        for row in text_rows:
            assert "prefix_tokens" in row
            assert "next_token" in row
            assert "teacher_logits" in row
            assert "entropy" in row
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/token_channel/test_teacher_batch.py::test_batch_extract_teacher_rows_basic -v`

Expected: FAIL with "batch_extract_teacher_rows not found"

- [ ] **Step 3: Implement batch_extract_teacher_rows**

Add to `wfcllm/watermark/token_channel/teacher.py`:

```python
def batch_extract_teacher_rows(
    tokenizer: object,
    model: object,
    texts: list[str],
    context_width: int,
    batch_size: int = 16,
    *,
    show_progress: bool = False,
) -> list[list[dict[str, object]]]:
    """Extract teacher rows for multiple texts using batch inference.

    Two-level parallelism:
    1. Intra-text: Single forward pass per text for all token positions
    2. Inter-text: Batch multiple texts together

    Args:
        tokenizer: Tokenizer
        model: Causal language model
        texts: List of text strings
        context_width: Maximum context window
        batch_size: Base batch size (dynamically adjusted)
        show_progress: Show progress bar

    Returns:
        List of row lists, one per input text
    """
    if context_width <= 0:
        raise ValueError("context_width must be > 0")

    if not texts:
        return []

    # Group texts by length to reduce padding waste
    groups = _group_texts_by_length(texts, tokenizer, tolerance=0.2)

    # Prepare result storage (preserve original order)
    results = [None] * len(texts)

    # Progress tracking
    total_texts = len(texts)
    text_iterator = (
        tqdm(total=total_texts, desc="批量提取 teacher logits", leave=False, dynamic_ncols=True)
        if show_progress and tqdm is not None
        else None
    )

    # Process each group
    for group in groups:
        if not group:
            continue

        # Get max length in this group
        max_length = max(len(_encode_text(tokenizer, text)) for _, text in group)

        # Compute dynamic batch size
        available_memory = _get_available_memory_gb(model)
        effective_batch_size = _compute_dynamic_batch_size(
            max_length=max_length,
            base_batch_size=batch_size,
            available_memory_gb=available_memory,
        )

        # Process group in batches
        for batch_start in range(0, len(group), effective_batch_size):
            batch_group = group[batch_start : batch_start + effective_batch_size]

            # Extract indices and texts
            batch_indices = [idx for idx, _ in batch_group]
            batch_texts = [text for _, text in batch_group]

            # Encode all texts
            batch_token_ids = [list(_encode_text(tokenizer, text)) for text in batch_texts]

            # Batch forward pass
            batch_logits = _batch_forward_all_positions(
                model=model,
                batch_token_ids=batch_token_ids,
                tokenizer=tokenizer,
            )

            # Convert logits to rows for each text
            for idx, text, token_ids, logits in zip(
                batch_indices, batch_texts, batch_token_ids, batch_logits
            ):
                token_spans = _align_token_spans(text, token_ids, tokenizer)
                rows = []

                for position in range(len(token_ids)):
                    # Compute effective prefix (respecting context_width)
                    prefix_start = max(0, position - context_width)
                    prefix_tokens = token_ids[prefix_start:position]

                    # Extract logits for this position
                    position_logits = logits[position]

                    # Get token span
                    token_start, token_end = token_spans[position]

                    rows.append({
                        "prefix_tokens": list(prefix_tokens),
                        "next_token": token_ids[position],
                        "teacher_logits": position_logits.tolist(),
                        "entropy": _compute_entropy(position_logits),
                        "token_text": text[token_start:token_end],
                        "token_start": token_start,
                        "token_end": token_end,
                        "token_index": position,
                    })

                results[idx] = rows

                if text_iterator is not None:
                    text_iterator.update(1)

    if text_iterator is not None:
        text_iterator.close()

    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/token_channel/test_teacher_batch.py::test_batch_extract_teacher_rows_basic -v`

Expected: PASS

- [ ] **Step 5: Commit main batch extract function**

```bash
git add wfcllm/watermark/token_channel/teacher.py tests/watermark/token_channel/test_teacher_batch.py
git commit -m "feat: implement batch_extract_teacher_rows with two-level parallelism

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 5: Consistency Validation Test

**Files:**
- Test: `tests/watermark/token_channel/test_teacher_batch.py`

- [ ] **Step 1: Write consistency test comparing batch vs sequential**

Add to `tests/watermark/token_channel/test_teacher_batch.py`:

```python
def test_batch_vs_sequential_consistency(mock_tokenizer, mock_model):
    """Verify batch inference produces same results as sequential."""
    text = "def foo():\n    return 42"

    from wfcllm.watermark.token_channel.teacher import (
        extract_teacher_rows,
        batch_extract_teacher_rows,
    )

    # Sequential inference (original)
    rows_sequential = extract_teacher_rows(
        tokenizer=mock_tokenizer,
        model=mock_model,
        text=text,
        context_width=64,
        show_progress=False,
    )

    # Batch inference (new)
    rows_batch_list = batch_extract_teacher_rows(
        tokenizer=mock_tokenizer,
        model=mock_model,
        texts=[text],
        context_width=64,
        batch_size=1,
        show_progress=False,
    )
    rows_batch = rows_batch_list[0]

    # Verify same number of rows
    assert len(rows_sequential) == len(rows_batch)

    # Verify each row matches
    for r_seq, r_batch in zip(rows_sequential, rows_batch):
        assert r_seq["prefix_tokens"] == r_batch["prefix_tokens"]
        assert r_seq["next_token"] == r_batch["next_token"]
        assert r_seq["token_text"] == r_batch["token_text"]
        assert r_seq["token_start"] == r_batch["token_start"]
        assert r_seq["token_end"] == r_batch["token_end"]
        assert r_seq["token_index"] == r_batch["token_index"]

        # Logits should be close (float16 precision)
        logits_seq = torch.tensor(r_seq["teacher_logits"])
        logits_batch = torch.tensor(r_batch["teacher_logits"])
        assert torch.allclose(logits_seq, logits_batch, atol=1e-4)

        # Entropy should be close
        assert abs(r_seq["entropy"] - r_batch["entropy"]) < 0.01
```

- [ ] **Step 2: Run consistency test**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/token_channel/test_teacher_batch.py::test_batch_vs_sequential_consistency -v`

Expected: PASS

- [ ] **Step 3: Commit consistency test**

```bash
git add tests/watermark/token_channel/test_teacher_batch.py
git commit -m "test: add batch vs sequential consistency validation

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 6: Integrate Batch Inference into Training Corpus

**Files:**
- Modify: `wfcllm/watermark/token_channel/train_corpus.py`
- Test: `tests/watermark/token_channel/test_train_corpus.py`

- [ ] **Step 1: Write failing test for batch corpus building**

Add to `tests/watermark/token_channel/test_train_corpus.py`:

```python
def test_build_training_rows_uses_batch_inference(
    mock_tokenizer,
    mock_teacher_model,
    monkeypatch,
):
    """Verify build_training_rows uses batch inference."""
    samples = [
        {"source_code": "def foo():\n    return 1"},
        {"source_code": "x = 2 + 3"},
    ]

    from wfcllm.watermark.token_channel import train_corpus

    # Track calls to batch_extract_teacher_rows
    batch_calls = []

    def mock_batch_extract(tokenizer, model, texts, context_width, batch_size, show_progress):
        batch_calls.append(len(texts))
        # Return mock results
        return [[{"prefix_tokens": [], "next_token": 1, "teacher_logits": [0.0] * 100,
                  "entropy": 1.0, "token_text": "x", "token_start": 0, "token_end": 1,
                  "token_index": 0}] for _ in texts]

    monkeypatch.setattr(
        "wfcllm.watermark.token_channel.teacher.batch_extract_teacher_rows",
        mock_batch_extract,
    )

    # Build training rows
    rows = train_corpus.build_training_rows(
        samples=samples,
        tokenizer=mock_tokenizer,
        teacher_model=mock_teacher_model,
        context_width=64,
        entropy_threshold=1.0,
        diversity_threshold=2,
        teacher_batch_size=8,
    )

    # Should have called batch inference (not per-variant sequential)
    assert len(batch_calls) > 0
    assert sum(batch_calls) >= len(samples)  # At least one call per sample's variants
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/token_channel/test_train_corpus.py::test_build_training_rows_uses_batch_inference -v`

Expected: FAIL (build_training_rows doesn't accept teacher_batch_size yet)

- [ ] **Step 3: Modify build_training_rows to use batch inference**

Modify `wfcllm/watermark/token_channel/train_corpus.py`:

```python
def build_training_rows(
    samples: list[dict[str, object]],
    tokenizer: object,
    teacher_model: object,
    context_width: int,
    *,
    transform_engine: TransformEngine | object | None = None,
    entropy_threshold: float,
    diversity_threshold: int,
    teacher_batch_size: int = 16,  # NEW PARAMETER
) -> list[dict[str, object]]:
    """Build offline supervised rows from base samples and positive variants."""

    if context_width <= 0:
        raise ValueError("context_width must be > 0")
    if diversity_threshold <= 0:
        raise ValueError("diversity_threshold must be > 0")

    rows: list[dict[str, object]] = []

    sample_iterator = (
        tqdm(samples, desc="      处理样本", unit="sample", dynamic_ncols=True)
        if tqdm is not None
        else samples
    )

    for sample in sample_iterator:
        source_code = sample.get("source_code")
        if not isinstance(source_code, str) or not source_code:
            raise ValueError("sample source_code must be a non-empty string")

        sample_rows: list[dict[str, object]] = []
        continuation_sets: dict[tuple[int, ...], set[int]] = defaultdict(set)

        # Generate all variants for this sample
        variants = build_augmented_variants(
            source_code,
            transform_engine=transform_engine,
        )

        # Prepare valid variants (filter out syntax errors)
        valid_variants = []
        valid_feature_contexts = []

        for variant_idx, variant_source in enumerate(variants, 1):
            if tqdm is not None and hasattr(sample_iterator, 'set_postfix'):
                sample_iterator.set_postfix({'variant': f'{variant_idx}/{len(variants)}'}, refresh=True)

            try:
                feature_context = prepare_token_channel_feature_context(variant_source)
                valid_variants.append(variant_source)
                valid_feature_contexts.append(feature_context)
            except SyntaxError:
                if tqdm is not None and hasattr(sample_iterator, 'set_postfix'):
                    sample_iterator.set_postfix({'variant': f'{variant_idx}/{len(variants)} (跳过-语法错误)'}, refresh=True)
                continue

        # Batch extract teacher rows for all valid variants
        if valid_variants:
            all_teacher_rows = batch_extract_teacher_rows(
                tokenizer=tokenizer,
                model=teacher_model,
                texts=valid_variants,
                context_width=context_width,
                batch_size=teacher_batch_size,
                show_progress=True,
            )

            # Process teacher rows with features
            for variant_source, feature_context, teacher_rows in zip(
                valid_variants, valid_feature_contexts, all_teacher_rows
            ):
                for teacher_row in teacher_rows:
                    token_start = teacher_row["token_start"]
                    token_end = teacher_row["token_end"]
                    if not isinstance(token_start, int) or not isinstance(token_end, int):
                        raise ValueError("teacher rows must include integer token spans")

                    features = build_token_channel_features_from_context(
                        feature_context,
                        token_start=token_start,
                        token_end=token_end,
                    )

                    row = {
                        "prefix_tokens": list(teacher_row["prefix_tokens"]),
                        "next_token": teacher_row["next_token"],
                        "teacher_logits": list(teacher_row["teacher_logits"]),
                        "entropy": teacher_row["entropy"],
                        "continuation_diversity": 0,
                        "node_type": features.node_type,
                        "parent_node_type": features.parent_node_type,
                        "block_relative_offset": features.block_relative_offset,
                        "in_code_body": features.in_code_body,
                        "structure_mask": features.structure_mask,
                        "language": features.language,
                        "switch_target": 0,
                    }
                    sample_rows.append(row)
                    continuation_sets[tuple(row["prefix_tokens"])].add(int(row["next_token"]))

        # Compute continuation diversity and switch targets
        for row in sample_rows:
            continuation_diversity = len(continuation_sets[tuple(row["prefix_tokens"])])
            row["continuation_diversity"] = continuation_diversity
            row["switch_target"] = int(
                bool(row["in_code_body"])
                and bool(row["structure_mask"])
                and float(row["entropy"]) >= entropy_threshold
                and continuation_diversity >= diversity_threshold
            )

        rows.extend(sample_rows)

    return rows
```

- [ ] **Step 4: Add import for batch_extract_teacher_rows**

Add to top of `wfcllm/watermark/token_channel/train_corpus.py`:

```python
from wfcllm.watermark.token_channel.teacher import batch_extract_teacher_rows
```

- [ ] **Step 5: Run test to verify it passes**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/token_channel/test_train_corpus.py::test_build_training_rows_uses_batch_inference -v`

Expected: PASS

- [ ] **Step 6: Commit corpus integration**

```bash
git add wfcllm/watermark/token_channel/train_corpus.py tests/watermark/token_channel/test_train_corpus.py
git commit -m "feat: integrate batch inference into training corpus builder

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 7: Add Configuration Parameter

**Files:**
- Modify: `wfcllm/watermark/token_channel/train_workflow.py`
- Test: `tests/watermark/token_channel/test_train_workflow.py`

- [ ] **Step 1: Write failing test for config parameter**

Add to `tests/watermark/token_channel/test_train_workflow.py`:

```python
def test_workflow_config_accepts_teacher_batch_size():
    """Test that config accepts teacher_batch_size parameter."""
    from wfcllm.watermark.token_channel.train_workflow import TokenChannelTrainWorkflowConfig
    from pathlib import Path

    config = TokenChannelTrainWorkflowConfig(
        dataset="humaneval",
        dataset_path=Path("data/datasets/humaneval"),
        lm_model_path=Path("data/models/codet5-base"),
        model_path=Path("data/artifacts/test"),
        cache_path=Path("data/cache/test.json"),
        context_width=64,
        hidden_size=128,
        batch_size=8,
        epochs=1,
        lr=0.001,
        entropy_threshold=1.0,
        diversity_threshold=2,
        split_ratio=0.8,
        seed=42,
        teacher_batch_size=16,  # NEW FIELD
    )

    assert config.teacher_batch_size == 16
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/token_channel/test_train_workflow.py::test_workflow_config_accepts_teacher_batch_size -v`

Expected: FAIL with "unexpected keyword argument 'teacher_batch_size'"

- [ ] **Step 3: Add teacher_batch_size to config**

Modify `wfcllm/watermark/token_channel/train_workflow.py`:

```python
@dataclass(frozen=True)
class TokenChannelTrainWorkflowConfig:
    """Validated configuration surface for token-channel training workflows."""

    dataset: SupportedTokenChannelDataset
    dataset_path: Path
    lm_model_path: Path
    model_path: Path
    cache_path: Path
    context_width: int
    hidden_size: int
    batch_size: int
    epochs: int
    lr: float
    entropy_threshold: float
    diversity_threshold: int
    split_ratio: float
    seed: int
    teacher_batch_size: int = 16  # NEW FIELD with default

    def __post_init__(self) -> None:
        # ... existing validation ...

        teacher_batch_size = _coerce_int(self.teacher_batch_size, "teacher_batch_size")
        object.__setattr__(self, "teacher_batch_size", teacher_batch_size)

        # ... existing validation ...

        if self.teacher_batch_size <= 0:
            raise ValueError("teacher_batch_size must be > 0")
```

- [ ] **Step 4: Update workflow to pass teacher_batch_size**

Modify `run_token_channel_train_workflow()` in `wfcllm/watermark/token_channel/train_workflow.py`:

```python
def run_token_channel_train_workflow(
    config: TokenChannelTrainWorkflowConfig,
) -> TokenChannelTrainWorkflowSummary:
    """Run the offline token-channel training workflow end to end."""

    # ... existing code ...

    training_rows = build_training_rows(
        samples=samples,
        tokenizer=tokenizer,
        teacher_model=teacher_model,
        context_width=config.context_width,
        entropy_threshold=config.entropy_threshold,
        diversity_threshold=config.diversity_threshold,
        teacher_batch_size=config.teacher_batch_size,  # NEW PARAMETER
    )

    # ... rest of function ...
```

- [ ] **Step 5: Run test to verify it passes**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/token_channel/test_train_workflow.py::test_workflow_config_accepts_teacher_batch_size -v`

Expected: PASS

- [ ] **Step 6: Commit config changes**

```bash
git add wfcllm/watermark/token_channel/train_workflow.py tests/watermark/token_channel/test_train_workflow.py
git commit -m "feat: add teacher_batch_size config parameter

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 8: End-to-End Integration Test

**Files:**
- Test: `tests/watermark/token_channel/test_train_workflow.py`

- [ ] **Step 1: Write end-to-end integration test**

Add to `tests/watermark/token_channel/test_train_workflow.py`:

```python
@pytest.mark.slow
def test_workflow_with_batch_inference_integration():
    """Integration test: full workflow with batch inference."""
    from wfcllm.watermark.token_channel.train_workflow import (
        TokenChannelTrainWorkflowConfig,
        run_token_channel_train_workflow,
    )
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        config = TokenChannelTrainWorkflowConfig(
            dataset="humaneval",
            dataset_path=Path("data/datasets/humaneval"),
            lm_model_path=Path("data/models/codet5-base"),
            model_path=Path(tmpdir) / "model",
            cache_path=Path(tmpdir) / "cache.json",
            context_width=64,
            hidden_size=128,
            batch_size=4,
            epochs=1,
            lr=0.001,
            entropy_threshold=1.0,
            diversity_threshold=2,
            split_ratio=0.8,
            seed=42,
            teacher_batch_size=8,  # Use batch inference
        )

        # Run workflow (should complete without errors)
        summary = run_token_channel_train_workflow(config)

        # Verify results
        assert summary.training_rows > 0
        assert summary.train_rows > 0
        assert summary.validation_rows > 0
        assert summary.compatibility_ok
        assert len(summary.epochs) == 1
```

- [ ] **Step 2: Run integration test**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/token_channel/test_train_workflow.py::test_workflow_with_batch_inference_integration -v -s`

Expected: PASS (may take a few minutes)

- [ ] **Step 3: Commit integration test**

```bash
git add tests/watermark/token_channel/test_train_workflow.py
git commit -m "test: add end-to-end batch inference integration test

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 9: Performance Benchmarking

**Files:**
- Create: `tests/watermark/token_channel/test_teacher_performance.py`

- [ ] **Step 1: Create performance benchmark test**

Create `tests/watermark/token_channel/test_teacher_performance.py`:

```python
"""Performance benchmarks for teacher inference."""

import time
import pytest


@pytest.mark.benchmark
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires GPU")
def test_batch_inference_speedup(benchmark_tokenizer, benchmark_model):
    """Benchmark batch vs sequential inference speedup."""
    from wfcllm.watermark.token_channel.teacher import (
        extract_teacher_rows,
        batch_extract_teacher_rows,
    )

    # Test data: 10 code samples
    texts = [
        "def foo():\n    return 42",
        "x = 1 + 2 + 3",
        "for i in range(10):\n    print(i)",
        "class Foo:\n    pass",
        "if x > 0:\n    y = 1\nelse:\n    y = 2",
    ] * 2

    # Sequential inference
    start = time.time()
    for text in texts:
        extract_teacher_rows(
            tokenizer=benchmark_tokenizer,
            model=benchmark_model,
            text=text,
            context_width=64,
            show_progress=False,
        )
    sequential_time = time.time() - start

    # Batch inference
    start = time.time()
    batch_extract_teacher_rows(
        tokenizer=benchmark_tokenizer,
        model=benchmark_model,
        texts=texts,
        context_width=64,
        batch_size=8,
        show_progress=False,
    )
    batch_time = time.time() - start

    speedup = sequential_time / batch_time
    print(f"\nSequential: {sequential_time:.2f}s")
    print(f"Batch: {batch_time:.2f}s")
    print(f"Speedup: {speedup:.1f}x")

    # Should be at least 5x faster
    assert speedup >= 5.0


@pytest.fixture
def benchmark_tokenizer():
    """Real tokenizer for benchmarking."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("data/models/codet5-base")


@pytest.fixture
def benchmark_model():
    """Real model for benchmarking."""
    from transformers import AutoModelForCausalLM
    import torch

    model = AutoModelForCausalLM.from_pretrained(
        "data/models/codet5-base",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return model
```

- [ ] **Step 2: Run benchmark (optional, requires GPU)**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/token_channel/test_teacher_performance.py -v -s -m benchmark`

Expected: PASS with speedup report

- [ ] **Step 3: Commit benchmark**

```bash
git add tests/watermark/token_channel/test_teacher_performance.py
git commit -m "test: add performance benchmark for batch inference

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Task 10: Documentation and Cleanup

**Files:**
- Modify: `wfcllm/watermark/token_channel/teacher.py`
- Create: `docs/token-channel-batch-inference.md`

- [ ] **Step 1: Add module docstring**

Add to top of `wfcllm/watermark/token_channel/teacher.py`:

```python
"""Offline teacher-row extraction helpers for token-channel training.

This module provides both sequential and batch inference modes:

- Sequential: extract_teacher_rows() - Original implementation, one token at a time
- Batch: batch_extract_teacher_rows() - Optimized two-level parallel inference

Batch inference provides 50-100x speedup through:
1. Intra-text parallelism: Single forward pass for all token positions
2. Inter-text parallelism: Batch multiple texts with dynamic sizing

Usage:
    # Batch mode (recommended for training)
    results = batch_extract_teacher_rows(
        tokenizer=tokenizer,
        model=model,
        texts=["def foo(): pass", "x = 1 + 2"],
        context_width=64,
        batch_size=16,
    )

    # Sequential mode (for debugging/validation)
    rows = extract_teacher_rows(
        tokenizer=tokenizer,
        model=model,
        text="def foo(): pass",
        context_width=64,
    )
"""
```

- [ ] **Step 2: Create usage documentation**

Create `docs/token-channel-batch-inference.md`:

```markdown
# Token Channel Batch Inference

## Overview

Batch inference optimization for teacher model in token-channel training.

**Performance:**
- GPU utilization: 67% → 90%+
- Memory usage: 42% → 78%+
- Speedup: 50-100x

## Usage

### Training Workflow

```python
from wfcllm.watermark.token_channel.train_workflow import (
    TokenChannelTrainWorkflowConfig,
    run_token_channel_train_workflow,
)

config = TokenChannelTrainWorkflowConfig(
    # ... other params ...
    teacher_batch_size=16,  # Adjust based on GPU memory
)

summary = run_token_channel_train_workflow(config)
```

### Direct API

```python
from wfcllm.watermark.token_channel.teacher import batch_extract_teacher_rows

results = batch_extract_teacher_rows(
    tokenizer=tokenizer,
    model=model,
    texts=["code1", "code2", "code3"],
    context_width=64,
    batch_size=16,
)
```

## Configuration

**teacher_batch_size**: Base batch size (default: 16)
- Dynamically adjusted based on sequence length
- Short sequences (< 128 tokens): up to 64
- Long sequences (> 512 tokens): down to 8

## Tuning

Monitor GPU usage:
```bash
nvidia-smi dmon -s mu
```

Adjust batch size:
- Memory < 70%: increase batch_size
- Memory > 90%: decrease batch_size
- Target: 80-85% memory usage

## Implementation Details

See design doc: `docs/superpowers/specs/2026-05-04-token-channel-batch-inference-design.md`
```

- [ ] **Step 3: Run all tests to verify everything works**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/token_channel/ -v`

Expected: All tests PASS

- [ ] **Step 4: Commit documentation**

```bash
git add wfcllm/watermark/token_channel/teacher.py docs/token-channel-batch-inference.md
git commit -m "docs: add batch inference documentation and module docstring

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Verification Checklist

After completing all tasks, verify:

- [ ] All unit tests pass
- [ ] Integration test passes
- [ ] Batch inference produces same results as sequential (within float16 precision)
- [ ] GPU utilization increased (monitor with `nvidia-smi`)
- [ ] Memory usage increased (monitor with `nvidia-smi`)
- [ ] Training workflow completes successfully
- [ ] Documentation is clear and complete

## Performance Validation

Run a real training workflow and compare:

```bash
# Before optimization (baseline)
time HF_HUB_OFFLINE=1 conda run -n WFCLLM python -m wfcllm.watermark.token_channel.train_workflow

# After optimization
time HF_HUB_OFFLINE=1 conda run -n WFCLLM python -m wfcllm.watermark.token_channel.train_workflow
```

Expected improvements:
- Corpus building time: 50-100x faster
- GPU utilization: 67% → 90%+
- Memory usage: 13.5GB → 25GB+

---

## Notes

- All tests use `HF_HUB_OFFLINE=1` to ensure offline operation
- Mock fixtures are used for fast unit tests
- Integration tests use real models (slower, marked with `@pytest.mark.slow`)
- Benchmark tests require GPU (marked with `@pytest.mark.benchmark`)
- Follow TDD: write test first, verify it fails, implement, verify it passes, commit
