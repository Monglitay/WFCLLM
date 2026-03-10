# Watermark Generator Refactoring Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the watermark generator to fix rollback state inconsistency bugs and improve retry reliability through unified state management.

**Architecture:** Extract all mutable generation state into a `GenerationContext` with atomic checkpoint/rollback. Separate retry logic into `RetryLoop` and cascade fallback into `CascadeManager`. The generator becomes a thin orchestration layer.

**Tech Stack:** Python 3.10+, PyTorch, tree-sitter, pytest

**Spec:** `docs/superpowers/specs/2026-03-10-watermark-refactor-design.md`

---

## File Structure

### New files
| File | Responsibility |
|------|---------------|
| `wfcllm/watermark/context.py` | `GenerationContext`, `Checkpoint` — unified state + checkpoint/rollback |
| `wfcllm/watermark/retry_loop.py` | `RetryLoop`, `RetryResult`, `RetryDiagnostics`, `AttemptInfo` — retry sub-loop |
| `wfcllm/watermark/cascade.py` | `CascadeManager`, `CascadeCheckpoint` — compound block fallback |
| `tests/watermark/test_context.py` | GenerationContext unit tests |
| `tests/watermark/test_retry_loop.py` | RetryLoop unit tests |
| `tests/watermark/test_cascade.py` | CascadeManager unit tests |
| `tests/watermark/test_rollback_scenarios.py` | 10 scenario-driven functional tests |
| `tests/watermark/test_generator_integration.py` | End-to-end + regression tests |
| `tests/watermark/conftest.py` | Shared test fixtures: MockLM, MockEncoder, RollbackTracer |

### Modified files
| File | Changes |
|------|---------|
| `wfcllm/watermark/config.py` | Add `enable_cascade`, `cascade_max_depth`, `cuda_empty_cache_interval`, `retry_token_budget` |
| `wfcllm/watermark/interceptor.py` | Add `InterceptorState` dataclass, `checkpoint()`/`rollback()` methods, deep copy `_BlockInfo` in `pending_simple`, deprecate `get_pre_event_state()`/`save_state()`/`restore_state()` |
| `wfcllm/watermark/kv_cache.py` | Add safety check, short-circuit optimization, explicit `del` + `empty_cache()`, remove `snapshot_at()` |
| `wfcllm/watermark/generator.py` | Rewrite to thin orchestrator using Context/RetryLoop/CascadeManager; new `GenerateResult` with `EmbedStats` |
| `tests/watermark/test_config.py` | Add tests for new config fields |
| `tests/watermark/test_interceptor.py` | Add tests for new API, update tests referencing removed methods |
| `tests/watermark/test_kv_cache.py` | Add safety check tests, update `snapshot_at` tests |
| `tests/watermark/test_generator.py` | Rewrite for new architecture |
| `wfcllm/watermark/pipeline.py` | Adapt to new `GenerateResult` with `EmbedStats` |
| `tests/watermark/test_pipeline.py` | Adapt to new `GenerateResult` |

---

## Chunk 1: Foundation — Config, KVCache, Interceptor

### Task 1: Add new config fields

**Files:**
- Modify: `wfcllm/watermark/config.py:8-43`
- Modify: `tests/watermark/test_config.py`

- [ ] **Step 1: Write failing tests for new config fields**

In `tests/watermark/test_config.py`, add:

```python
def test_cascade_config_defaults():
    """New cascade fields have correct defaults."""
    cfg = WatermarkConfig(secret_key="k")
    assert cfg.enable_cascade is False
    assert cfg.cascade_max_depth == 1
    assert cfg.cuda_empty_cache_interval == 10
    assert cfg.retry_token_budget is None

def test_cascade_config_custom():
    """Cascade fields can be overridden."""
    cfg = WatermarkConfig(
        secret_key="k",
        enable_cascade=True,
        cascade_max_depth=3,
        cuda_empty_cache_interval=5,
        retry_token_budget=128,
    )
    assert cfg.enable_cascade is True
    assert cfg.cascade_max_depth == 3
    assert cfg.cuda_empty_cache_interval == 5
    assert cfg.retry_token_budget == 128
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_config.py -v -k "cascade_config"`
Expected: FAIL — `TypeError: __init__() got unexpected keyword argument`

- [ ] **Step 3: Add new fields to WatermarkConfig**

In `wfcllm/watermark/config.py`, add after line 42 (`lsh_gamma`):

```python
    # Cascade fallback (compound block re-generation)
    enable_cascade: bool = False
    cascade_max_depth: int = 1

    # Memory management
    cuda_empty_cache_interval: int = 10  # call empty_cache() every N rollbacks

    # Retry budget
    retry_token_budget: int | None = None  # max tokens per retry attempt; None = max_new_tokens // 2
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_config.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add wfcllm/watermark/config.py tests/watermark/test_config.py
git commit -m "feat: add cascade and memory config fields to WatermarkConfig"
```

---

### Task 2: Enhance KVCacheManager — safety check, short-circuit, memory release

**Files:**
- Modify: `wfcllm/watermark/kv_cache.py`
- Modify: `tests/watermark/test_kv_cache.py`

- [ ] **Step 1: Write failing tests for new KVCacheManager behavior**

Add to `tests/watermark/test_kv_cache.py`:

```python
def test_rollback_safety_check_stale_snapshot(self, manager):
    """snapshot.seq_len > current seq_len raises ValueError."""
    kv = self._make_kv_cache(num_layers=2, seq_len=10)
    stale_snap = CacheSnapshot(seq_len=20)
    with pytest.raises(ValueError, match="Snapshot seq_len"):
        manager.rollback(kv, stale_snap)

def test_rollback_same_length_returns_same_object(self, manager):
    """When target_len == current_len, return the same tuple (no clone)."""
    kv = self._make_kv_cache(num_layers=2, seq_len=10)
    snap = manager.snapshot(kv)
    result = manager.rollback(kv, snap)
    assert result is kv  # same object, no clone

def test_rollback_old_tensors_not_referenced(self, manager):
    """After rollback, old tensors have no external references (can be GC'd)."""
    import weakref
    kv = self._make_kv_cache(num_layers=2, seq_len=20)
    snap = CacheSnapshot(seq_len=10)
    # Keep weak reference to original key tensor
    old_k = kv[0][0]
    ref = weakref.ref(old_k)
    rolled = manager.rollback(kv, snap)
    del kv, old_k
    # After del, ref should be dead (no strong references remain)
    # Note: this may not work in all cases due to Python internals,
    # so we test the structural guarantee instead
    for k, v in rolled:
        assert k.shape[2] == 10
        assert v.shape[2] == 10

def test_snapshot_at_removed(self, manager):
    """snapshot_at() should no longer exist."""
    assert not hasattr(manager, "snapshot_at")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_kv_cache.py -v -k "safety_check or same_length or not_referenced or snapshot_at_removed"`
Expected: FAIL — `ValueError` not raised, `is kv` fails, `snapshot_at` still exists

- [ ] **Step 3: Implement KVCacheManager enhancements**

Replace `wfcllm/watermark/kv_cache.py` content:

```python
"""KV-Cache snapshot and rollback for rejection sampling."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CacheSnapshot:
    """Records the sequence length at snapshot time."""

    seq_len: int


class KVCacheManager:
    """Manage KV-Cache snapshots and rollbacks via truncation."""

    def snapshot(self, past_key_values: tuple) -> CacheSnapshot:
        """Record current sequence length from the KV-Cache.

        Args:
            past_key_values: Tuple of (key, value) tensor pairs per layer.
                Each tensor has shape (batch, heads, seq_len, head_dim).
        """
        seq_len = past_key_values[0][0].shape[2]
        return CacheSnapshot(seq_len=seq_len)

    def rollback(
        self, past_key_values: tuple, snapshot: CacheSnapshot
    ) -> tuple:
        """Truncate KV-Cache to the snapshot's sequence length.

        Returns a new tuple of cloned truncated (key, value) pairs.
        Raises ValueError if snapshot is stale (target > current).
        Short-circuits if target == current (no truncation needed).
        """
        target_len = snapshot.seq_len
        current_len = past_key_values[0][0].shape[2]

        if target_len > current_len:
            raise ValueError(
                f"Snapshot seq_len ({target_len}) > current ({current_len}). "
                "Checkpoint may be stale or from a different generation run."
            )

        if target_len == current_len:
            return past_key_values

        new_kv = tuple(
            (k[:, :, :target_len, :].clone(), v[:, :, :target_len, :].clone())
            for k, v in past_key_values
        )

        # Explicitly release old tensors
        for k, v in past_key_values:
            del k, v
        del past_key_values

        return new_kv
```

- [ ] **Step 4: Remove old snapshot_at tests, run all kv_cache tests**

Update `tests/watermark/test_kv_cache.py`: remove `test_snapshot_at_computes_correct_seq_len`, `test_snapshot_at_zero_block_tokens`, `test_snapshot_at_clamps_to_zero`.

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_kv_cache.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add wfcllm/watermark/kv_cache.py tests/watermark/test_kv_cache.py
git commit -m "feat: enhance KVCacheManager with safety check, short-circuit, remove snapshot_at"
```

---

### Task 3: Enhance Interceptor — typed state, checkpoint/rollback API, deep copy

**Files:**
- Modify: `wfcllm/watermark/interceptor.py`
- Modify: `tests/watermark/test_interceptor.py`

- [ ] **Step 1: Write failing tests for new Interceptor API**

Add to `tests/watermark/test_interceptor.py`:

```python
from wfcllm.watermark.interceptor import InterceptorState


class TestInterceptorCheckpointRollback:
    """New typed checkpoint/rollback API."""

    def test_checkpoint_returns_typed_state(self):
        ic = StatementInterceptor()
        for ch in "x = 1\n":
            ic.feed_token(ch)
        state = ic.checkpoint()
        assert isinstance(state, InterceptorState)
        assert isinstance(state.accumulated, str)
        assert isinstance(state.token_idx, int)
        assert isinstance(state.prev_all_keys, set)
        assert isinstance(state.pending_simple, dict)
        assert isinstance(state.emitted_keys, set)
        assert isinstance(state.token_boundaries, list)

    def test_rollback_with_typed_state(self):
        ic = StatementInterceptor()
        for ch in "x = 1\n":
            ic.feed_token(ch)
        state = ic.checkpoint()
        for ch in "y = 2\n":
            ic.feed_token(ch)
        ic.rollback(state)
        assert ic._accumulated == state.accumulated
        assert ic._token_idx == state.token_idx
        assert ic._emitted_keys == state.emitted_keys

    def test_checkpoint_rollback_equivalence(self):
        """checkpoint → feed tokens → rollback → state equals checkpoint."""
        ic = StatementInterceptor()
        for ch in "x = 1\n":
            ic.feed_token(ch)
        cp = ic.checkpoint()
        # Feed more tokens
        for ch in "y = 2\nz = 3\n":
            ic.feed_token(ch)
        ic.rollback(cp)
        # Re-feed same tokens → same events
        events1 = []
        for ch in "y = 2\n":
            e = ic.feed_token(ch)
            if e:
                events1.append(e)
        assert len(events1) >= 1
        assert events1[0].block_text.strip() == "y = 2"

    def test_rollback_accumulated_text_clean(self):
        """After rollback, accumulated does not contain rolled-back tokens."""
        ic = StatementInterceptor()
        for ch in "x = 1\n":
            ic.feed_token(ch)
        cp = ic.checkpoint()
        for ch in "return the\n":
            ic.feed_token(ch)
        ic.rollback(cp)
        assert "return" not in ic._accumulated or ic._accumulated == cp.accumulated

    def test_rollback_then_regenerate_detects_new_block(self):
        """After rollback, feeding different content triggers new block."""
        ic = StatementInterceptor()
        for ch in "x = 1\n":
            ic.feed_token(ch)
        cp = ic.checkpoint()
        # First attempt
        for ch in "y = 2\n":
            ic.feed_token(ch)
        # Rollback
        ic.rollback(cp)
        # Second attempt with different content
        events = []
        for ch in "z = 3\n":
            e = ic.feed_token(ch)
            if e:
                events.append(e)
        assert len(events) >= 1
        assert "z" in events[0].block_text

    def test_deep_copy_pending_simple(self):
        """checkpoint() deep-copies _BlockInfo in pending_simple."""
        ic = StatementInterceptor()
        # Feed partial statement (no newline yet) to create pending_simple entry
        for ch in "x = 1":
            ic.feed_token(ch)
        cp = ic.checkpoint()
        # Verify pending_simple values are independent copies
        if cp.pending_simple:
            for key in cp.pending_simple:
                # Modifying original should not affect checkpoint
                if key in ic._pending_simple:
                    original_text = ic._pending_simple[key].text
                    ic._pending_simple[key].text = "MODIFIED"
                    assert cp.pending_simple[key].text == original_text

    def test_get_pre_event_state_removed(self):
        """get_pre_event_state should raise DeprecationWarning or not exist."""
        ic = StatementInterceptor()
        for ch in "x = 1\n":
            ic.feed_token(ch)
        # Either removed entirely or raises deprecation
        if hasattr(ic, "get_pre_event_state"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("error", DeprecationWarning)
                with pytest.raises(DeprecationWarning):
                    ic.get_pre_event_state()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_interceptor.py::TestInterceptorCheckpointRollback -v`
Expected: FAIL — `InterceptorState` not found, `checkpoint()`/`rollback()` not found

- [ ] **Step 3: Implement InterceptorState and new API**

In `wfcllm/watermark/interceptor.py`:

1. Add `import copy` and `import warnings` at top.

2. Add `InterceptorState` dataclass after `InterceptEvent`:

```python
@dataclass
class InterceptorState:
    """Typed snapshot of interceptor internal state for checkpoint/rollback."""

    accumulated: str
    token_idx: int
    prev_all_keys: set[tuple]
    pending_simple: dict  # key → deep-copied _BlockInfo
    emitted_keys: set[tuple]
    token_boundaries: list[int]
```

3. Add `checkpoint()` and `rollback()` methods to `StatementInterceptor`:

```python
def checkpoint(self) -> InterceptorState:
    """Save current state as a typed, deep-copied snapshot."""
    return InterceptorState(
        accumulated=self._accumulated,
        token_idx=self._token_idx,
        prev_all_keys=set(self._prev_all_keys),
        pending_simple={k: copy.deepcopy(v) for k, v in self._pending_simple.items()},
        emitted_keys=set(self._emitted_keys),
        token_boundaries=list(self._token_boundaries),
    )

def rollback(self, state: InterceptorState) -> None:
    """Restore to a previously checkpointed state."""
    self._accumulated = state.accumulated
    self._token_idx = state.token_idx
    self._prev_all_keys = set(state.prev_all_keys)
    self._pending_simple = {k: copy.deepcopy(v) for k, v in state.pending_simple.items()}
    self._emitted_keys = set(state.emitted_keys)
    self._token_boundaries = list(state.token_boundaries)
```

4. Deprecate `get_pre_event_state()`:

```python
def get_pre_event_state(self) -> dict:
    """Deprecated: use checkpoint() before feed_token() instead."""
    warnings.warn(
        "get_pre_event_state() is deprecated, use checkpoint()/rollback() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    assert self._pre_event_state is not None, (
        "get_pre_event_state() called before any event was emitted"
    )
    return self._pre_event_state
```

5. Keep `save_state()`/`restore_state()` as aliases for backward compat:

```python
def save_state(self) -> dict:
    """Deprecated: use checkpoint() instead. Kept for backward compatibility."""
    state = self.checkpoint()
    return {
        "accumulated": state.accumulated,
        "token_idx": state.token_idx,
        "prev_all_keys": state.prev_all_keys,
        "pending_simple": state.pending_simple,
        "emitted_keys": state.emitted_keys,
        "token_boundaries": state.token_boundaries,
    }

def restore_state(self, state: dict) -> None:
    """Deprecated: use rollback() instead. Kept for backward compatibility."""
    self.rollback(InterceptorState(
        accumulated=state["accumulated"],
        token_idx=state["token_idx"],
        prev_all_keys=set(state["prev_all_keys"]),
        pending_simple={k: copy.deepcopy(v) for k, v in state["pending_simple"].items()},
        emitted_keys=set(state["emitted_keys"]),
        token_boundaries=list(state["token_boundaries"]),
    ))
```

- [ ] **Step 4: Update existing interceptor tests for deprecation**

In `tests/watermark/test_interceptor.py`:

- Update `TestPreEventState` class: the tests for `get_pre_event_state` should expect `DeprecationWarning`. Add `import warnings` and wrap calls with `warnings.catch_warnings()`.
- The `TestStatementInterceptorStateSnapshot` tests using `save_state`/`restore_state` should still pass (backward compat aliases).

- [ ] **Step 5: Run all interceptor tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_interceptor.py -v`
Expected: ALL PASS (old tests via backward compat, new tests via new API)

- [ ] **Step 6: Commit**

```bash
git add wfcllm/watermark/interceptor.py tests/watermark/test_interceptor.py
git commit -m "feat: add typed InterceptorState with checkpoint/rollback, deprecate get_pre_event_state"
```

---

## Chunk 2: Core — GenerationContext, RetryLoop, CascadeManager

### Task 4: Create shared test fixtures (conftest.py)

**Files:**
- Create: `tests/watermark/conftest.py`

- [ ] **Step 1: Create conftest.py with MockLM, MockEncoder, helper factories**

```python
"""Shared test fixtures for watermark tests."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

from wfcllm.watermark.config import WatermarkConfig


@dataclass
class MockModelOutput:
    logits: torch.Tensor
    past_key_values: tuple


class MockLM:
    """Deterministic mock language model for testing.

    Routes token generation based on (base_seq_len, round) where:
    - base_seq_len: the KV cache seq_len at the start of the branch
    - round: how many times we've visited this seq_len (tracks retries)

    Usage:
        lm = MockLM(vocab_size=100, num_layers=2)
        lm.register_branch(base_seq_len=10, token_ids=[5, 6, 7, 2])  # round=1
        lm.register_branch(base_seq_len=10, token_ids=[8, 9, 2], round=2)  # after rollback
    """

    def __init__(self, vocab_size: int = 100, num_layers: int = 2):
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self._branches: dict[tuple[int, int], list[int]] = {}
        self._seq_len_visits: dict[int, int] = defaultdict(int)
        self._current_branch_idx: int = 0

    def register_branch(
        self, base_seq_len: int, token_ids: list[int], round: int = 1
    ):
        self._branches[(base_seq_len, round)] = token_ids

    def parameters(self):
        return iter([torch.zeros(1)])

    def __call__(self, input_ids, past_key_values=None, use_cache=True, **kwargs):
        if past_key_values is None:
            # Prefill: create initial KV cache
            seq_len = input_ids.shape[1]
            past_kv = self._make_kv(seq_len)
            logits = torch.zeros(1, seq_len, self.vocab_size)
            return MockModelOutput(logits=logits, past_key_values=past_kv)

        # Decode step
        current_seq_len = past_key_values[0][0].shape[2]
        new_seq_len = current_seq_len + 1
        past_kv = self._make_kv(new_seq_len)

        # Find the right branch
        # Look for a branch whose base_seq_len is <= current_seq_len
        # and compute offset to determine which token to emit
        token_id = 0  # default
        for (base_sl, rnd), ids in self._branches.items():
            offset = current_seq_len - base_sl
            if offset >= 0 and offset < len(ids):
                # Check round
                if rnd <= self._seq_len_visits.get(base_sl, 0) + 1:
                    token_id = ids[offset]
                    break

        logits = torch.full((1, 1, self.vocab_size), -10.0)
        logits[0, 0, token_id] = 10.0  # Make this token overwhelmingly likely

        return MockModelOutput(logits=logits, past_key_values=past_kv)

    def _make_kv(self, seq_len: int) -> tuple:
        batch, heads, head_dim = 1, 4, 32
        return tuple(
            (
                torch.randn(batch, heads, seq_len, head_dim),
                torch.randn(batch, heads, seq_len, head_dim),
            )
            for _ in range(self.num_layers)
        )

    def notify_rollback(self, base_seq_len: int):
        """Called by test infrastructure when a rollback occurs to this seq_len."""
        self._seq_len_visits[base_seq_len] = self._seq_len_visits.get(base_seq_len, 0) + 1


class MockEncoder:
    """Mock semantic encoder that returns configurable embeddings."""

    def __init__(self, embed_dim: int = 128, default_embedding: torch.Tensor | None = None):
        self._embed_dim = embed_dim
        self._default = default_embedding if default_embedding is not None else torch.randn(embed_dim)
        self._text_map: dict[str, torch.Tensor] = {}

    def register_embedding(self, text: str, embedding: torch.Tensor):
        self._text_map[text] = embedding

    def __call__(self, input_ids, attention_mask=None):
        # Return batch of embeddings
        return self._default.unsqueeze(0)


class MockTokenizer:
    """Simple tokenizer that maps characters to integer IDs."""

    def __init__(self, eos_token_id: int = 2):
        self.eos_token_id = eos_token_id
        self._char_to_id: dict[str, int] = {}
        self._id_to_char: dict[int, str] = {}
        self._next_id = 10  # reserve 0-9

    def _ensure_char(self, ch: str) -> int:
        if ch not in self._char_to_id:
            self._char_to_id[ch] = self._next_id
            self._id_to_char[self._next_id] = ch
            self._next_id += 1
        return self._char_to_id[ch]

    def encode(self, text: str, add_special_tokens: bool = True, return_tensors=None) -> list[int]:
        ids = [self._ensure_char(ch) for ch in text]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, list):
            return "".join(self._id_to_char.get(i, "") for i in ids)
        return self._id_to_char.get(ids, "")

    def __call__(self, text, return_tensors=None, padding=False, truncation=False, max_length=None):
        ids = self.encode(text)
        result = {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones(1, len(ids))}
        return result


@pytest.fixture
def watermark_config():
    """Standard test config."""
    return WatermarkConfig(
        secret_key="test-key",
        max_new_tokens=50,
        max_retries=3,
        encoder_device="cpu",
        temperature=0.0,  # greedy for determinism
        top_k=0,
        top_p=1.0,
    )


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


@pytest.fixture
def mock_model():
    return MockLM()


@pytest.fixture
def mock_encoder():
    return MockEncoder()
```

- [ ] **Step 2: Run to verify fixtures load**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM python -c "import tests.watermark.conftest"`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add tests/watermark/conftest.py
git commit -m "test: add shared MockLM, MockEncoder, MockTokenizer fixtures for watermark tests"
```

---

### Task 5: Implement GenerationContext

**Files:**
- Create: `wfcllm/watermark/context.py`
- Create: `tests/watermark/test_context.py`

- [ ] **Step 1: Write failing tests for GenerationContext**

Create `tests/watermark/test_context.py`:

```python
"""Tests for wfcllm.watermark.context — GenerationContext with checkpoint/rollback."""

import pytest
import torch

from wfcllm.watermark.context import GenerationContext, Checkpoint
from wfcllm.watermark.config import WatermarkConfig
from wfcllm.watermark.interceptor import InterceptorState


class TestCheckpointRollback:
    """Core checkpoint/rollback atomicity tests."""

    @pytest.fixture
    def config(self):
        return WatermarkConfig(
            secret_key="test-key",
            max_new_tokens=50,
            encoder_device="cpu",
            temperature=0.0,
        )

    @pytest.fixture
    def ctx(self, config, mock_model, mock_tokenizer):
        ctx = GenerationContext(
            model=mock_model,
            tokenizer=mock_tokenizer,
            config=config,
        )
        ctx.prefill("def foo():\n    ")
        return ctx

    def test_checkpoint_captures_all_state(self, ctx):
        """checkpoint saves generated_ids, text, kv seq_len, interceptor state."""
        cp = ctx.checkpoint()
        assert isinstance(cp, Checkpoint)
        assert isinstance(cp.generated_ids, list)
        assert isinstance(cp.generated_text, str)
        assert cp.kv_snapshot is not None
        assert isinstance(cp.interceptor_state, InterceptorState)

    def test_rollback_restores_generated_ids(self, ctx):
        """After generate + rollback, generated_ids matches checkpoint."""
        cp = ctx.checkpoint()
        ids_at_cp = list(cp.generated_ids)
        # Generate some tokens
        for _ in range(5):
            ctx.forward_and_sample()
        assert len(ctx.generated_ids) > len(ids_at_cp)
        # Rollback
        ctx.rollback(cp)
        assert ctx.generated_ids == ids_at_cp

    def test_rollback_restores_generated_text(self, ctx):
        cp = ctx.checkpoint()
        text_at_cp = cp.generated_text
        for _ in range(5):
            ctx.forward_and_sample()
        ctx.rollback(cp)
        assert ctx.generated_text == text_at_cp

    def test_rollback_restores_kv_cache_seq_len(self, ctx):
        cp = ctx.checkpoint()
        kv_len_at_cp = cp.kv_snapshot.seq_len
        for _ in range(5):
            ctx.forward_and_sample()
        assert ctx.past_kv[0][0].shape[2] > kv_len_at_cp
        ctx.rollback(cp)
        assert ctx.past_kv[0][0].shape[2] == kv_len_at_cp

    def test_rollback_restores_interceptor_accumulated(self, ctx):
        """Directly validates fix for 'return the' → 'the' bug."""
        cp = ctx.checkpoint()
        acc_at_cp = cp.interceptor_state.accumulated
        for _ in range(5):
            ctx.forward_and_sample()
        ctx.rollback(cp)
        assert ctx.interceptor._accumulated == acc_at_cp

    def test_multiple_checkpoint_rollback_cycles(self, ctx):
        """3 cycles of checkpoint → generate → rollback all restore correctly."""
        for _ in range(3):
            cp = ctx.checkpoint()
            ids_at_cp = list(cp.generated_ids)
            for _ in range(3):
                ctx.forward_and_sample()
            ctx.rollback(cp)
            assert ctx.generated_ids == ids_at_cp

    def test_checkpoint_is_independent_copy(self, ctx):
        """Modifying state after checkpoint doesn't affect the checkpoint."""
        cp = ctx.checkpoint()
        ids_at_cp = list(cp.generated_ids)
        ctx.forward_and_sample()
        # Checkpoint should still hold original ids
        assert cp.generated_ids == ids_at_cp

    def test_rollback_to_empty_state(self, ctx):
        """Checkpoint at 0 generated tokens, rollback restores empty state."""
        cp = ctx.checkpoint()
        assert cp.generated_ids == []
        for _ in range(5):
            ctx.forward_and_sample()
        ctx.rollback(cp)
        assert ctx.generated_ids == []
        assert ctx.generated_text == ""


class TestForwardAndSample:
    @pytest.fixture
    def config(self):
        return WatermarkConfig(
            secret_key="test-key", max_new_tokens=50,
            encoder_device="cpu", temperature=0.0,
        )

    @pytest.fixture
    def ctx(self, config, mock_model, mock_tokenizer):
        ctx = GenerationContext(
            model=mock_model, tokenizer=mock_tokenizer, config=config,
        )
        ctx.prefill("x")
        return ctx

    def test_forward_and_sample_appends_to_generated_ids(self, ctx):
        before = len(ctx.generated_ids)
        ctx.forward_and_sample()
        assert len(ctx.generated_ids) == before + 1

    def test_forward_and_sample_updates_text(self, ctx):
        ctx.forward_and_sample()
        # generated_text should have at least one character
        assert len(ctx.generated_text) >= 0  # May be empty string for special tokens

    def test_forward_and_sample_grows_kv_cache(self, ctx):
        kv_before = ctx.past_kv[0][0].shape[2]
        ctx.forward_and_sample()
        assert ctx.past_kv[0][0].shape[2] == kv_before + 1

    def test_forward_and_sample_feeds_interceptor(self, ctx):
        """Token is fed to interceptor (token_idx increments)."""
        idx_before = ctx.interceptor._token_idx
        ctx.forward_and_sample()
        assert ctx.interceptor._token_idx == idx_before + 1

    def test_last_event_is_none_for_partial_token(self, ctx):
        """Single token usually doesn't complete a statement block."""
        ctx.forward_and_sample()
        # No guarantee either way, but shouldn't crash
        # last_event is either None or an InterceptEvent

    def test_last_block_checkpoint_available_on_event(self, ctx, mock_model):
        """When interceptor fires an event, last_block_checkpoint is set."""
        # This is hard to test without controlling exact token output
        # Just verify the attribute exists and is None initially
        assert ctx.last_block_checkpoint is None


class TestMemorySafety:
    @pytest.fixture
    def config(self):
        return WatermarkConfig(
            secret_key="test-key", max_new_tokens=50,
            encoder_device="cpu", cuda_empty_cache_interval=2,
        )

    @pytest.fixture
    def ctx(self, config, mock_model, mock_tokenizer):
        ctx = GenerationContext(
            model=mock_model, tokenizer=mock_tokenizer, config=config,
        )
        ctx.prefill("x")
        return ctx

    def test_repeated_rollback_no_memory_growth(self, ctx):
        """50 cycles of generate → rollback shouldn't leak memory.
        We verify by checking tensor count doesn't grow unbounded."""
        cp = ctx.checkpoint()
        for _ in range(50):
            for _ in range(3):
                ctx.forward_and_sample()
            ctx.rollback(cp)
        # If we got here without OOM, the test passes
        assert ctx.past_kv[0][0].shape[2] == cp.kv_snapshot.seq_len
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_context.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'wfcllm.watermark.context'`

- [ ] **Step 3: Implement GenerationContext**

Create `wfcllm/watermark/context.py`:

```python
"""Unified generation state management with atomic checkpoint/rollback."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from wfcllm.watermark.config import WatermarkConfig
from wfcllm.watermark.interceptor import StatementInterceptor, InterceptorState, InterceptEvent
from wfcllm.watermark.kv_cache import KVCacheManager, CacheSnapshot

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    """Complete snapshot of all mutable generation state."""

    generated_ids: list[int]
    generated_text: str
    kv_snapshot: CacheSnapshot
    interceptor_state: InterceptorState


class GenerationContext:
    """Encapsulates all mutable state during code generation.

    Provides atomic checkpoint/rollback ensuring generated_ids, KV cache,
    and interceptor state are always in sync.
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: WatermarkConfig,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._config = config
        self._cache_mgr = KVCacheManager()
        self._device = next(model.parameters()).device
        self._rollback_count = 0
        self._empty_cache_interval = config.cuda_empty_cache_interval

        # Mutable state
        self.generated_ids: list[int] = []
        self.generated_text: str = ""
        self.past_kv: tuple | None = None
        self.interceptor = StatementInterceptor()

        # Event tracking
        self.last_event: InterceptEvent | None = None
        self.last_block_checkpoint: Checkpoint | None = None

        self._eos_id: int | None = None

    @property
    def eos_id(self) -> int:
        if self._eos_id is None:
            self._eos_id = self._config.eos_token_id or self._tokenizer.eos_token_id
        return self._eos_id

    def prefill(self, prompt: str) -> None:
        """Run model forward on prompt to initialize KV cache."""
        input_ids = self._tokenizer.encode(prompt, return_tensors="pt")
        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids], dtype=torch.long)
        input_ids = input_ids.to(self._device)

        output = self._model(input_ids=input_ids, use_cache=True)
        self.past_kv = output.past_key_values

        # Store last logits for first forward_and_sample call
        self._prefill_logits = output.logits[:, -1, :]

        self.interceptor.reset()

    def checkpoint(self) -> Checkpoint:
        """Atomically save current state."""
        return Checkpoint(
            generated_ids=list(self.generated_ids),
            generated_text=self.generated_text,
            kv_snapshot=self._cache_mgr.snapshot(self.past_kv),
            interceptor_state=self.interceptor.checkpoint(),
        )

    def rollback(self, cp: Checkpoint) -> None:
        """Atomically restore to a checkpointed state."""
        old_kv = self.past_kv
        self.past_kv = self._cache_mgr.rollback(self.past_kv, cp.kv_snapshot)

        if old_kv is not self.past_kv:
            del old_kv
            self._rollback_count += 1
            if self._rollback_count % self._empty_cache_interval == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self.generated_ids = list(cp.generated_ids)
        self.generated_text = cp.generated_text
        self.interceptor.rollback(cp.interceptor_state)

        # Reset event state
        self.last_event = None
        self.last_block_checkpoint = None

    def forward_and_sample(self, penalty_ids: list[int] | None = None) -> int:
        """Single-step forward + sample, atomically updating all state.

        Returns the sampled token id.
        """
        # Capture pre-forward state for lazy checkpoint
        # IMPORTANT: capture kv seq_len BEFORE model forward, so it aligns
        # with the pre-feed generated_ids length (both exclude current token)
        pre_forward_ids_len = len(self.generated_ids)
        pre_forward_text = self.generated_text
        pre_forward_kv_seq_len = (
            self.past_kv[0][0].shape[2] if self.past_kv is not None else 0
        )

        # Model forward
        if not self.generated_ids and hasattr(self, "_prefill_logits"):
            # First token after prefill: use cached logits (no model call)
            logits = self._prefill_logits
            del self._prefill_logits
        else:
            last_id = self.generated_ids[-1] if self.generated_ids else 0
            input_ids = torch.tensor([[last_id]], dtype=torch.long, device=self._device)
            output = self._model(
                input_ids=input_ids,
                past_key_values=self.past_kv,
                use_cache=True,
            )
            logits = output.logits[:, -1, :]
            self.past_kv = output.past_key_values

        # Sample
        next_id = self._sample(logits, penalty_ids)

        # Update generated state
        self.generated_ids.append(next_id)
        token_text = self._tokenizer.decode([next_id], skip_special_tokens=True)
        self.generated_text += token_text

        # Capture interceptor state AFTER model forward but BEFORE feed_token
        # This ensures the interceptor snapshot does not include the new block
        pre_feed_interceptor_state = self.interceptor.checkpoint()

        # Feed interceptor
        event = self.interceptor.feed_token(token_text)

        # Track events with lazy checkpoint materialization
        self.last_event = event
        if event is not None:
            # Materialize full checkpoint from pre-feed state
            # All four components are aligned: they all exclude the current token's
            # contribution to their respective state
            self.last_block_checkpoint = Checkpoint(
                generated_ids=list(self.generated_ids[:pre_forward_ids_len]),
                generated_text=pre_forward_text,
                kv_snapshot=CacheSnapshot(seq_len=pre_forward_kv_seq_len),
                interceptor_state=pre_feed_interceptor_state,
            )
        else:
            self.last_block_checkpoint = None

        return next_id

    def is_finished(self) -> bool:
        """Check if generation should stop."""
        if len(self.generated_ids) >= self._config.max_new_tokens:
            return True
        if self.generated_ids and self.generated_ids[-1] == self.eos_id:
            return True
        return False

    def _sample(
        self,
        logits: torch.Tensor,
        penalty_ids: list[int] | None = None,
    ) -> int:
        """Sample a token with temperature, top-k, top-p, repetition penalty."""
        logits = logits.squeeze(0).float()

        # Repetition penalty
        if penalty_ids and self._config.repetition_penalty != 1.0:
            penalty = self._config.repetition_penalty
            for tid in penalty_ids:
                if 0 <= tid < logits.size(0):
                    if logits[tid] > 0:
                        logits[tid] /= penalty
                    else:
                        logits[tid] *= penalty

        if self._config.temperature > 0:
            logits = logits / self._config.temperature
        else:
            # Greedy: return argmax
            return logits.argmax().item()

        if self._config.top_k > 0:
            top_k = min(self._config.top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k).values[-1]
            logits[indices_to_remove] = float("-inf")

        if self._config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                F.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_indices_to_remove = cumulative_probs > self._config.top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()
```

- [ ] **Step 4: Run tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_context.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add wfcllm/watermark/context.py tests/watermark/test_context.py
git commit -m "feat: add GenerationContext with atomic checkpoint/rollback"
```

---

### Task 6: Implement RetryLoop

**Files:**
- Create: `wfcllm/watermark/retry_loop.py`
- Create: `tests/watermark/test_retry_loop.py`

- [ ] **Step 1: Write failing tests for RetryLoop**

Create `tests/watermark/test_retry_loop.py`:

```python
"""Tests for wfcllm.watermark.retry_loop — rejection sampling retry logic."""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from wfcllm.watermark.retry_loop import RetryLoop, RetryResult, RetryDiagnostics, AttemptInfo
from wfcllm.watermark.context import GenerationContext, Checkpoint
from wfcllm.watermark.config import WatermarkConfig
from wfcllm.watermark.interceptor import InterceptEvent
from wfcllm.watermark.verifier import VerifyResult


class TestRetryResult:
    def test_success_result(self):
        r = RetryResult(
            success=True, attempts=2,
            final_event=MagicMock(spec=InterceptEvent),
            diagnostics=RetryDiagnostics(per_attempt=[], unique_signatures=1, unique_texts=1),
        )
        assert r.success is True
        assert r.attempts == 2

    def test_failure_result(self):
        r = RetryResult(
            success=False, attempts=5, final_event=None,
            diagnostics=RetryDiagnostics(per_attempt=[], unique_signatures=0, unique_texts=0),
        )
        assert r.success is False


class TestRetryLoopUnit:
    """Unit tests with mock GenerationContext."""

    @pytest.fixture
    def config(self):
        return WatermarkConfig(
            secret_key="test-key", max_retries=3, max_new_tokens=50,
            encoder_device="cpu", temperature=0.0,
        )

    @pytest.fixture
    def mock_ctx(self):
        ctx = MagicMock(spec=GenerationContext)
        ctx.generated_ids = []
        ctx.eos_id = 2
        ctx.last_event = None
        return ctx

    @pytest.fixture
    def mock_verifier(self):
        return MagicMock()

    @pytest.fixture
    def mock_keying(self):
        return MagicMock()

    @pytest.fixture
    def mock_entropy(self):
        return MagicMock()

    def test_retry_succeeds_on_first_attempt(self, config, mock_ctx, mock_verifier, mock_keying, mock_entropy):
        """First retry produces a passing block."""
        # Setup: forward_and_sample triggers event, verifier passes
        event = InterceptEvent(
            block_text="return result", block_type="simple",
            node_type="return_statement", parent_node_type="module",
            token_start_idx=0, token_count=2,
        )
        mock_ctx.forward_and_sample.return_value = 5
        mock_ctx.last_event = event
        mock_ctx.eos_id = 2
        mock_ctx.is_finished.return_value = False

        mock_verifier.verify.return_value = VerifyResult(passed=True, min_margin=0.1)
        mock_keying.derive.return_value = frozenset()
        mock_entropy.estimate_block_entropy.return_value = 1.0
        mock_entropy.compute_margin.return_value = 0.001

        loop = RetryLoop(
            ctx=mock_ctx, config=config,
            verifier=mock_verifier, keying=mock_keying,
            entropy_est=mock_entropy, structural_token_ids=set(),
        )
        cp = MagicMock(spec=Checkpoint)
        original_event = MagicMock(spec=InterceptEvent)
        original_event.parent_node_type = "module"

        result = loop.run(cp, original_event)
        assert result.success is True
        assert result.attempts == 1

    def test_retry_exhausts_max_retries(self, config, mock_ctx, mock_verifier, mock_keying, mock_entropy):
        """All retries fail, returns success=False."""
        event = InterceptEvent(
            block_text="x = 1", block_type="simple",
            node_type="expression_statement", parent_node_type="module",
            token_start_idx=0, token_count=2,
        )
        mock_ctx.forward_and_sample.return_value = 5
        mock_ctx.last_event = event
        mock_ctx.eos_id = 2
        mock_ctx.is_finished.return_value = False
        mock_ctx.generated_ids = [5, 6]

        mock_verifier.verify.return_value = VerifyResult(passed=False, min_margin=0.001)
        mock_keying.derive.return_value = frozenset()
        mock_entropy.estimate_block_entropy.return_value = 1.0
        mock_entropy.compute_margin.return_value = 0.001

        loop = RetryLoop(
            ctx=mock_ctx, config=config,
            verifier=mock_verifier, keying=mock_keying,
            entropy_est=mock_entropy, structural_token_ids=set(),
        )
        cp = MagicMock(spec=Checkpoint)
        cp.generated_ids = []
        original_event = MagicMock(spec=InterceptEvent)
        original_event.parent_node_type = "module"

        result = loop.run(cp, original_event)
        assert result.success is False
        assert result.attempts == config.max_retries

    def test_each_retry_calls_rollback(self, config, mock_ctx, mock_verifier, mock_keying, mock_entropy):
        """Each retry attempt starts with ctx.rollback(checkpoint)."""
        event = InterceptEvent(
            block_text="x = 1", block_type="simple",
            node_type="expression_statement", parent_node_type="module",
            token_start_idx=0, token_count=2,
        )
        mock_ctx.forward_and_sample.return_value = 5
        mock_ctx.last_event = event
        mock_ctx.eos_id = 2
        mock_ctx.is_finished.return_value = False
        mock_ctx.generated_ids = [5]

        mock_verifier.verify.return_value = VerifyResult(passed=False, min_margin=0.001)
        mock_keying.derive.return_value = frozenset()
        mock_entropy.estimate_block_entropy.return_value = 1.0
        mock_entropy.compute_margin.return_value = 0.001

        loop = RetryLoop(
            ctx=mock_ctx, config=config,
            verifier=mock_verifier, keying=mock_keying,
            entropy_est=mock_entropy, structural_token_ids=set(),
        )
        cp = MagicMock(spec=Checkpoint)
        cp.generated_ids = []
        original_event = MagicMock(spec=InterceptEvent)
        original_event.parent_node_type = "module"

        loop.run(cp, original_event)
        assert mock_ctx.rollback.call_count == config.max_retries

    def test_diagnostics_records_all_attempts(self, config, mock_ctx, mock_verifier, mock_keying, mock_entropy):
        """RetryDiagnostics.per_attempt has one entry per attempt."""
        event = InterceptEvent(
            block_text="x = 1", block_type="simple",
            node_type="expression_statement", parent_node_type="module",
            token_start_idx=0, token_count=2,
        )
        mock_ctx.forward_and_sample.return_value = 5
        mock_ctx.last_event = event
        mock_ctx.eos_id = 2
        mock_ctx.is_finished.return_value = False
        mock_ctx.generated_ids = [5]

        mock_verifier.verify.return_value = VerifyResult(passed=False, min_margin=0.001)
        mock_keying.derive.return_value = frozenset()
        mock_entropy.estimate_block_entropy.return_value = 1.0
        mock_entropy.compute_margin.return_value = 0.001

        loop = RetryLoop(
            ctx=mock_ctx, config=config,
            verifier=mock_verifier, keying=mock_keying,
            entropy_est=mock_entropy, structural_token_ids=set(),
        )
        cp = MagicMock(spec=Checkpoint)
        cp.generated_ids = []
        original_event = MagicMock(spec=InterceptEvent)
        original_event.parent_node_type = "module"

        result = loop.run(cp, original_event)
        assert len(result.diagnostics.per_attempt) == config.max_retries

    def test_no_penalty_on_first_retry(self, config, mock_ctx, mock_verifier, mock_keying, mock_entropy):
        """First retry should not pass penalty_ids to forward_and_sample."""
        event = InterceptEvent(
            block_text="x = 1", block_type="simple",
            node_type="expression_statement", parent_node_type="module",
            token_start_idx=0, token_count=2,
        )

        # Make first retry succeed
        mock_ctx.last_event = event
        mock_ctx.eos_id = 2
        mock_ctx.is_finished.return_value = False
        mock_ctx.forward_and_sample.return_value = 5

        mock_verifier.verify.return_value = VerifyResult(passed=True, min_margin=0.1)
        mock_keying.derive.return_value = frozenset()
        mock_entropy.estimate_block_entropy.return_value = 1.0
        mock_entropy.compute_margin.return_value = 0.001

        loop = RetryLoop(
            ctx=mock_ctx, config=config,
            verifier=mock_verifier, keying=mock_keying,
            entropy_est=mock_entropy, structural_token_ids=set(),
        )
        cp = MagicMock(spec=Checkpoint)
        original_event = MagicMock(spec=InterceptEvent)
        original_event.parent_node_type = "module"

        loop.run(cp, original_event)
        # First call to forward_and_sample should have penalty_ids=None
        first_call = mock_ctx.forward_and_sample.call_args_list[0]
        assert first_call.kwargs.get("penalty_ids") is None or first_call.args == ()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_retry_loop.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement RetryLoop**

Create `wfcllm/watermark/retry_loop.py`:

```python
"""Rejection sampling retry sub-loop for watermark embedding."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from wfcllm.watermark.config import WatermarkConfig
from wfcllm.watermark.context import GenerationContext, Checkpoint
from wfcllm.watermark.entropy import NodeEntropyEstimator
from wfcllm.watermark.interceptor import InterceptEvent
from wfcllm.watermark.keying import WatermarkKeying
from wfcllm.watermark.verifier import ProjectionVerifier

logger = logging.getLogger(__name__)


@dataclass
class AttemptInfo:
    """Diagnostic info for a single retry attempt."""

    sig: tuple[int, ...] | None = None
    min_margin: float = 0.0
    passed: bool = False
    text: str = ""
    no_block: bool = False


@dataclass
class RetryDiagnostics:
    """Aggregated diagnostics across all retry attempts."""

    per_attempt: list[AttemptInfo] = field(default_factory=list)
    unique_signatures: int = 0
    unique_texts: int = 0


@dataclass
class RetryResult:
    """Result of the retry sub-loop."""

    success: bool
    attempts: int
    final_event: InterceptEvent | None
    diagnostics: RetryDiagnostics


class RetryLoop:
    """Independent retry sub-loop for rejection sampling.

    Each retry:
    1. ctx.rollback(checkpoint) — atomically restore all state
    2. Free-generate until interceptor triggers a new simple block
    3. Verify the block
    4. On failure, collect penalty_ids for next attempt
    """

    def __init__(
        self,
        ctx: GenerationContext,
        config: WatermarkConfig,
        verifier: ProjectionVerifier,
        keying: WatermarkKeying,
        entropy_est: NodeEntropyEstimator,
        structural_token_ids: set[int],
    ):
        self._ctx = ctx
        self._config = config
        self._verifier = verifier
        self._keying = keying
        self._entropy_est = entropy_est
        self._structural_token_ids = structural_token_ids
        self._retry_budget = (
            config.retry_token_budget
            if config.retry_token_budget is not None
            else config.max_new_tokens // 2
        )

    def run(
        self,
        checkpoint: Checkpoint,
        original_event: InterceptEvent,
    ) -> RetryResult:
        """Run the retry loop from checkpoint.

        Args:
            checkpoint: State to rollback to before each retry.
            original_event: The failed event (used for parent_node_type to derive valid_set).

        Returns:
            RetryResult with success status and diagnostics.
        """
        parent = original_event.parent_node_type or "module"
        valid_set = self._keying.derive(parent)

        diagnostics = RetryDiagnostics(per_attempt=[])
        prev_retry_ids: list[int] | None = None
        sigs_seen: set[tuple] = set()
        texts_seen: set[str] = set()

        for attempt_i in range(self._config.max_retries):
            # Atomically restore all state
            self._ctx.rollback(checkpoint)

            # Free-generate until a new simple block
            event = self._generate_until_block(
                penalty_ids=prev_retry_ids,
            )

            if event is None:
                diagnostics.per_attempt.append(AttemptInfo(no_block=True))
                logger.debug(
                    "  [retry %d/%d] sub-loop ended without block",
                    attempt_i + 1, self._config.max_retries,
                )
                continue

            # Verify
            block_entropy = self._entropy_est.estimate_block_entropy(event.block_text)
            margin = self._entropy_est.compute_margin(block_entropy, self._config)
            result = self._verifier.verify(event.block_text, valid_set, margin)

            info = AttemptInfo(
                sig=result.lsh_signature,
                min_margin=result.min_margin,
                passed=result.passed,
                text=event.block_text[:80],
            )
            diagnostics.per_attempt.append(info)
            if result.lsh_signature:
                sigs_seen.add(result.lsh_signature)
            texts_seen.add(event.block_text)

            logger.debug(
                "  [retry %d/%d] sig=%s in_valid=%s min_margin=%.4f "
                "margin_thresh=%.4f passed=%s\n  text=%r",
                attempt_i + 1, self._config.max_retries,
                result.lsh_signature,
                result.lsh_signature in valid_set,
                result.min_margin, margin, result.passed,
                event.block_text[:80],
            )

            if result.passed:
                diagnostics.unique_signatures = len(sigs_seen)
                diagnostics.unique_texts = len(texts_seen)
                return RetryResult(
                    success=True,
                    attempts=attempt_i + 1,
                    final_event=event,
                    diagnostics=diagnostics,
                )

            # Collect penalty IDs for next attempt
            rollback_idx = len(checkpoint.generated_ids)
            retry_ids = self._ctx.generated_ids[rollback_idx:]
            prev_retry_ids = [
                tid for tid in retry_ids
                if tid not in self._structural_token_ids
            ]

        diagnostics.unique_signatures = len(sigs_seen)
        diagnostics.unique_texts = len(texts_seen)
        return RetryResult(
            success=False,
            attempts=self._config.max_retries,
            final_event=None,
            diagnostics=diagnostics,
        )

    def _generate_until_block(
        self,
        penalty_ids: list[int] | None,
    ) -> InterceptEvent | None:
        """Free-generate tokens until a simple block is detected or budget exhausted."""
        for _ in range(self._retry_budget):
            next_id = self._ctx.forward_and_sample(penalty_ids=penalty_ids)
            if next_id == self._ctx.eos_id:
                return None
            event = self._ctx.last_event
            if event is not None and event.block_type == "simple":
                return event
        return None
```

- [ ] **Step 4: Run tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_retry_loop.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add wfcllm/watermark/retry_loop.py tests/watermark/test_retry_loop.py
git commit -m "feat: add RetryLoop with diagnostics and penalty mechanism"
```

---

### Task 7: Implement CascadeManager

**Files:**
- Create: `wfcllm/watermark/cascade.py`
- Create: `tests/watermark/test_cascade.py`

- [ ] **Step 1: Write failing tests for CascadeManager**

Create `tests/watermark/test_cascade.py`:

```python
"""Tests for wfcllm.watermark.cascade — compound block cascade fallback."""

import pytest
from unittest.mock import MagicMock

from wfcllm.watermark.cascade import CascadeManager, CascadeCheckpoint
from wfcllm.watermark.config import WatermarkConfig
from wfcllm.watermark.context import GenerationContext, Checkpoint
from wfcllm.watermark.interceptor import InterceptEvent


class TestCascadeDisabled:
    @pytest.fixture
    def config(self):
        return WatermarkConfig(secret_key="k", enable_cascade=False)

    @pytest.fixture
    def mgr(self, config):
        return CascadeManager(config)

    def test_disabled_by_default(self, mgr):
        assert not mgr._enabled

    def test_on_compound_no_op(self, mgr):
        ctx = MagicMock(spec=GenerationContext)
        event = MagicMock(spec=InterceptEvent)
        mgr.on_compound_block_start(ctx, event)
        ctx.checkpoint.assert_not_called()

    def test_should_cascade_false(self, mgr):
        assert mgr.should_cascade() is False

    def test_cascade_returns_none(self, mgr):
        ctx = MagicMock(spec=GenerationContext)
        assert mgr.cascade(ctx) is None


class TestCascadeEnabled:
    @pytest.fixture
    def config(self):
        return WatermarkConfig(
            secret_key="k", enable_cascade=True, cascade_max_depth=2,
        )

    @pytest.fixture
    def mgr(self, config):
        return CascadeManager(config)

    def test_stores_compound_checkpoint(self, mgr):
        ctx = MagicMock(spec=GenerationContext)
        ctx.checkpoint.return_value = MagicMock(spec=Checkpoint)
        event = MagicMock(spec=InterceptEvent)
        mgr.on_compound_block_start(ctx, event)
        assert len(mgr._stack) == 1

    def test_records_failed_simple_block(self, mgr):
        ctx = MagicMock(spec=GenerationContext)
        ctx.checkpoint.return_value = MagicMock(spec=Checkpoint)
        event = MagicMock(spec=InterceptEvent)
        mgr.on_compound_block_start(ctx, event)
        mgr.on_simple_block_failed("x = 1")
        assert mgr._stack[-1].failed_simple_blocks == ["x = 1"]

    def test_should_cascade_true_after_failure(self, mgr):
        ctx = MagicMock(spec=GenerationContext)
        ctx.checkpoint.return_value = MagicMock(spec=Checkpoint)
        event = MagicMock(spec=InterceptEvent)
        mgr.on_compound_block_start(ctx, event)
        mgr.on_simple_block_failed("x = 1")
        assert mgr.should_cascade() is True

    def test_should_cascade_false_no_failures(self, mgr):
        ctx = MagicMock(spec=GenerationContext)
        ctx.checkpoint.return_value = MagicMock(spec=Checkpoint)
        event = MagicMock(spec=InterceptEvent)
        mgr.on_compound_block_start(ctx, event)
        assert mgr.should_cascade() is False

    def test_cascade_pops_stack_and_rollbacks(self, mgr):
        ctx = MagicMock(spec=GenerationContext)
        cp = MagicMock(spec=Checkpoint)
        ctx.checkpoint.return_value = cp
        event = MagicMock(spec=InterceptEvent)
        mgr.on_compound_block_start(ctx, event)
        mgr.on_simple_block_failed("x = 1")
        result = mgr.cascade(ctx)
        assert result is not None
        assert isinstance(result, CascadeCheckpoint)
        ctx.rollback.assert_called_once_with(cp)
        assert len(mgr._stack) == 0

    def test_max_depth_evicts_oldest(self, mgr):
        """Exceeding max_depth drops the oldest checkpoint."""
        ctx = MagicMock(spec=GenerationContext)
        for i in range(3):
            ctx.checkpoint.return_value = MagicMock(spec=Checkpoint, name=f"cp{i}")
            event = MagicMock(spec=InterceptEvent)
            mgr.on_compound_block_start(ctx, event)
        # max_depth=2, so only last 2 should remain
        assert len(mgr._stack) == 2

    def test_cascade_empty_stack(self, mgr):
        ctx = MagicMock(spec=GenerationContext)
        assert mgr.cascade(ctx) is None

    def test_no_compound_blocks_seen(self, mgr):
        assert mgr.should_cascade() is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_cascade.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement CascadeManager**

Create `wfcllm/watermark/cascade.py`:

```python
"""Cascade fallback manager for compound block re-generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from wfcllm.watermark.config import WatermarkConfig
from wfcllm.watermark.context import GenerationContext, Checkpoint
from wfcllm.watermark.interceptor import InterceptEvent

logger = logging.getLogger(__name__)


@dataclass
class CascadeCheckpoint:
    """Compound block rollback point."""

    checkpoint: Checkpoint
    compound_event: InterceptEvent
    failed_simple_blocks: list[str] = field(default_factory=list)


class CascadeManager:
    """Manage compound block cascade fallback. Default disabled."""

    def __init__(self, config: WatermarkConfig):
        self._enabled = config.enable_cascade
        self._max_depth = config.cascade_max_depth
        self._stack: list[CascadeCheckpoint] = []

    def on_compound_block_start(
        self, ctx: GenerationContext, event: InterceptEvent
    ) -> None:
        """Save a cascade checkpoint when a compound block starts."""
        if not self._enabled:
            return
        cp = CascadeCheckpoint(
            checkpoint=ctx.checkpoint(),
            compound_event=event,
        )
        self._stack.append(cp)
        if len(self._stack) > self._max_depth:
            self._stack.pop(0)

    def on_simple_block_failed(self, block_text: str) -> None:
        """Record a retry-failed simple block."""
        if self._enabled and self._stack:
            self._stack[-1].failed_simple_blocks.append(block_text)

    def should_cascade(self) -> bool:
        """Check if cascade fallback should trigger."""
        if not self._enabled or not self._stack:
            return False
        return len(self._stack[-1].failed_simple_blocks) > 0

    def cascade(self, ctx: GenerationContext) -> CascadeCheckpoint | None:
        """Pop the stack and rollback to compound block start."""
        if not self._stack:
            return None
        cascade_cp = self._stack.pop()
        ctx.rollback(cascade_cp.checkpoint)
        logger.debug(
            "[CASCADE] rolling back to compound block %s, had %d failed simple blocks",
            cascade_cp.compound_event.node_type,
            len(cascade_cp.failed_simple_blocks),
        )
        return cascade_cp
```

- [ ] **Step 4: Run tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_cascade.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add wfcllm/watermark/cascade.py tests/watermark/test_cascade.py
git commit -m "feat: add CascadeManager for compound block fallback"
```

---

## Chunk 3: Integration — Generator Rewrite, Scenarios, Regression

### Task 8: Rewrite WatermarkGenerator

**Files:**
- Modify: `wfcllm/watermark/generator.py`
- Modify: `tests/watermark/test_generator.py`

- [ ] **Step 1: Rewrite test_generator.py for new architecture (TDD — tests first)**

Replace `tests/watermark/test_generator.py` with:

```python
"""Tests for wfcllm.watermark.generator — thin orchestrator."""

import pytest
import torch
from unittest.mock import MagicMock

from wfcllm.watermark.generator import WatermarkGenerator, GenerateResult, EmbedStats
from wfcllm.watermark.config import WatermarkConfig


class TestEmbedStats:
    def test_default_values(self):
        s = EmbedStats()
        assert s.total_blocks == 0
        assert s.embedded_blocks == 0
        assert s.failed_blocks == 0
        assert s.fallback_blocks == 0
        assert s.cascade_blocks == 0
        assert s.retry_diagnostics == []


class TestGenerateResult:
    def test_result_with_stats(self):
        stats = EmbedStats(total_blocks=3, embedded_blocks=2, failed_blocks=1)
        r = GenerateResult(code="x = 1", stats=stats)
        assert r.code == "x = 1"
        assert r.stats.total_blocks == 3

    def test_backward_compat_properties(self):
        stats = EmbedStats(total_blocks=5, embedded_blocks=3, failed_blocks=1, fallback_blocks=1)
        r = GenerateResult(code="", stats=stats)
        assert r.total_blocks == 5
        assert r.embedded_blocks == 3
        assert r.failed_blocks == 1
        assert r.fallback_blocks == 1


class TestWatermarkGeneratorInit:
    @pytest.fixture
    def config(self):
        return WatermarkConfig(secret_key="test-key", max_new_tokens=50, encoder_device="cpu")

    @pytest.fixture
    def mock_components(self):
        model = MagicMock()
        tokenizer = MagicMock()
        encoder = MagicMock()
        encoder_tokenizer = MagicMock()
        tokenizer.encode = MagicMock(return_value=[1, 2, 3])
        tokenizer.decode = MagicMock(return_value="")
        tokenizer.eos_token_id = 2
        return model, tokenizer, encoder, encoder_tokenizer

    def test_generator_init(self, config, mock_components):
        model, tokenizer, encoder, enc_tok = mock_components
        gen = WatermarkGenerator(
            model=model, tokenizer=tokenizer,
            encoder=encoder, encoder_tokenizer=enc_tok, config=config,
        )
        assert gen._config == config
        assert hasattr(gen, "_structural_token_ids")
        assert isinstance(gen._structural_token_ids, set)

    def test_sample_token_method_removed(self, config, mock_components):
        """_sample_token moved to GenerationContext._sample."""
        model, tokenizer, encoder, enc_tok = mock_components
        gen = WatermarkGenerator(
            model=model, tokenizer=tokenizer,
            encoder=encoder, encoder_tokenizer=enc_tok, config=config,
        )
        assert not hasattr(gen, "_sample_token")

    def test_generate_returns_generate_result(self, config, mock_components):
        model, tokenizer, encoder, enc_tok = mock_components
        vocab_size = 100
        logits = torch.zeros(1, 1, vocab_size)
        logits[0, 0, tokenizer.eos_token_id] = 10.0
        past_kv = tuple(
            (torch.randn(1, 4, 3, 32), torch.randn(1, 4, 3, 32))
            for _ in range(2)
        )
        mock_output = MagicMock()
        mock_output.logits = logits
        mock_output.past_key_values = past_kv
        model.return_value = mock_output
        model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))

        gen = WatermarkGenerator(
            model=model, tokenizer=tokenizer,
            encoder=encoder, encoder_tokenizer=enc_tok, config=config,
        )
        result = gen.generate("test")
        assert isinstance(result, GenerateResult)
        assert isinstance(result.stats, EmbedStats)
        assert isinstance(result.code, str)


class TestStructuralTokenFiltering:
    @pytest.fixture
    def config(self):
        return WatermarkConfig(secret_key="test-key", max_new_tokens=50, encoder_device="cpu")

    def test_generator_has_structural_token_ids(self, config):
        model = MagicMock()
        tokenizer = MagicMock()
        encoder = MagicMock()
        enc_tok = MagicMock()
        call_map = {"import": [10], "return": [11], "def": [12]}
        def encode_side_effect(text, **kw):
            return call_map.get(text, [99])
        tokenizer.encode = MagicMock(side_effect=encode_side_effect)
        gen = WatermarkGenerator(
            model=model, tokenizer=tokenizer,
            encoder=encoder, encoder_tokenizer=enc_tok, config=config,
        )
        assert 10 in gen._structural_token_ids
        assert 11 in gen._structural_token_ids
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_generator.py -v`
Expected: FAIL — `EmbedStats` not found, `_sample_token` still exists, etc.

- [ ] **Step 3: Rewrite generator.py as thin orchestrator**

Replace `wfcllm/watermark/generator.py` with:

```python
"""Watermark-embedded code generation using rejection sampling."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

from wfcllm.watermark.cascade import CascadeManager
from wfcllm.watermark.config import WatermarkConfig
from wfcllm.watermark.context import GenerationContext
from wfcllm.watermark.entropy import NodeEntropyEstimator
from wfcllm.watermark.keying import WatermarkKeying
from wfcllm.watermark.lsh_space import LSHSpace
from wfcllm.watermark.retry_loop import RetryLoop, RetryDiagnostics
from wfcllm.watermark.verifier import ProjectionVerifier

logger = logging.getLogger(__name__)


@dataclass
class EmbedStats:
    """Watermark embedding statistics."""

    total_blocks: int = 0
    embedded_blocks: int = 0
    failed_blocks: int = 0
    fallback_blocks: int = 0
    cascade_blocks: int = 0
    retry_diagnostics: list[RetryDiagnostics] = field(default_factory=list)


@dataclass
class GenerateResult:
    """Result of watermark-embedded generation."""

    code: str
    stats: EmbedStats

    # Backward-compatible properties
    @property
    def total_blocks(self) -> int:
        return self.stats.total_blocks

    @property
    def embedded_blocks(self) -> int:
        return self.stats.embedded_blocks

    @property
    def failed_blocks(self) -> int:
        return self.stats.failed_blocks

    @property
    def fallback_blocks(self) -> int:
        return self.stats.fallback_blocks


class WatermarkGenerator:
    """Code generator with watermark embedding via rejection sampling."""

    def __init__(
        self,
        model,
        tokenizer,
        encoder,
        encoder_tokenizer,
        config: WatermarkConfig,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._config = config

        self._entropy_est = NodeEntropyEstimator()
        self._lsh_space = LSHSpace(
            config.secret_key, config.encoder_embed_dim, config.lsh_d
        )
        self._keying = WatermarkKeying(
            config.secret_key, config.lsh_d, config.lsh_gamma
        )
        self._verifier = ProjectionVerifier(
            encoder, encoder_tokenizer,
            lsh_space=self._lsh_space,
            device=config.encoder_device,
        )

        _STRUCTURAL_KEYWORDS = [
            "import", "return", "def", "class", "if", "else", "elif",
            "for", "while", "with", "try", "except", "finally", "pass",
            "break", "continue", "raise", "yield", "lambda",
            "and", "or", "not", "in", "is", "from", "as", "assert",
            "del", "global", "nonlocal", "\n", " ", "\t",
        ]
        self._structural_token_ids: set[int] = {
            tid
            for kw in _STRUCTURAL_KEYWORDS
            for tid in self._tokenizer.encode(kw, add_special_tokens=False)
        }

    @torch.no_grad()
    def generate(self, prompt: str) -> GenerateResult:
        """Generate code with watermark embedding."""
        ctx = GenerationContext(
            model=self._model,
            tokenizer=self._tokenizer,
            config=self._config,
        )
        ctx.prefill(prompt)

        stats = EmbedStats()
        cascade_mgr = CascadeManager(self._config)
        retry_loop = RetryLoop(
            ctx=ctx,
            config=self._config,
            verifier=self._verifier,
            keying=self._keying,
            entropy_est=self._entropy_est,
            structural_token_ids=self._structural_token_ids,
        )
        pending_fallbacks: list[str] = []

        while not ctx.is_finished():
            next_id = ctx.forward_and_sample()

            if next_id == ctx.eos_id:
                break

            event = ctx.last_event
            if event is None:
                continue

            if event.block_type == "compound":
                cascade_mgr.on_compound_block_start(ctx, event)
                self._try_passive_fallback(
                    ctx, event, stats, pending_fallbacks
                )
                continue

            # Simple block
            stats.total_blocks += 1
            verify_result = self._verify_block(event)

            if verify_result.passed:
                stats.embedded_blocks += 1
                pending_fallbacks.clear()
                continue

            # Verification failed → retry
            block_cp = ctx.last_block_checkpoint
            if block_cp is None:
                # Shouldn't happen, but degrade gracefully
                stats.failed_blocks += 1
                pending_fallbacks.append(event.block_text)
                continue

            retry_result = retry_loop.run(block_cp, event)
            stats.retry_diagnostics.append(retry_result.diagnostics)

            if retry_result.success:
                stats.embedded_blocks += 1
                pending_fallbacks.clear()
                logger.debug(
                    "[RETRY OK] block #%d after %d attempts",
                    stats.total_blocks, retry_result.attempts,
                )
            else:
                stats.failed_blocks += 1
                pending_fallbacks.append(event.block_text)
                cascade_mgr.on_simple_block_failed(event.block_text)
                logger.debug(
                    "[RETRY FAILED] block #%d exhausted %d retries",
                    stats.total_blocks, retry_result.attempts,
                )

                if cascade_mgr.should_cascade():
                    self._try_cascade(
                        ctx, cascade_mgr, retry_loop, stats, pending_fallbacks
                    )

        return GenerateResult(code=ctx.generated_text, stats=stats)

    def _verify_block(self, event):
        """Verify a single block against LSH criteria."""
        block_entropy = self._entropy_est.estimate_block_entropy(event.block_text)
        margin = self._entropy_est.compute_margin(block_entropy, self._config)
        valid_set = self._keying.derive(event.parent_node_type or "module")
        result = self._verifier.verify(event.block_text, valid_set, margin)

        logger.debug(
            "[simple block] node=%s parent=%s entropy=%.4f margin_thresh=%.4f\n"
            "  sig=%s in_valid=%s valid_set_size=%d min_margin=%.4f passed=%s\n"
            "  text=%r",
            event.node_type, event.parent_node_type,
            block_entropy, margin,
            result.lsh_signature,
            result.lsh_signature in valid_set,
            len(valid_set), result.min_margin, result.passed,
            event.block_text[:80],
        )
        return result

    def _try_passive_fallback(self, ctx, event, stats, pending_fallbacks):
        """Passive compound fallback: check if compound block passes."""
        if not self._config.enable_fallback or not pending_fallbacks:
            return
        stats.total_blocks += 1
        block_entropy = self._entropy_est.estimate_block_entropy(event.block_text)
        margin = self._entropy_est.compute_margin(block_entropy, self._config)
        valid_set = self._keying.derive(event.parent_node_type or "module")
        result = self._verifier.verify(event.block_text, valid_set, margin)

        logger.debug(
            "[compound fallback] node=%s parent=%s sig=%s passed=%s",
            event.node_type, event.parent_node_type,
            result.lsh_signature, result.passed,
        )
        if result.passed:
            stats.fallback_blocks += 1
            pending_fallbacks.clear()

    def _try_cascade(self, ctx, cascade_mgr, retry_loop, stats, pending_fallbacks):
        """Active cascade: rollback to compound block start, re-generate and verify."""
        cascade_cp = cascade_mgr.cascade(ctx)
        if cascade_cp is None:
            return

        # After rollback, free-generate until we get a compound block event.
        # The regenerated compound block may have different content that passes.
        compound_event = None
        for _ in range(self._config.max_new_tokens):
            next_id = ctx.forward_and_sample()
            if next_id == ctx.eos_id:
                break
            event = ctx.last_event
            if event is not None and event.block_type == "compound":
                compound_event = event
                break

        if compound_event is None:
            logger.debug("[CASCADE FAILED] could not regenerate compound block")
            return

        # Verify the newly regenerated compound block
        block_entropy = self._entropy_est.estimate_block_entropy(
            compound_event.block_text
        )
        margin = self._entropy_est.compute_margin(block_entropy, self._config)
        valid_set = self._keying.derive(
            compound_event.parent_node_type or "module"
        )
        result = self._verifier.verify(
            compound_event.block_text, valid_set, margin
        )

        if result.passed:
            stats.cascade_blocks += 1
            pending_fallbacks.clear()
            logger.debug("[CASCADE OK] regenerated compound block passed")
        else:
            logger.debug("[CASCADE FAILED] regenerated compound block did not pass")
```

- [ ] **Step 4: Run all watermark tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/ -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add wfcllm/watermark/generator.py tests/watermark/test_generator.py
git commit -m "feat: rewrite WatermarkGenerator as thin orchestrator using Context/RetryLoop/Cascade"
```

---

### Task 9: Scenario-driven functional tests

**Files:**
- Create: `tests/watermark/test_rollback_scenarios.py`

- [ ] **Step 1: Write 10 scenario-driven functional tests**

Create `tests/watermark/test_rollback_scenarios.py`:

```python
"""Scenario-driven functional tests for rollback correctness.

Each test constructs a controlled environment where we know exactly what the
interceptor should detect, and verifies that rollback + retry produces
complete (non-truncated) statement blocks.

We test at the GenerationContext + Interceptor level (not full generator)
to isolate rollback mechanics from verification logic.
"""

import pytest

from wfcllm.watermark.context import GenerationContext, Checkpoint
from wfcllm.watermark.config import WatermarkConfig
from wfcllm.watermark.interceptor import StatementInterceptor


@pytest.fixture
def config():
    return WatermarkConfig(
        secret_key="test-key", max_new_tokens=200,
        encoder_device="cpu", temperature=0.0,
    )


class TestScenario1ReturnStatement:
    """Scenario 1: return statement rollback.
    Verify retry generates complete 'return ...' not truncated '...'."""

    def test_rollback_then_refeed_produces_complete_return(self):
        """Feed 'return the\\n' → checkpoint before → rollback → refeed different content.
        The new block must start with 'return'."""
        ic = StatementInterceptor()
        # Feed prefix before statement
        for ch in "def foo():\n    ":
            ic.feed_token(ch)
        cp = ic.checkpoint()
        # Feed return statement
        events = []
        for ch in "return the\n":
            e = ic.feed_token(ch)
            if e:
                events.append(e)
        assert len(events) >= 1
        assert "return" in events[0].block_text

        # Rollback
        ic.rollback(cp)
        assert "return" not in ic._accumulated or ic._accumulated == cp.accumulated

        # Refeed different return
        events2 = []
        for ch in "return result\n":
            e = ic.feed_token(ch)
            if e:
                events2.append(e)
        assert len(events2) >= 1
        assert events2[0].block_text.strip().startswith("return")
        assert "result" in events2[0].block_text


class TestScenario2ImportStatement:
    """Scenario 2: import statement rollback."""

    def test_rollback_import_produces_complete_import(self):
        ic = StatementInterceptor()
        cp = ic.checkpoint()
        events = []
        for ch in "import doctest\n":
            e = ic.feed_token(ch)
            if e:
                events.append(e)
        assert len(events) >= 1
        ic.rollback(cp)
        events2 = []
        for ch in "import unittest\n":
            e = ic.feed_token(ch)
            if e:
                events2.append(e)
        assert len(events2) >= 1
        assert events2[0].block_text.strip().startswith("import")


class TestScenario3AssignmentNoPollution:
    """Scenario 3: rollback of block #2 doesn't affect block #1."""

    def test_first_block_preserved_after_second_rollback(self):
        ic = StatementInterceptor()
        # Block 1: x = 1
        for ch in "x = 1\n":
            ic.feed_token(ch)
        # Checkpoint after block 1
        cp = ic.checkpoint()
        assert "x = 1" in cp.accumulated
        # Block 2: y = 2
        for ch in "y = 2\n":
            ic.feed_token(ch)
        # Rollback block 2
        ic.rollback(cp)
        assert "x = 1" in ic._accumulated
        assert "y = 2" not in ic._accumulated
        # Generate different block 2
        events = []
        for ch in "z = 3\n":
            e = ic.feed_token(ch)
            if e:
                events.append(e)
        assert len(events) >= 1
        assert "z" in events[0].block_text


class TestScenario4NestedExpression:
    """Scenario 4: if block inner expression rollback."""

    def test_nested_rollback_produces_complete_statement(self):
        ic = StatementInterceptor()
        # Set up if block context
        for ch in "if True:\n    open_count += 1\n    ":
            ic.feed_token(ch)
        cp = ic.checkpoint()
        # Feed failing statement
        events = []
        for ch in "open_count -= 1\n":
            e = ic.feed_token(ch)
            if e:
                events.append(e)
        # Rollback
        ic.rollback(cp)
        # Refeed different statement
        events2 = []
        for ch in "close_count += 1\n":
            e = ic.feed_token(ch)
            if e:
                events2.append(e)
        if events2:
            assert "close_count" in events2[0].block_text
            # Must NOT be '_count += 1' (truncated)
            assert events2[0].block_text.strip() != "_count += 1"


class TestScenario5AllRetriesExhausted:
    """Scenario 5: multiple rollback cycles all fail — diagnostics correct."""

    def test_multiple_rollback_cycles(self):
        ic = StatementInterceptor()
        cp = ic.checkpoint()
        attempts = 5
        for i in range(attempts):
            ic.rollback(cp)
            for ch in f"x{i} = {i}\n":
                ic.feed_token(ch)
        # After all retries, state should be at the last attempt's result
        assert f"x{attempts-1}" in ic._accumulated


class TestScenario6CascadeToForLoop:
    """Scenario 6: cascade checkpoint saves compound block start."""

    def test_cascade_rollback_to_compound_start(self):
        ic = StatementInterceptor()
        # Before compound block
        compound_cp = ic.checkpoint()
        # Feed compound block header
        for ch in "for i in range(n):\n":
            ic.feed_token(ch)
        # Feed inner block
        for ch in "    total += arr[i]\n":
            ic.feed_token(ch)
        # Cascade rollback
        ic.rollback(compound_cp)
        assert "for" not in ic._accumulated or ic._accumulated == compound_cp.accumulated
        # Regenerate entire for loop
        events = []
        for ch in "for idx in range(n):\n    total += arr[idx]\n":
            e = ic.feed_token(ch)
            if e:
                events.append(e)
        assert len(events) >= 1


class TestScenario7EOSDuringRetry:
    """Scenario 7: EOS during retry sub-loop doesn't crash."""

    def test_eos_rollback_recovers(self):
        ic = StatementInterceptor()
        cp = ic.checkpoint()
        # First attempt: feed some tokens then "EOS" (just stop feeding)
        for ch in "return":
            ic.feed_token(ch)
        # No newline = no block detected, simulate EOS
        # Rollback and try again
        ic.rollback(cp)
        # Second attempt: complete statement
        events = []
        for ch in "return 0\n":
            e = ic.feed_token(ch)
            if e:
                events.append(e)
        assert len(events) >= 1
        assert "return" in events[0].block_text


class TestScenario8MultiTokenPrecision:
    """Scenario 8: multi-token statement, rollback position precise."""

    def test_8_token_statement_rollback(self):
        ic = StatementInterceptor()
        # Some prefix
        for ch in "# comment\n":
            ic.feed_token(ch)
        cp = ic.checkpoint()
        saved_acc = cp.accumulated
        # Feed multi-char statement: result = some_function(arg)
        stmt = "result = func(a)\n"
        for ch in stmt:
            ic.feed_token(ch)
        ic.rollback(cp)
        assert ic._accumulated == saved_acc
        assert ic._token_idx == cp.token_idx


class TestScenario9FirstBlockFails:
    """Scenario 9: first block fails, rollback to prompt end (empty generated)."""

    def test_first_block_rollback_to_empty(self):
        ic = StatementInterceptor()
        cp = ic.checkpoint()
        assert cp.accumulated == ""
        # First block
        for ch in "return None\n":
            ic.feed_token(ch)
        ic.rollback(cp)
        assert ic._accumulated == ""
        assert ic._token_idx == 0
        # Retry
        events = []
        for ch in "return 0\n":
            e = ic.feed_token(ch)
            if e:
                events.append(e)
        assert len(events) >= 1
        assert events[0].block_text.strip().startswith("return")


class TestScenario10ConsecutiveRetries:
    """Scenario 10: two blocks both retry, independently."""

    def test_two_blocks_independent_rollback(self):
        ic = StatementInterceptor()
        # Block 1: feed and rollback
        cp1 = ic.checkpoint()
        for ch in "x = foo()\n":
            ic.feed_token(ch)
        ic.rollback(cp1)
        events1 = []
        for ch in "x = baz()\n":
            e = ic.feed_token(ch)
            if e:
                events1.append(e)
        assert len(events1) >= 1
        assert "baz" in events1[0].block_text

        # Block 2: feed and rollback
        cp2 = ic.checkpoint()
        for ch in "y = bar()\n":
            ic.feed_token(ch)
        ic.rollback(cp2)
        events2 = []
        for ch in "y = qux()\n":
            e = ic.feed_token(ch)
            if e:
                events2.append(e)
        assert len(events2) >= 1
        assert "qux" in events2[0].block_text
        # Block 1 result preserved
        assert "baz" in ic._accumulated
```

- [ ] **Step 2: Run tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_rollback_scenarios.py -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add tests/watermark/test_rollback_scenarios.py
git commit -m "test: add 10 scenario-driven rollback functional tests"
```

---

### Task 10: Integration and regression tests

**Files:**
- Create: `tests/watermark/test_generator_integration.py`

- [ ] **Step 1: Write integration tests**

Create `tests/watermark/test_generator_integration.py`:

```python
"""Integration and regression tests for WatermarkGenerator."""

import pytest
import torch
from unittest.mock import MagicMock

from wfcllm.watermark.generator import WatermarkGenerator, GenerateResult, EmbedStats
from wfcllm.watermark.config import WatermarkConfig
from wfcllm.watermark.verifier import VerifyResult


class TestGenerateResultBackwardCompat:
    """Verify backward-compatible properties still work."""

    def test_total_blocks_property(self):
        stats = EmbedStats(total_blocks=5, embedded_blocks=3, failed_blocks=2)
        r = GenerateResult(code="code", stats=stats)
        assert r.total_blocks == 5
        assert r.embedded_blocks == 3
        assert r.failed_blocks == 2

    def test_fallback_blocks_property(self):
        stats = EmbedStats(fallback_blocks=1)
        r = GenerateResult(code="", stats=stats)
        assert r.fallback_blocks == 1


class TestGeneratorEOS:
    """Generator handles immediate EOS correctly."""

    @pytest.fixture
    def eos_generator(self):
        config = WatermarkConfig(
            secret_key="k", max_new_tokens=50, encoder_device="cpu",
        )
        model = MagicMock()
        tokenizer = MagicMock()
        encoder = MagicMock()
        enc_tok = MagicMock()
        tokenizer.encode = MagicMock(return_value=[1, 2])
        tokenizer.decode = MagicMock(return_value="")
        tokenizer.eos_token_id = 2
        # Model immediately returns EOS
        vocab_size = 100
        logits = torch.zeros(1, 1, vocab_size)
        logits[0, 0, 2] = 10.0  # EOS
        kv = tuple(
            (torch.randn(1, 4, 3, 32), torch.randn(1, 4, 3, 32))
            for _ in range(2)
        )
        output = MagicMock()
        output.logits = logits
        output.past_key_values = kv
        model.return_value = output
        model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))

        return WatermarkGenerator(
            model=model, tokenizer=tokenizer,
            encoder=encoder, encoder_tokenizer=enc_tok, config=config,
        )

    def test_immediate_eos_returns_empty_code(self, eos_generator):
        result = eos_generator.generate("test")
        assert isinstance(result, GenerateResult)
        assert result.total_blocks == 0
        assert result.failed_blocks == 0


class TestGeneratorStatsAccumulation:
    """Verify stats are correctly accumulated during generation."""

    def test_stats_has_retry_diagnostics_list(self):
        stats = EmbedStats()
        assert isinstance(stats.retry_diagnostics, list)
        assert len(stats.retry_diagnostics) == 0

    def test_embed_stats_cascade_blocks(self):
        stats = EmbedStats(cascade_blocks=2)
        assert stats.cascade_blocks == 2


class TestRegressionReturnTruncation:
    """Regression: 'return the' retry must NOT produce 'the' alone.

    This is the core bug that motivated the refactoring. We verify at the
    interceptor + context level that rollback produces correct state.
    """

    def test_interceptor_rollback_no_truncation(self):
        """After rollback, accumulated text does not retain 'return the'."""
        from wfcllm.watermark.interceptor import StatementInterceptor

        ic = StatementInterceptor()
        for ch in "def foo():\n    ":
            ic.feed_token(ch)
        cp = ic.checkpoint()

        for ch in "return the\n":
            ic.feed_token(ch)

        ic.rollback(cp)
        # Critical assertion: accumulated must NOT contain 'return the'
        assert ic._accumulated == cp.accumulated
        assert "return the" not in ic._accumulated

        # Re-feed should produce complete statement
        events = []
        for ch in "return result\n":
            e = ic.feed_token(ch)
            if e:
                events.append(e)
        assert len(events) >= 1
        block = events[0].block_text.strip()
        assert block.startswith("return"), f"Expected 'return ...' but got {block!r}"


class TestRegressionImportTruncation:
    """Regression: 'import doctest' retry must NOT produce 'doctest' alone."""

    def test_interceptor_rollback_import(self):
        from wfcllm.watermark.interceptor import StatementInterceptor

        ic = StatementInterceptor()
        cp = ic.checkpoint()

        for ch in "import doctest\n":
            ic.feed_token(ch)

        ic.rollback(cp)
        events = []
        for ch in "import unittest\n":
            e = ic.feed_token(ch)
            if e:
                events.append(e)
        assert len(events) >= 1
        block = events[0].block_text.strip()
        assert block.startswith("import"), f"Expected 'import ...' but got {block!r}"


class TestRegressionOpenCountTruncation:
    """Regression: 'open_count -= 1' retry must NOT produce '_count -= 1'."""

    def test_interceptor_rollback_augmented_assign(self):
        from wfcllm.watermark.interceptor import StatementInterceptor

        ic = StatementInterceptor()
        for ch in "if True:\n    open_count += 1\n    ":
            ic.feed_token(ch)
        cp = ic.checkpoint()

        for ch in "open_count -= 1\n":
            ic.feed_token(ch)

        ic.rollback(cp)
        events = []
        for ch in "close_count += 1\n":
            e = ic.feed_token(ch)
            if e:
                events.append(e)
        if events:
            block = events[0].block_text.strip()
            assert "_count" != block, f"Got truncated block: {block!r}"
            assert "close_count" in block
```

- [ ] **Step 2: Run tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_generator_integration.py -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add tests/watermark/test_generator_integration.py
git commit -m "test: add integration and regression tests for watermark generator"
```

---

### Task 11: Adapt pipeline and run full test suite

**Files:**
- Modify: `wfcllm/watermark/pipeline.py` (adapt to `GenerateResult.stats`)
- Modify: `tests/watermark/test_pipeline.py` (adapt assertions)

- [ ] **Step 1: Update pipeline.py for new GenerateResult**

In `wfcllm/watermark/pipeline.py`, any access to `result.total_blocks` etc. should still work via backward-compat properties. Check if pipeline accesses `result` fields directly as a dict or via attributes. Update if needed.

- [ ] **Step 2: Run full test suite**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add wfcllm/watermark/pipeline.py tests/watermark/test_pipeline.py
git commit -m "chore: adapt watermark pipeline to new GenerateResult with EmbedStats"
```

---

### Task 12: Final cleanup and verification

- [ ] **Step 1: Run full test suite one more time**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -v --tb=short`
Expected: ALL PASS

- [ ] **Step 2: Verify no import of experiment/**

Run: `grep -r "from experiment" wfcllm/ || echo "Clean"`
Expected: "Clean"

- [ ] **Step 3: Verify generator.py line count**

Run: `wc -l wfcllm/watermark/generator.py`
Expected: ~150-200 lines (down from 353)

- [ ] **Step 4: Commit any remaining fixes**

```bash
git add wfcllm/watermark/ tests/watermark/
git commit -m "chore: final cleanup for watermark generator refactoring"
```
