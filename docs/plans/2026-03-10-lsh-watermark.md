# LSH Watermark Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the projection-based watermark (single direction vector v + target bit t) with an LSH-based watermark that divides the semantic space into 2^d regions and embeds watermarks by steering code generation into a valid region set G.

**Architecture:** A new `LSHSpace` class holds d global hyperplanes derived from the secret key. `WatermarkKeying.derive()` now returns a `frozenset` of valid LSH signatures G (derived from parent_node_type only). `ProjectionVerifier.verify()` checks if the encoded block's LSH signature falls in G and exceeds the margin distance from all hyperplanes.

**Tech Stack:** Python, PyTorch (cosine similarity, random normal vectors), HMAC-SHA256 (key derivation), pytest (TDD)

---

## Pre-work: Create the develop-LSH branch

```bash
git checkout develop
git checkout -b develop-LSH
```

---

### Task 1: Add lsh_d and lsh_gamma to WatermarkConfig

**Files:**
- Modify: `wfcllm/watermark/config.py`
- Modify: `tests/watermark/test_config.py`

**Step 1: Write the failing test**

Open `tests/watermark/test_config.py` and add:

```python
def test_lsh_defaults():
    cfg = WatermarkConfig(secret_key="k")
    assert cfg.lsh_d == 3
    assert cfg.lsh_gamma == 0.5
```

**Step 2: Run test to verify it fails**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_config.py::test_lsh_defaults -v
```
Expected: FAIL with `AttributeError: 'WatermarkConfig' object has no attribute 'lsh_d'`

**Step 3: Write minimal implementation**

In `wfcllm/watermark/config.py`, add two fields inside the `WatermarkConfig` dataclass after the `repetition_penalty` field:

```python
    # LSH parameters
    lsh_d: int = 3
    lsh_gamma: float = 0.5
```

**Step 4: Run test to verify it passes**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_config.py::test_lsh_defaults -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add wfcllm/watermark/config.py tests/watermark/test_config.py
git commit -m "feat: add lsh_d and lsh_gamma to WatermarkConfig"
```

---

### Task 2: Create LSHSpace class

**Files:**
- Create: `wfcllm/watermark/lsh_space.py`
- Create: `tests/watermark/test_lsh_space.py`

**Step 1: Write the failing tests**

Create `tests/watermark/test_lsh_space.py`:

```python
"""Tests for LSHSpace."""
from __future__ import annotations

import torch
import pytest
from wfcllm.watermark.lsh_space import LSHSpace


class TestLSHSpace:
    @pytest.fixture
    def space(self):
        return LSHSpace(secret_key="test-secret", embed_dim=128, d=3)

    def test_hyperplanes_shape(self, space):
        """planes tensor should be (d, embed_dim)."""
        assert space._planes.shape == (3, 128)

    def test_hyperplanes_are_unit_normalized(self, space):
        """Each hyperplane normal vector should be L2-normalized."""
        norms = torch.norm(space._planes, dim=1)
        assert torch.allclose(norms, torch.ones(3), atol=1e-5)

    def test_sign_returns_tuple_of_length_d(self, space):
        u = torch.randn(128)
        sig = space.sign(u)
        assert isinstance(sig, tuple)
        assert len(sig) == 3
        assert all(b in (0, 1) for b in sig)

    def test_sign_deterministic(self, space):
        u = torch.randn(128)
        assert space.sign(u) == space.sign(u)

    def test_sign_opposite_vector_flips_all_bits(self, space):
        u = torch.randn(128)
        sig_u = space.sign(u)
        sig_neg = space.sign(-u)
        assert all(a != b for a, b in zip(sig_u, sig_neg))

    def test_min_margin_returns_float(self, space):
        u = torch.randn(128)
        m = space.min_margin(u)
        assert isinstance(m, float)
        assert 0.0 <= m <= 1.0

    def test_min_margin_is_min_of_abs_cosines(self, space):
        """min_margin should equal the smallest |cos(u, n_i)| across all planes."""
        import torch.nn.functional as F
        u = torch.randn(128)
        u_norm = F.normalize(u.unsqueeze(0), dim=1).squeeze(0)
        expected = min(
            abs(F.cosine_similarity(u_norm.unsqueeze(0), space._planes[i].unsqueeze(0)).item())
            for i in range(3)
        )
        assert abs(space.min_margin(u) - expected) < 1e-5

    def test_same_key_same_planes(self):
        """Same key and d -> identical hyperplanes."""
        s1 = LSHSpace("key", 64, 2)
        s2 = LSHSpace("key", 64, 2)
        assert torch.allclose(s1._planes, s2._planes)

    def test_different_key_different_planes(self):
        s1 = LSHSpace("key-a", 64, 2)
        s2 = LSHSpace("key-b", 64, 2)
        assert not torch.allclose(s1._planes, s2._planes)
```

**Step 2: Run tests to verify they fail**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_lsh_space.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'wfcllm.watermark.lsh_space'`

**Step 3: Write minimal implementation**

Create `wfcllm/watermark/lsh_space.py`:

```python
"""Global LSH hyperplane space for watermark embedding and extraction."""

from __future__ import annotations

import hashlib
import hmac

import torch
import torch.nn.functional as F


class LSHSpace:
    """Manage d global hyperplanes derived from a secret key.

    The hyperplanes statically partition the semantic embedding space into
    2^d regions identified by binary LSH signatures.
    """

    def __init__(self, secret_key: str, embed_dim: int, d: int):
        self._d = d
        self._planes = self._init_planes(secret_key, embed_dim, d)

    @staticmethod
    def _init_planes(secret_key: str, embed_dim: int, d: int) -> torch.Tensor:
        key_bytes = secret_key.encode("utf-8")
        digest = hmac.new(key_bytes, b"lsh", hashlib.sha256).digest()
        seed = int.from_bytes(digest[:8], "big")
        gen = torch.Generator()
        gen.manual_seed(seed)
        planes = torch.randn(d, embed_dim, generator=gen)
        return F.normalize(planes, dim=1)

    def sign(self, u: torch.Tensor) -> tuple[int, ...]:
        """Compute d-bit LSH signature for embedding vector u.

        Args:
            u: Embedding vector of shape (embed_dim,).

        Returns:
            Tuple of d bits in {0, 1}.
        """
        u_norm = F.normalize(u.float().unsqueeze(0), dim=1)
        dots = (self._planes.float() @ u_norm.T).squeeze(1)
        return tuple((dots > 0).int().tolist())

    def min_margin(self, u: torch.Tensor) -> float:
        """Return minimum absolute cosine distance from u to all hyperplanes.

        Used as the margin guard: a large value means u is well inside its
        LSH region, far from any decision boundary.

        Args:
            u: Embedding vector of shape (embed_dim,).

        Returns:
            Minimum |cos(u, n_i)| across all d hyperplanes.
        """
        u_norm = F.normalize(u.float().unsqueeze(0), dim=1)
        dots = (self._planes.float() @ u_norm.T).squeeze(1)
        return dots.abs().min().item()
```

**Step 4: Run tests to verify they pass**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_lsh_space.py -v
```
Expected: All PASS

**Step 5: Commit**

```bash
git add wfcllm/watermark/lsh_space.py tests/watermark/test_lsh_space.py
git commit -m "feat: add LSHSpace for global hyperplane management"
```

---

### Task 3: Rewrite WatermarkKeying to return valid-set G

**Files:**
- Modify: `wfcllm/watermark/keying.py`
- Modify: `tests/watermark/test_keying.py`

**Context:** The old `derive(parent_node_type, node_type) -> (v, t)` interface is replaced. The new interface `derive(parent_node_type) -> frozenset[tuple[int,...]]` uses only parent_node_type as seed and returns a frozenset of valid LSH signatures. The `WatermarkKeying.__init__` now takes `d` and `gamma` instead of `embed_dim`.

**Step 1: Replace test file**

Replace the entire content of `tests/watermark/test_keying.py` with:

```python
"""Tests for wfcllm.watermark.keying (LSH version)."""

from __future__ import annotations

import pytest
from wfcllm.watermark.keying import WatermarkKeying


class TestWatermarkKeying:
    @pytest.fixture
    def keying(self):
        return WatermarkKeying(secret_key="test-secret", d=3, gamma=0.5)

    def test_derive_returns_frozenset(self, keying):
        G = keying.derive("module")
        assert isinstance(G, frozenset)

    def test_derive_set_size_matches_gamma(self, keying):
        """With d=3 and gamma=0.5, G should have round(0.5 * 8) = 4 elements."""
        G = keying.derive("module")
        assert len(G) == 4

    def test_derive_elements_are_d_tuples(self, keying):
        G = keying.derive("module")
        for sig in G:
            assert isinstance(sig, tuple)
            assert len(sig) == 3
            assert all(b in (0, 1) for b in sig)

    def test_derive_deterministic(self, keying):
        G1 = keying.derive("module")
        G2 = keying.derive("module")
        assert G1 == G2

    def test_different_parent_different_G(self, keying):
        G1 = keying.derive("module")
        G2 = keying.derive("for_statement")
        assert G1 != G2

    def test_different_key_different_G(self):
        k1 = WatermarkKeying(secret_key="key-a", d=3, gamma=0.5)
        k2 = WatermarkKeying(secret_key="key-b", d=3, gamma=0.5)
        G1 = k1.derive("module")
        G2 = k2.derive("module")
        assert G1 != G2

    def test_gamma_controls_set_size(self):
        k_25 = WatermarkKeying(secret_key="k", d=3, gamma=0.25)
        k_75 = WatermarkKeying(secret_key="k", d=3, gamma=0.75)
        assert len(k_25.derive("module")) == 2  # round(0.25 * 8)
        assert len(k_75.derive("module")) == 6  # round(0.75 * 8)

    def test_G_is_subset_of_all_signatures(self, keying):
        """G must only contain valid d-bit signatures."""
        all_sigs = {
            tuple(int(b) for b in format(i, f"0{3}b"))
            for i in range(2 ** 3)
        }
        G = keying.derive("module")
        assert G.issubset(all_sigs)
```

**Step 2: Run tests to verify they fail**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_keying.py -v
```
Expected: FAIL (old derive signature, old return type)

**Step 3: Replace implementation**

Replace the entire content of `wfcllm/watermark/keying.py` with:

```python
"""Watermark key derivation: valid LSH region set from AST topology."""

from __future__ import annotations

import hashlib
import hmac


class WatermarkKeying:
    """Derive the valid LSH signature set G from secret key and parent node type.

    The seed uses ONLY parent_node_type (not the current node's type) so that
    semantically equivalent transformations of a block do not change its target region.
    """

    def __init__(self, secret_key: str, d: int, gamma: float):
        self._key = secret_key.encode("utf-8")
        self._d = d
        self._gamma = gamma

    def derive(self, parent_node_type: str) -> frozenset[tuple[int, ...]]:
        """Return valid LSH signature set G for a block with given parent node type.

        Args:
            parent_node_type: AST type of the parent node (e.g. "module", "for_statement").

        Returns:
            frozenset of d-bit tuples that constitute the valid region set G.
            A block passes the watermark check iff its LSH signature is in G.
        """
        message = parent_node_type.encode("utf-8")
        digest = hmac.new(self._key, message, hashlib.sha256).digest()

        seed = int.from_bytes(digest[:8], "big")

        # Enumerate all 2^d possible signatures
        all_sigs = [
            tuple(int(b) for b in format(i, f"0{self._d}b"))
            for i in range(2 ** self._d)
        ]

        # Deterministic Fisher-Yates shuffle using seed
        import random
        rng = random.Random(seed)
        shuffled = list(all_sigs)
        rng.shuffle(shuffled)

        k = round(self._gamma * len(all_sigs))
        return frozenset(shuffled[:k])
```

**Step 4: Run tests to verify they pass**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_keying.py -v
```
Expected: All PASS

**Step 5: Commit**

```bash
git add wfcllm/watermark/keying.py tests/watermark/test_keying.py
git commit -m "feat: rewrite WatermarkKeying to return LSH valid-set G"
```

---

### Task 4: Rewrite ProjectionVerifier for LSH

**Files:**
- Modify: `wfcllm/watermark/verifier.py`
- Modify: `tests/watermark/test_verifier.py`

**Context:** The old `verify(code_text, v, t, margin) -> VerifyResult` with `projection` and `target_sign` fields is replaced. The new interface `verify(code_text, valid_set, margin) -> VerifyResult` stores `min_margin` instead of `projection` and has no `target_sign`. The verifier now holds an `LSHSpace` reference.

**Step 1: Replace test file**

Replace the entire content of `tests/watermark/test_verifier.py` with:

```python
"""Tests for wfcllm.watermark.verifier (LSH version)."""

from __future__ import annotations

import torch
import pytest
from unittest.mock import MagicMock

from wfcllm.watermark.lsh_space import LSHSpace
from wfcllm.watermark.verifier import ProjectionVerifier, VerifyResult


def _make_lsh_space(d: int = 3, embed_dim: int = 128) -> LSHSpace:
    return LSHSpace(secret_key="test-secret", embed_dim=embed_dim, d=d)


def _make_encoder_returning(vec: torch.Tensor):
    """Return mock encoder that always outputs vec (shape (1, embed_dim))."""
    encoder = MagicMock()
    encoder.return_value = vec.unsqueeze(0)
    return encoder


def _make_tokenizer():
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.zeros(1, 10, dtype=torch.long),
        "attention_mask": torch.ones(1, 10, dtype=torch.long),
    }
    return tokenizer


class TestVerifyResult:
    def test_passed_true(self):
        r = VerifyResult(passed=True, min_margin=0.5)
        assert r.passed is True

    def test_passed_false(self):
        r = VerifyResult(passed=False, min_margin=0.05)
        assert r.passed is False


class TestProjectionVerifier:
    def test_verify_pass_when_sign_in_valid_set_and_margin_ok(self):
        """verify passes when sign ∈ valid_set and min_margin > margin."""
        lsh = _make_lsh_space(d=3)
        # Build a vector u, find its sign, put that sign in valid_set
        u = torch.randn(128)
        sig = lsh.sign(u)
        valid_set = frozenset([sig])

        verifier = ProjectionVerifier(
            _make_encoder_returning(u), _make_tokenizer(), lsh_space=lsh, device="cpu"
        )
        # margin=0.0 to always pass the margin check
        result = verifier.verify("x = 1", valid_set, margin=0.0)
        assert result.passed is True

    def test_verify_fail_when_sign_not_in_valid_set(self):
        """verify fails when sign ∉ valid_set."""
        lsh = _make_lsh_space(d=3)
        u = torch.randn(128)
        sig = lsh.sign(u)
        # Flip one bit to get a different signature
        wrong_sig = tuple(1 - b for b in sig)
        valid_set = frozenset([wrong_sig])

        verifier = ProjectionVerifier(
            _make_encoder_returning(u), _make_tokenizer(), lsh_space=lsh, device="cpu"
        )
        result = verifier.verify("x = 1", valid_set, margin=0.0)
        assert result.passed is False

    def test_verify_fail_when_margin_not_satisfied(self):
        """verify fails when min_margin <= margin threshold."""
        lsh = _make_lsh_space(d=3)
        u = torch.randn(128)
        sig = lsh.sign(u)
        valid_set = frozenset([sig])

        verifier = ProjectionVerifier(
            _make_encoder_returning(u), _make_tokenizer(), lsh_space=lsh, device="cpu"
        )
        # margin=1.0 is impossible to satisfy
        result = verifier.verify("x = 1", valid_set, margin=1.0)
        assert result.passed is False

    def test_verify_result_has_min_margin_field(self):
        lsh = _make_lsh_space(d=3)
        u = torch.randn(128)
        sig = lsh.sign(u)
        valid_set = frozenset([sig])

        verifier = ProjectionVerifier(
            _make_encoder_returning(u), _make_tokenizer(), lsh_space=lsh, device="cpu"
        )
        result = verifier.verify("x = 1", valid_set, margin=0.0)
        assert isinstance(result.min_margin, float)
        assert 0.0 <= result.min_margin <= 1.0
```

**Step 2: Run tests to verify they fail**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_verifier.py -v
```
Expected: FAIL (old interface)

**Step 3: Replace implementation**

Replace the entire content of `wfcllm/watermark/verifier.py` with:

```python
"""LSH-based semantic verification for watermark embedding."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from wfcllm.watermark.lsh_space import LSHSpace


@dataclass
class VerifyResult:
    """Result of a watermark LSH verification."""

    passed: bool
    min_margin: float


class ProjectionVerifier:
    """Verify if a code block's LSH signature falls in the valid set G."""

    def __init__(self, encoder, tokenizer, lsh_space: LSHSpace, device: str = "cuda"):
        self._encoder = encoder
        self._tokenizer = tokenizer
        self._lsh_space = lsh_space
        self._device = device

    @torch.no_grad()
    def verify(
        self,
        code_text: str,
        valid_set: frozenset[tuple[int, ...]],
        margin: float,
    ) -> VerifyResult:
        """Check if the block's LSH signature is in valid_set with sufficient margin.

        Args:
            code_text: Source code of the statement block.
            valid_set: Set of valid LSH signatures G for this block's position.
            margin: Minimum min_margin required (0.0 at extraction time).

        Returns:
            VerifyResult with passed=True iff sign ∈ valid_set AND min_margin > margin.
        """
        inputs = self._tokenizer(
            code_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        input_ids = inputs["input_ids"].to(self._device)
        attention_mask = inputs["attention_mask"].to(self._device)
        u = self._encoder(input_ids, attention_mask).squeeze(0).cpu()

        sig = self._lsh_space.sign(u)
        mm = self._lsh_space.min_margin(u)

        passed = (sig in valid_set) and (mm > margin)
        return VerifyResult(passed=passed, min_margin=mm)
```

**Step 4: Run tests to verify they pass**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_verifier.py -v
```
Expected: All PASS

**Step 5: Commit**

```bash
git add wfcllm/watermark/verifier.py tests/watermark/test_verifier.py
git commit -m "feat: rewrite ProjectionVerifier for LSH signature verification"
```

---

### Task 5: Update extract/config.py — replace projection/target_sign with min_margin

**Files:**
- Modify: `wfcllm/extract/config.py`
- Modify: `tests/extract/test_config.py`

**Context:** `BlockScore` currently has `projection: float` and `target_sign: int`. These belong to the old projection scheme. The LSH version uses `min_margin: float` instead.

**Step 1: Write the failing test**

Open `tests/extract/test_config.py` and add:

```python
def test_block_score_has_min_margin():
    from wfcllm.extract.config import BlockScore
    s = BlockScore(block_id="0", score=1, min_margin=0.3)
    assert s.min_margin == 0.3

def test_block_score_no_projection_field():
    import dataclasses
    from wfcllm.extract.config import BlockScore
    fields = {f.name for f in dataclasses.fields(BlockScore)}
    assert "projection" not in fields
    assert "target_sign" not in fields
```

**Step 2: Run tests to verify they fail**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_config.py -v
```
Expected: FAIL

**Step 3: Write minimal implementation**

Replace the `BlockScore` dataclass in `wfcllm/extract/config.py`:

```python
@dataclass
class BlockScore:
    """Score result for a single statement block."""

    block_id: str
    score: int        # +1 (hit) or -1 (miss)
    min_margin: float
    selected: bool = False
```

**Step 4: Run tests to verify they pass**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_config.py -v
```
Expected: All PASS

**Step 5: Commit**

```bash
git add wfcllm/extract/config.py tests/extract/test_config.py
git commit -m "feat: replace projection/target_sign with min_margin in BlockScore"
```

---

### Task 6: Update HypothesisTester to parameterize gamma

**Files:**
- Modify: `wfcllm/extract/hypothesis.py`
- Modify: `tests/extract/test_hypothesis.py`

**Context:** The Z-score formula is currently hardcoded for γ=0.5: `z = (x - m/2) / sqrt(m/4)`. It must be parameterized: `z = (x - m*gamma) / sqrt(m * gamma * (1-gamma))`.

**Step 1: Write the failing test**

In `tests/extract/test_hypothesis.py`, add a new test class (keep all existing tests unchanged):

```python
class TestHypothesisTesterGamma:
    def test_custom_gamma_z_score(self):
        """With gamma=0.25 and all hits, Z-score uses correct formula."""
        import math
        tester = HypothesisTester(z_threshold=3.0, gamma=0.25)
        scores = [_make_score(str(i), 1) for i in range(20)]
        result = tester.test(selected_scores=scores, total_blocks=20)
        # Z = (20 - 20*0.25) / sqrt(20 * 0.25 * 0.75) = 15 / sqrt(3.75)
        expected_z = 15 / math.sqrt(3.75)
        assert result.z_score == pytest.approx(expected_z, rel=1e-6)

    def test_default_gamma_is_half(self):
        """Default gamma=0.5 produces same result as original formula."""
        import math
        tester = HypothesisTester(z_threshold=3.0)
        scores = [_make_score(str(i), 1) for i in range(20)]
        result = tester.test(selected_scores=scores, total_blocks=20)
        expected_z = (20 - 10) / math.sqrt(5)
        assert result.z_score == pytest.approx(expected_z, rel=1e-6)
```

**Step 2: Run tests to verify they fail**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_hypothesis.py::TestHypothesisTesterGamma -v
```
Expected: FAIL

**Step 3: Write minimal implementation**

In `wfcllm/extract/hypothesis.py`, update `__init__` and the Z-score line:

```python
class HypothesisTester:
    """One-sided Z-test for watermark presence."""

    def __init__(self, z_threshold: float = 3.0, gamma: float = 0.5):
        self._z_threshold = z_threshold
        self._gamma = gamma

    def test(self, selected_scores, total_blocks):
        m = len(selected_scores)
        if m == 0:
            return DetectionResult(
                is_watermarked=False, z_score=0.0, p_value=1.0,
                total_blocks=total_blocks, independent_blocks=0, hit_blocks=0,
                block_details=list(selected_scores),
            )
        x = sum(1 for s in selected_scores if s.score == 1)
        gamma = self._gamma
        z_score = (x - m * gamma) / math.sqrt(m * gamma * (1 - gamma))
        p_value = float(norm.sf(z_score))
        return DetectionResult(
            is_watermarked=z_score > self._z_threshold,
            z_score=z_score, p_value=p_value,
            total_blocks=total_blocks, independent_blocks=m, hit_blocks=x,
            block_details=list(selected_scores),
        )
```

**Step 4: Run all hypothesis tests**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_hypothesis.py -v
```
Expected: All PASS (old tests still pass because default gamma=0.5)

**Step 5: Commit**

```bash
git add wfcllm/extract/hypothesis.py tests/extract/test_hypothesis.py
git commit -m "feat: parameterize HypothesisTester Z-score formula with gamma"
```

---

### Task 7: Update BlockScorer for LSH interface

**Files:**
- Modify: `wfcllm/extract/scorer.py`
- Modify: `tests/extract/test_scorer.py`

**Context:** `BlockScorer` currently calls `keying.derive(parent, node_type)` → `(v, t)` and `verifier.verify(text, v, t, 0.0)`. The new interface is `keying.derive(parent_node_type)` → `frozenset` and `verifier.verify(text, valid_set, 0.0)`. `BlockScore` no longer has `projection`/`target_sign`, only `min_margin`.

**Step 1: Replace test file**

Replace the entire content of `tests/extract/test_scorer.py` with:

```python
"""Tests for BlockScorer (LSH version)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from wfcllm.common.ast_parser import StatementBlock
from wfcllm.extract.config import BlockScore
from wfcllm.extract.scorer import BlockScorer
from wfcllm.watermark.keying import WatermarkKeying
from wfcllm.watermark.verifier import VerifyResult


def _make_block(
    block_id: str,
    node_type: str = "expression_statement",
    source: str = "x = 1",
    parent_id: str | None = None,
    children_ids: list[str] | None = None,
    depth: int = 0,
) -> StatementBlock:
    return StatementBlock(
        block_id=block_id,
        block_type="simple",
        node_type=node_type,
        source=source,
        start_line=1,
        end_line=1,
        depth=depth,
        parent_id=parent_id,
        children_ids=children_ids or [],
    )


class TestBlockScorer:
    @pytest.fixture
    def keying(self):
        return WatermarkKeying(secret_key="test-key", d=3, gamma=0.5)

    @pytest.fixture
    def mock_verifier(self):
        return MagicMock()

    def test_score_single_block_hit(self, keying, mock_verifier):
        """passed=True from verifier -> score = +1."""
        mock_verifier.verify.return_value = VerifyResult(passed=True, min_margin=0.4)
        scorer = BlockScorer(keying, mock_verifier)
        block = _make_block("0")
        result = scorer.score_block(block, blocks=[block])

        assert isinstance(result, BlockScore)
        assert result.block_id == "0"
        assert result.score == 1
        assert result.min_margin == 0.4

    def test_score_single_block_miss(self, keying, mock_verifier):
        """passed=False from verifier -> score = -1."""
        mock_verifier.verify.return_value = VerifyResult(passed=False, min_margin=0.05)
        scorer = BlockScorer(keying, mock_verifier)
        block = _make_block("0")
        result = scorer.score_block(block, blocks=[block])

        assert result.score == -1
        assert result.min_margin == 0.05

    def test_verify_called_with_margin_zero(self, keying, mock_verifier):
        """Extraction always calls verify with margin=0.0."""
        mock_verifier.verify.return_value = VerifyResult(passed=True, min_margin=0.3)
        scorer = BlockScorer(keying, mock_verifier)
        block = _make_block("0")
        scorer.score_block(block, blocks=[block])

        call_args = mock_verifier.verify.call_args
        # verify(code_text, valid_set, margin)
        assert call_args[0][2] == 0.0

    def test_root_block_uses_module_parent(self, keying, mock_verifier):
        """Root-level block derives G from parent='module'."""
        mock_verifier.verify.return_value = VerifyResult(passed=True, min_margin=0.3)
        scorer = BlockScorer(keying, mock_verifier)
        block = _make_block("0", parent_id=None)

        scorer.score_block(block, blocks=[block])

        expected_G = keying.derive("module")
        call_args = mock_verifier.verify.call_args
        actual_G = call_args[0][1]
        assert actual_G == expected_G

    def test_nested_block_uses_parent_node_type(self, keying, mock_verifier):
        """Nested block derives G from parent's node_type."""
        mock_verifier.verify.return_value = VerifyResult(passed=True, min_margin=0.3)
        scorer = BlockScorer(keying, mock_verifier)
        parent = _make_block("0", node_type="for_statement", children_ids=["1"])
        child = _make_block("1", node_type="expression_statement", parent_id="0", depth=1)

        scorer.score_block(child, blocks=[parent, child])

        expected_G = keying.derive("for_statement")
        call_args = mock_verifier.verify.call_args
        actual_G = call_args[0][1]
        assert actual_G == expected_G

    def test_score_all_returns_all_blocks(self, keying, mock_verifier):
        mock_verifier.verify.return_value = VerifyResult(passed=True, min_margin=0.3)
        scorer = BlockScorer(keying, mock_verifier)
        blocks = [_make_block("0"), _make_block("1")]
        results = scorer.score_all(blocks)

        assert len(results) == 2
        assert results[0].block_id == "0"
        assert results[1].block_id == "1"
```

**Step 2: Run tests to verify they fail**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_scorer.py -v
```
Expected: FAIL

**Step 3: Replace implementation**

Replace the entire content of `wfcllm/extract/scorer.py` with:

```python
"""Semantic feature scoring for statement blocks (LSH version)."""

from __future__ import annotations

from wfcllm.common.ast_parser import StatementBlock
from wfcllm.extract.config import BlockScore
from wfcllm.watermark.keying import WatermarkKeying
from wfcllm.watermark.verifier import ProjectionVerifier


class BlockScorer:
    """Score each statement block for watermark hit/miss via LSH."""

    def __init__(self, keying: WatermarkKeying, verifier: ProjectionVerifier):
        self._keying = keying
        self._verifier = verifier

    def score_block(
        self, block: StatementBlock, blocks: list[StatementBlock]
    ) -> BlockScore:
        parent_node_type = self._resolve_parent_type(block, blocks)
        valid_set = self._keying.derive(parent_node_type)
        result = self._verifier.verify(block.source, valid_set, 0.0)

        score = 1 if result.passed else -1
        return BlockScore(
            block_id=block.block_id,
            score=score,
            min_margin=result.min_margin,
        )

    def score_all(self, blocks: list[StatementBlock]) -> list[BlockScore]:
        return [self.score_block(b, blocks) for b in blocks]

    @staticmethod
    def _resolve_parent_type(block: StatementBlock, blocks: list[StatementBlock]) -> str:
        if block.parent_id is None:
            return "module"
        block_map = {b.block_id: b for b in blocks}
        return block_map[block.parent_id].node_type
```

**Step 4: Run tests to verify they pass**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_scorer.py -v
```
Expected: All PASS

**Step 5: Commit**

```bash
git add wfcllm/extract/scorer.py tests/extract/test_scorer.py
git commit -m "feat: update BlockScorer to use LSH keying and verifier interface"
```

---

### Task 8: Update WatermarkDetector (extract/detector.py)

**Files:**
- Modify: `wfcllm/extract/detector.py`
- Modify: `wfcllm/extract/config.py` (add lsh fields to ExtractConfig)
- Modify: `tests/extract/test_detector.py`

**Context:** `WatermarkDetector.__init__` currently builds `WatermarkKeying(secret_key, embed_dim)` and `ProjectionVerifier(encoder, tokenizer)`. The LSH version needs `LSHSpace`, updated `WatermarkKeying`, and updated `ProjectionVerifier`. Also `ExtractConfig` needs `lsh_d` and `lsh_gamma`.

**Step 1: Add lsh fields to ExtractConfig**

In `wfcllm/extract/config.py`, update the `ExtractConfig` dataclass:

```python
@dataclass
class ExtractConfig:
    """Configuration for the watermark extraction pipeline."""

    secret_key: str
    embed_dim: int = 128
    z_threshold: float = 3.0
    lsh_d: int = 3
    lsh_gamma: float = 0.5
```

**Step 2: Update WatermarkDetector**

Replace the entire content of `wfcllm/extract/detector.py` with:

```python
"""High-level watermark detection entry point."""

from __future__ import annotations

from wfcllm.common.ast_parser import extract_statement_blocks
from wfcllm.extract.config import DetectionResult, ExtractConfig
from wfcllm.extract.dp_selector import DPSelector
from wfcllm.extract.hypothesis import HypothesisTester
from wfcllm.extract.scorer import BlockScorer
from wfcllm.watermark.keying import WatermarkKeying
from wfcllm.watermark.lsh_space import LSHSpace
from wfcllm.watermark.verifier import ProjectionVerifier


class WatermarkDetector:
    """One-call watermark detection pipeline."""

    def __init__(
        self,
        config: ExtractConfig,
        encoder,
        tokenizer,
        device: str = "cuda",
    ):
        lsh_space = LSHSpace(config.secret_key, config.embed_dim, config.lsh_d)
        keying = WatermarkKeying(config.secret_key, config.lsh_d, config.lsh_gamma)
        verifier = ProjectionVerifier(encoder, tokenizer, lsh_space=lsh_space, device=device)
        self._scorer = BlockScorer(keying, verifier)
        self._dp = DPSelector()
        self._tester = HypothesisTester(config.z_threshold, gamma=config.lsh_gamma)

    def detect(self, code: str) -> DetectionResult:
        blocks = extract_statement_blocks(code)
        if not blocks:
            return DetectionResult(
                is_watermarked=False, z_score=0.0, p_value=1.0,
                total_blocks=0, independent_blocks=0, hit_blocks=0,
                block_details=[],
            )

        scores = self._scorer.score_all(blocks)
        selected_ids = set(self._dp.select(blocks, scores))

        for s in scores:
            s.selected = s.block_id in selected_ids

        selected_scores = [s for s in scores if s.selected]
        result = self._tester.test(selected_scores, total_blocks=len(blocks))
        result.block_details = scores
        return result
```

**Step 3: Run existing detector tests**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_detector.py -v
```
Expected: PASS (detector tests use mocks and should continue to work)

**Step 4: Commit**

```bash
git add wfcllm/extract/detector.py wfcllm/extract/config.py
git commit -m "feat: update WatermarkDetector and ExtractConfig for LSH pipeline"
```

---

### Task 9: Update WatermarkGenerator

**Files:**
- Modify: `wfcllm/watermark/generator.py`
- Modify: `tests/watermark/test_generator.py`

**Context:** `WatermarkGenerator.__init__` creates `WatermarkKeying(secret_key, embed_dim)` and `ProjectionVerifier(encoder, encoder_tokenizer)`. In `generate()`, the calls are `keying.derive(parent, node_type)` → `(v, t)` and `verifier.verify(text, v, t, margin)`. All must be updated to LSH interface.

**Step 1: Check existing generator tests for breakage**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_generator.py -v 2>&1 | head -40
```
Note which tests fail due to interface changes.

**Step 2: Update generator.py**

In `wfcllm/watermark/generator.py`, make these targeted changes:

1. Add import at top: `from wfcllm.watermark.lsh_space import LSHSpace`

2. In `__init__`, replace:
```python
self._keying = WatermarkKeying(config.secret_key, config.encoder_embed_dim)
self._verifier = ProjectionVerifier(
    encoder, encoder_tokenizer, device=config.encoder_device
)
```
with:
```python
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
```

3. In `generate()`, replace the simple-block branch derive call:
```python
v, t = self._keying.derive(
    event.parent_node_type or "module", event.node_type
)
result = self._verifier.verify(event.block_text, v, t, margin)
```
with:
```python
valid_set = self._keying.derive(event.parent_node_type or "module")
result = self._verifier.verify(event.block_text, valid_set, margin)
```

4. In the sub-loop retry section, replace:
```python
result = self._verifier.verify(
    sub_event.block_text, v, t, margin
)
```
with:
```python
result = self._verifier.verify(
    sub_event.block_text, valid_set, margin
)
```

5. In the compound fallback section, replace:
```python
v, t = self._keying.derive(
    event.parent_node_type or "module", event.node_type
)
result = self._verifier.verify(event.block_text, v, t, margin)
```
with:
```python
valid_set = self._keying.derive(event.parent_node_type or "module")
result = self._verifier.verify(event.block_text, valid_set, margin)
```

**Step 3: Run generator tests**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_generator.py -v
```
Expected: All PASS. If tests reference old `VerifyResult` fields (`projection`, `target_sign`), update only the test helper/mock setup to use the new `VerifyResult(passed=..., min_margin=...)` signature.

**Step 4: Commit**

```bash
git add wfcllm/watermark/generator.py tests/watermark/test_generator.py
git commit -m "feat: update WatermarkGenerator to use LSHSpace and new keying/verifier interface"
```

---

### Task 10: Run full test suite and fix any remaining breakage

**Step 1: Run all tests**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -v 2>&1 | tail -40
```

**Step 2: Fix failures**

If any test fails due to:
- `BlockScore` construction using old `projection`/`target_sign` args: update the test helper (`_make_score`) to use `min_margin=0.5` and remove `target_sign`.
- `VerifyResult` construction using old args: replace with `VerifyResult(passed=..., min_margin=...)`.
- `WatermarkKeying` constructed with old `embed_dim` arg: replace with `d=3, gamma=0.5`.
- `ProjectionVerifier` constructed without `lsh_space`: add a fixture that creates an `LSHSpace`.

Fix each failing test file one at a time.

**Step 3: Re-run until clean**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -v
```
Expected: All PASS

**Step 4: Commit all fixes**

```bash
git add -u
git commit -m "test: fix remaining test breakage from LSH interface migration"
```

---

### Task 11: Final verification and branch summary

**Step 1: Run full suite one last time**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -v --tb=short
```
Expected: All PASS, 0 failures

**Step 2: Review git log**

```bash
git log develop..HEAD --oneline
```
Should show all commits from Tasks 1-10.

**Step 3: Done**

The `develop-LSH` branch is ready for review / PR into `develop`.
