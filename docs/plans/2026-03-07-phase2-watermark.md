# Phase 2: Watermark Embedding Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement generation-time watermark embedding in `wfcllm/watermark/` — intercept LLM-generated code at statement-block boundaries, verify semantic projection against watermark targets, and use rejection sampling with KV-Cache rollback to embed watermark signals.

**Architecture:** Custom token-by-token generation loop using `model.forward()` (not `model.generate()`). Seven modules: config, interceptor, entropy estimator, key derivation, projection verifier, KV-Cache manager, and generator orchestrator. Each module is independently testable. The encoder uses original CodeT5 as placeholder (not yet fine-tuned).

**Tech Stack:** PyTorch, HuggingFace transformers (AutoModelForCausalLM, AutoTokenizer), tree-sitter / tree-sitter-python, HMAC-SHA256 (hashlib), existing `wfcllm/encoder/` and `wfcllm/common/ast_parser.py`.

---

### Task 1: WatermarkConfig dataclass

**Files:**
- Create: `wfcllm/watermark/config.py`
- Test: `tests/watermark/test_config.py`

**Step 1: Write the failing test**

Create `tests/watermark/__init__.py` (empty) and `tests/watermark/test_config.py`:

```python
"""Tests for wfcllm.watermark.config."""

from wfcllm.watermark.config import WatermarkConfig


class TestWatermarkConfig:
    def test_required_secret_key(self):
        cfg = WatermarkConfig(secret_key="test-key")
        assert cfg.secret_key == "test-key"

    def test_default_encoder_path(self):
        cfg = WatermarkConfig(secret_key="k")
        assert cfg.encoder_model_path == "Salesforce/codet5-base"

    def test_default_embed_dim(self):
        cfg = WatermarkConfig(secret_key="k")
        assert cfg.encoder_embed_dim == 128

    def test_default_margin_params(self):
        cfg = WatermarkConfig(secret_key="k")
        assert cfg.margin_base == 0.1
        assert cfg.margin_alpha == 0.05

    def test_default_sampling_params(self):
        cfg = WatermarkConfig(secret_key="k")
        assert cfg.max_retries == 5
        assert cfg.temperature == 0.8
        assert cfg.top_p == 0.95
        assert cfg.top_k == 50

    def test_default_generation_params(self):
        cfg = WatermarkConfig(secret_key="k")
        assert cfg.max_new_tokens == 512
        assert cfg.eos_token_id is None
        assert cfg.enable_fallback is True

    def test_custom_values(self):
        cfg = WatermarkConfig(
            secret_key="my-key",
            margin_base=0.2,
            max_retries=3,
            temperature=0.6,
        )
        assert cfg.margin_base == 0.2
        assert cfg.max_retries == 3
        assert cfg.temperature == 0.6
```

**Step 2: Run test to verify it fails**

Run: `conda run -n WFCLLM pytest tests/watermark/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'wfcllm.watermark.config'`

**Step 3: Write minimal implementation**

Create `wfcllm/watermark/config.py`:

```python
"""Configuration for the watermark embedding pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WatermarkConfig:
    """All parameters for watermark embedding during code generation."""

    # Key
    secret_key: str

    # Encoder
    encoder_model_path: str = "Salesforce/codet5-base"
    encoder_embed_dim: int = 128
    encoder_device: str = "cuda"

    # Margin
    margin_base: float = 0.1
    margin_alpha: float = 0.05

    # Rejection sampling
    max_retries: int = 5
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50

    # Generation
    max_new_tokens: int = 512
    eos_token_id: int | None = None

    # Fallback
    enable_fallback: bool = True
```

Update `wfcllm/watermark/__init__.py`:

```python
"""Generation-time watermark embedding module."""

from wfcllm.watermark.config import WatermarkConfig

__all__ = ["WatermarkConfig"]
```

**Step 4: Run test to verify it passes**

Run: `conda run -n WFCLLM pytest tests/watermark/test_config.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add tests/watermark/__init__.py tests/watermark/test_config.py wfcllm/watermark/config.py wfcllm/watermark/__init__.py
git commit -m "feat(watermark): add WatermarkConfig dataclass"
```

---

### Task 2: NodeEntropyEstimator — heuristic entropy lookup

**Files:**
- Create: `wfcllm/watermark/entropy.py`
- Test: `tests/watermark/test_entropy.py`

**Context:** The entropy lookup table comes from experiment results at `experiment/node_entropy/results/node_entropy_results.json`. There are 133 AST node types with pre-computed mean entropy values. The estimator traverses all AST sub-nodes of a statement block, looks up each node type's entropy, and sums them. The dynamic margin formula is: `m = m_base + alpha * block_entropy`.

**Step 1: Write the failing test**

Create `tests/watermark/test_entropy.py`:

```python
"""Tests for wfcllm.watermark.entropy."""

import pytest
from wfcllm.watermark.entropy import NodeEntropyEstimator
from wfcllm.watermark.config import WatermarkConfig


class TestNodeEntropyEstimator:
    @pytest.fixture
    def estimator(self):
        return NodeEntropyEstimator()

    def test_known_node_type_in_table(self, estimator):
        """The table should contain well-known AST node types."""
        assert "identifier" in estimator.ENTROPY_TABLE
        assert "assignment" in estimator.ENTROPY_TABLE
        assert "return_statement" in estimator.ENTROPY_TABLE

    def test_zero_entropy_tokens(self, estimator):
        """Punctuation/keyword tokens should have zero entropy."""
        assert estimator.ENTROPY_TABLE["="] == 0.0
        assert estimator.ENTROPY_TABLE[":"] == 0.0
        assert estimator.ENTROPY_TABLE["("] == 0.0

    def test_high_entropy_types(self, estimator):
        """Complex node types should have higher entropy."""
        assert estimator.ENTROPY_TABLE["boolean_operator"] > 0.9
        assert estimator.ENTROPY_TABLE["for_statement"] > 0.5

    def test_estimate_simple_assignment(self, estimator):
        """'x = 1' has identifiers, =, integer — sum their entropies."""
        entropy = estimator.estimate_block_entropy("x = 1")
        # Should be > 0 because of identifier and expression_statement
        assert entropy > 0.0

    def test_estimate_empty_string(self, estimator):
        entropy = estimator.estimate_block_entropy("")
        assert entropy == 0.0

    def test_estimate_complex_block_higher(self, estimator):
        """A for loop should have higher total entropy than a simple assignment."""
        simple = estimator.estimate_block_entropy("x = 1")
        compound = estimator.estimate_block_entropy(
            "for i in range(10):\n    x = i + 1"
        )
        assert compound > simple

    def test_compute_margin(self, estimator):
        """Margin = m_base + alpha * entropy."""
        config = WatermarkConfig(secret_key="k", margin_base=0.1, margin_alpha=0.05)
        margin = estimator.compute_margin(2.0, config)
        assert margin == pytest.approx(0.1 + 0.05 * 2.0)

    def test_compute_margin_zero_entropy(self, estimator):
        config = WatermarkConfig(secret_key="k", margin_base=0.1, margin_alpha=0.05)
        margin = estimator.compute_margin(0.0, config)
        assert margin == pytest.approx(0.1)

    def test_unknown_node_type_uses_default(self, estimator):
        """Unknown node types should use DEFAULT_ENTROPY."""
        assert estimator.ENTROPY_TABLE.get("nonexistent_type") is None
        # The default is applied during traversal, not table lookup
        # Just verify the constant exists
        assert estimator.DEFAULT_ENTROPY >= 0.0

    def test_table_has_133_entries(self, estimator):
        """Table should contain all 133 known AST node types."""
        assert len(estimator.ENTROPY_TABLE) == 133
```

**Step 2: Run test to verify it fails**

Run: `conda run -n WFCLLM pytest tests/watermark/test_entropy.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'wfcllm.watermark.entropy'`

**Step 3: Write minimal implementation**

Create `wfcllm/watermark/entropy.py`:

```python
"""Heuristic entropy estimation for AST nodes.

Uses a pre-computed lookup table of mean entropy values per AST node type,
derived from experiment/node_entropy/results/node_entropy_results.json.
"""

from __future__ import annotations

from wfcllm.common.ast_parser import PythonParser
from wfcllm.watermark.config import WatermarkConfig


class NodeEntropyEstimator:
    """Estimate statement block entropy via AST node type lookup table."""

    DEFAULT_ENTROPY: float = 0.1

    # Pre-computed from experiment/node_entropy — 133 AST node types
    ENTROPY_TABLE: dict[str, float] = {
        "boolean_operator": 0.959,
        "for_statement": 0.7095,
        "function_definition": 0.504,
        "comparison_operator": 0.4944,
        "identifier": 0.4909,
        "conditional_expression": 0.4487,
        "if_statement": 0.4348,
        "while_statement": 0.4266,
        "binary_operator": 0.4026,
        "default_parameter": 0.3584,
        "module": 0.3571,
        "comment": 0.3333,
        "call": 0.3198,
        "block": 0.3116,
        "dictionary_comprehension": 0.3062,
        "elif_clause": 0.2992,
        "assignment": 0.2971,
        "expression_statement": 0.2817,
        "parenthesized_expression": 0.2809,
        "list_comprehension": 0.2779,
        "argument_list": 0.2251,
        "slice": 0.2214,
        "if_clause": 0.1979,
        "string_content": 0.1949,
        "generator_expression": 0.1901,
        "else_clause": 0.168,
        "float": 0.1672,
        "for_in_clause": 0.1556,
        "return_statement": 0.1539,
        "integer": 0.1522,
        "not_operator": 0.1496,
        "subscript": 0.1225,
        "yield": 0.1138,
        "tuple": 0.1105,
        "augmented_assignment": 0.107,
        "set_comprehension": 0.1048,
        "string": 0.0881,
        "expression_list": 0.0749,
        "lambda": 0.0716,
        "attribute": 0.0688,
        "list": 0.0569,
        "set": 0.0467,
        "parameters": 0.0451,
        "unary_operator": 0.0375,
        "class_definition": 0.0317,
        "pattern_list": 0.0259,
        "keyword_argument": 0.0187,
        "string_start": 0.017,
        "pair": 0.0145,
        "string_end": 0.0133,
        "import_from_statement": 0.008,
        "dotted_name": 0.0077,
        "=": 0.0,
        "def": 0.0,
        ":": 0.0,
        "(": 0.0,
        ")": 0.0,
        ",": 0.0,
        ".": 0.0,
        "class": 0.0,
        "return": 0.0,
        "[": 0.0,
        "]": 0.0,
        "if": 0.0,
        "<": 0.0,
        "and": 0.0,
        "in": 0.0,
        ">": 0.0,
        "for": 0.0,
        "+": 0.0,
        "+=": 0.0,
        "!=": 0.0,
        "while": 0.0,
        "-": 0.0,
        "==": 0.0,
        "%": 0.0,
        "//": 0.0,
        "true": 0.0,
        "false": 0.0,
        "else": 0.0,
        ">=": 0.0,
        "import": 0.0,
        "import_statement": 0.0,
        "/": 0.0,
        "*": 0.0,
        "list_splat": 0.0,
        "lambda_parameters": 0.0,
        "or": 0.0,
        ";": 0.0,
        "<=": 0.0,
        "break_statement": 0.0,
        "break": 0.0,
        "elif": 0.0,
        "**": 0.0,
        "-=": 0.0,
        "from": 0.0,
        "not": 0.0,
        "^": 0.0,
        "as": 0.0,
        "aliased_import": 0.0,
        "\\": 0.0,
        "*=": 0.0,
        "not in": 0.0,
        "{": 0.0,
        "}": 0.0,
        "&": 0.0,
        "|": 0.0,
        "<<": 0.0,
        "~": 0.0,
        "continue": 0.0,
        "continue_statement": 0.0,
        "<<=": 0.0,
        "/=": 0.0,
        "tuple_pattern": 0.0,
        ">>": 0.0,
        "|=": 0.0,
        "is": 0.0,
        "none": 0.0,
        "line_continuation": 0.0,
        "dictionary": 0.0,
        "except_clause": 0.0,
        "except": 0.0,
        "try_statement": 0.0,
        "try": 0.0,
        "escape_sequence": 0.0,
        ">>=": 0.0,
        "//=": 0.0,
        "%=": 0.0,
        "pass_statement": 0.0,
        "pass": 0.0,
        "del": 0.0,
        "delete_statement": 0.0,
        "is not": 0.0,
    }

    def estimate_block_entropy(self, block_source: str) -> float:
        """Parse block AST, traverse all sub-nodes, sum their entropies."""
        if not block_source.strip():
            return 0.0
        parser = PythonParser()
        tree = parser.parse(block_source)
        total = 0.0
        self._walk_and_sum(tree.root_node, total_ref=[0.0])
        return self._walk_and_sum_result(tree.root_node)

    def _walk_and_sum_result(self, node) -> float:
        total = self.ENTROPY_TABLE.get(node.type, self.DEFAULT_ENTROPY)
        for child in node.children:
            total += self._walk_and_sum_result(child)
        return total

    def _walk_and_sum(self, node, total_ref: list[float]) -> None:
        total_ref[0] += self.ENTROPY_TABLE.get(node.type, self.DEFAULT_ENTROPY)
        for child in node.children:
            self._walk_and_sum(child, total_ref)

    def compute_margin(self, block_entropy: float, config: WatermarkConfig) -> float:
        """Dynamic margin = m_base + alpha * block_entropy."""
        return config.margin_base + config.margin_alpha * block_entropy
```

Note: `estimate_block_entropy` uses `_walk_and_sum_result` (the clean recursive version). The `_walk_and_sum` helper is unused and should be removed during implementation — only `_walk_and_sum_result` is needed. The implementation should be simplified to:

```python
    def estimate_block_entropy(self, block_source: str) -> float:
        if not block_source.strip():
            return 0.0
        parser = PythonParser()
        tree = parser.parse(block_source)
        return self._sum_entropy(tree.root_node)

    def _sum_entropy(self, node) -> float:
        total = self.ENTROPY_TABLE.get(node.type, self.DEFAULT_ENTROPY)
        for child in node.children:
            total += self._sum_entropy(child)
        return total
```

**Step 4: Run test to verify it passes**

Run: `conda run -n WFCLLM pytest tests/watermark/test_entropy.py -v`
Expected: All 10 tests PASS

**Step 5: Commit**

```bash
git add wfcllm/watermark/entropy.py tests/watermark/test_entropy.py wfcllm/watermark/__init__.py
git commit -m "feat(watermark): add NodeEntropyEstimator with heuristic lookup table"
```

---

### Task 3: WatermarkKeying — key derivation

**Files:**
- Create: `wfcllm/watermark/keying.py`
- Test: `tests/watermark/test_keying.py`

**Context:** Derives a deterministic pseudo-random direction vector `v` and target bit `t` from the secret key + local AST topology (parent_node_type + node_type). Uses HMAC-SHA256 to hash the topology feature, seeds a PRNG, generates a unit vector in R^embed_dim, and extracts the target bit from the hash.

**Step 1: Write the failing test**

Create `tests/watermark/test_keying.py`:

```python
"""Tests for wfcllm.watermark.keying."""

import torch
import pytest
from wfcllm.watermark.keying import WatermarkKeying


class TestWatermarkKeying:
    @pytest.fixture
    def keying(self):
        return WatermarkKeying(secret_key="test-secret", embed_dim=128)

    def test_derive_returns_vector_and_bit(self, keying):
        v, t = keying.derive("module", "expression_statement")
        assert isinstance(v, torch.Tensor)
        assert v.shape == (128,)
        assert t in (0, 1)

    def test_vector_is_unit_normalized(self, keying):
        v, _ = keying.derive("module", "assignment")
        norm = torch.norm(v).item()
        assert abs(norm - 1.0) < 1e-5

    def test_deterministic(self, keying):
        v1, t1 = keying.derive("if_statement", "return_statement")
        v2, t2 = keying.derive("if_statement", "return_statement")
        assert torch.allclose(v1, v2)
        assert t1 == t2

    def test_different_topology_different_output(self, keying):
        v1, _ = keying.derive("module", "expression_statement")
        v2, _ = keying.derive("for_statement", "expression_statement")
        assert not torch.allclose(v1, v2)

    def test_different_key_different_output(self):
        k1 = WatermarkKeying(secret_key="key-a", embed_dim=128)
        k2 = WatermarkKeying(secret_key="key-b", embed_dim=128)
        v1, _ = k1.derive("module", "assignment")
        v2, _ = k2.derive("module", "assignment")
        assert not torch.allclose(v1, v2)

    def test_different_embed_dim(self):
        k64 = WatermarkKeying(secret_key="k", embed_dim=64)
        v, _ = k64.derive("module", "assignment")
        assert v.shape == (64,)

    def test_target_bit_distribution(self, keying):
        """Over many different inputs, t should be roughly 50/50."""
        bits = []
        for i in range(100):
            _, t = keying.derive(f"type_{i}", "expression_statement")
            bits.append(t)
        ratio = sum(bits) / len(bits)
        # Should be roughly balanced — allow 30-70% range
        assert 0.3 < ratio < 0.7
```

**Step 2: Run test to verify it fails**

Run: `conda run -n WFCLLM pytest tests/watermark/test_keying.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'wfcllm.watermark.keying'`

**Step 3: Write minimal implementation**

Create `wfcllm/watermark/keying.py`:

```python
"""Watermark key derivation from secret key and AST topology."""

from __future__ import annotations

import hashlib
import hmac

import torch


class WatermarkKeying:
    """Derive deterministic (direction vector, target bit) from key + topology."""

    def __init__(self, secret_key: str, embed_dim: int):
        self._key = secret_key.encode("utf-8")
        self._embed_dim = embed_dim

    def derive(self, parent_node_type: str, node_type: str) -> tuple[torch.Tensor, int]:
        """Derive (v, t) from local AST topology.

        Args:
            parent_node_type: AST type of the parent node.
            node_type: AST type of the current statement block node.

        Returns:
            v: Unit vector in R^embed_dim (float32).
            t: Target bit in {0, 1}.
        """
        # 1. HMAC-SHA256 of topology feature
        message = f"{parent_node_type}|{node_type}".encode("utf-8")
        digest = hmac.new(self._key, message, hashlib.sha256).digest()

        # 2. Seed PRNG with hash
        seed = int.from_bytes(digest[:8], "big")
        gen = torch.Generator()
        gen.manual_seed(seed)

        # 3. Generate direction vector from standard normal, then normalize
        v = torch.randn(self._embed_dim, generator=gen)
        v = v / v.norm()

        # 4. Target bit from last byte LSB
        t = digest[-1] & 1

        return v, t
```

**Step 4: Run test to verify it passes**

Run: `conda run -n WFCLLM pytest tests/watermark/test_keying.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add wfcllm/watermark/keying.py tests/watermark/test_keying.py
git commit -m "feat(watermark): add WatermarkKeying for key derivation"
```

---

### Task 4: ProjectionVerifier — semantic projection check

**Files:**
- Create: `wfcllm/watermark/verifier.py`
- Test: `tests/watermark/test_verifier.py`

**Context:** Uses the semantic encoder E to compute a code block's embedding vector u, then checks if `cos(u, v)` has the correct sign and exceeds the margin threshold. The encoder is `wfcllm.encoder.model.SemanticEncoder`. For now, we use original CodeT5 (no fine-tuning).

**Step 1: Write the failing test**

Create `tests/watermark/test_verifier.py`:

```python
"""Tests for wfcllm.watermark.verifier."""

import torch
import pytest
from unittest.mock import MagicMock
from wfcllm.watermark.verifier import ProjectionVerifier, VerifyResult


class TestVerifyResult:
    def test_passed_true(self):
        r = VerifyResult(passed=True, projection=0.5, target_sign=1, margin=0.1)
        assert r.passed is True

    def test_passed_false(self):
        r = VerifyResult(passed=False, projection=-0.05, target_sign=1, margin=0.1)
        assert r.passed is False


class TestProjectionVerifier:
    @pytest.fixture
    def mock_encoder(self):
        """Mock encoder that returns a fixed vector."""
        encoder = MagicMock()
        # Return a normalized vector pointing in positive direction
        fixed_vec = torch.randn(1, 128)
        fixed_vec = fixed_vec / fixed_vec.norm()
        encoder.return_value = fixed_vec
        encoder.eval = MagicMock(return_value=encoder)
        encoder.config = MagicMock()
        encoder.config.model_name = "Salesforce/codet5-base"
        return encoder, fixed_vec.squeeze(0)

    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }
        return tokenizer

    def test_verify_pass_positive_projection(self, mock_encoder, mock_tokenizer):
        encoder, fixed_vec = mock_encoder
        verifier = ProjectionVerifier(encoder, mock_tokenizer, device="cpu")
        # Direction = same as encoder output -> cos ~ 1.0
        v = fixed_vec.clone()
        result = verifier.verify("x = 1", v, t=1, margin=0.1)
        assert result.passed is True
        assert result.projection > 0

    def test_verify_pass_negative_projection(self, mock_encoder, mock_tokenizer):
        encoder, fixed_vec = mock_encoder
        verifier = ProjectionVerifier(encoder, mock_tokenizer, device="cpu")
        # Direction = negated -> cos ~ -1.0, target t=0 -> t*=-1
        v = -fixed_vec.clone()
        result = verifier.verify("x = 1", v, t=0, margin=0.1)
        # cos(u, -u) = -1.0, sign = -1, t* = -1 -> match
        assert result.passed is True

    def test_verify_fail_wrong_sign(self, mock_encoder, mock_tokenizer):
        encoder, fixed_vec = mock_encoder
        verifier = ProjectionVerifier(encoder, mock_tokenizer, device="cpu")
        # Direction same as output -> cos > 0, but target t=0 -> t*=-1
        v = fixed_vec.clone()
        result = verifier.verify("x = 1", v, t=0, margin=0.1)
        assert result.passed is False

    def test_verify_fail_below_margin(self, mock_encoder, mock_tokenizer):
        encoder, fixed_vec = mock_encoder
        verifier = ProjectionVerifier(encoder, mock_tokenizer, device="cpu")
        # Use nearly orthogonal direction -> small |cos|
        v = torch.zeros(128)
        v[0] = 1.0  # Arbitrary direction likely not aligned
        # With random encoder output, projection could be small
        result = verifier.verify("x = 1", v, t=1, margin=0.99)
        # Margin 0.99 is very strict — almost certainly fails
        assert result.passed is False

    def test_verify_result_contains_values(self, mock_encoder, mock_tokenizer):
        encoder, fixed_vec = mock_encoder
        verifier = ProjectionVerifier(encoder, mock_tokenizer, device="cpu")
        v = fixed_vec.clone()
        result = verifier.verify("x = 1", v, t=1, margin=0.1)
        assert isinstance(result.projection, float)
        assert result.target_sign in (-1, 1)
        assert isinstance(result.margin, float)
```

**Step 2: Run test to verify it fails**

Run: `conda run -n WFCLLM pytest tests/watermark/test_verifier.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'wfcllm.watermark.verifier'`

**Step 3: Write minimal implementation**

Create `wfcllm/watermark/verifier.py`:

```python
"""Semantic projection verification for watermark embedding."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class VerifyResult:
    """Result of a watermark projection verification."""

    passed: bool
    projection: float
    target_sign: int
    margin: float


class ProjectionVerifier:
    """Verify if a code block's semantic projection matches the watermark target."""

    def __init__(self, encoder, tokenizer, device: str = "cuda"):
        self._encoder = encoder
        self._tokenizer = tokenizer
        self._device = device

    @torch.no_grad()
    def verify(
        self, code_text: str, v: torch.Tensor, t: int, margin: float
    ) -> VerifyResult:
        """Check semantic projection of code against target direction.

        Args:
            code_text: Source code of the statement block.
            v: Direction vector (embed_dim,).
            t: Target bit in {0, 1}.
            margin: Minimum absolute projection required.

        Returns:
            VerifyResult with pass/fail and diagnostic values.
        """
        # Encode
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

        # Cosine projection
        v = v.float()
        u = u.float()
        p = F.cosine_similarity(u.unsqueeze(0), v.unsqueeze(0)).item()

        # Target sign: {0,1} -> {-1,+1}
        target_sign = 2 * t - 1

        # Check: sign matches AND absolute value exceeds margin
        sign_match = (p > 0 and target_sign == 1) or (p < 0 and target_sign == -1)
        passed = sign_match and abs(p) > margin

        return VerifyResult(
            passed=passed,
            projection=p,
            target_sign=target_sign,
            margin=margin,
        )
```

**Step 4: Run test to verify it passes**

Run: `conda run -n WFCLLM pytest tests/watermark/test_verifier.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add wfcllm/watermark/verifier.py tests/watermark/test_verifier.py
git commit -m "feat(watermark): add ProjectionVerifier for semantic projection check"
```

---

### Task 5: KVCacheManager — snapshot and rollback

**Files:**
- Create: `wfcllm/watermark/kv_cache.py`
- Test: `tests/watermark/test_kv_cache.py`

**Context:** HuggingFace CausalLM returns `past_key_values` as a tuple of tuples: `((K0, V0), (K1, V1), ...)` where each Ki/Vi has shape `(batch, num_heads, seq_len, head_dim)`. Snapshot records the seq_len. Rollback truncates to saved seq_len — no deep copy needed.

**Step 1: Write the failing test**

Create `tests/watermark/test_kv_cache.py`:

```python
"""Tests for wfcllm.watermark.kv_cache."""

import torch
import pytest
from wfcllm.watermark.kv_cache import KVCacheManager, CacheSnapshot


class TestCacheSnapshot:
    def test_snapshot_stores_seq_len(self):
        snap = CacheSnapshot(seq_len=42)
        assert snap.seq_len == 42


class TestKVCacheManager:
    @pytest.fixture
    def manager(self):
        return KVCacheManager()

    def _make_kv_cache(self, num_layers: int, seq_len: int) -> tuple:
        """Create a mock past_key_values structure."""
        batch, heads, head_dim = 1, 4, 32
        return tuple(
            (
                torch.randn(batch, heads, seq_len, head_dim),
                torch.randn(batch, heads, seq_len, head_dim),
            )
            for _ in range(num_layers)
        )

    def test_snapshot_records_seq_len(self, manager):
        kv = self._make_kv_cache(num_layers=2, seq_len=50)
        snap = manager.snapshot(kv)
        assert snap.seq_len == 50

    def test_rollback_truncates(self, manager):
        # Create cache with 50 tokens, snapshot at 30, grow to 50, rollback
        kv_30 = self._make_kv_cache(num_layers=2, seq_len=30)
        snap = manager.snapshot(kv_30)

        kv_50 = self._make_kv_cache(num_layers=2, seq_len=50)
        rolled = manager.rollback(kv_50, snap)

        for k, v in rolled:
            assert k.shape[2] == 30
            assert v.shape[2] == 30

    def test_rollback_preserves_values(self, manager):
        kv = self._make_kv_cache(num_layers=2, seq_len=50)
        snap = manager.snapshot(kv)

        # "Grow" the cache by extending (simulate more tokens generated)
        extended = tuple(
            (
                torch.cat([k, torch.randn_like(k[:, :, :10, :])], dim=2),
                torch.cat([v, torch.randn_like(v[:, :, :10, :])], dim=2),
            )
            for k, v in kv
        )
        # extended has seq_len=60, snap was at 50
        rolled = manager.rollback(extended, snap)
        for (orig_k, orig_v), (roll_k, roll_v) in zip(kv, rolled):
            assert torch.allclose(orig_k, roll_k)
            assert torch.allclose(orig_v, roll_v)

    def test_rollback_structure_matches(self, manager):
        num_layers = 4
        kv = self._make_kv_cache(num_layers=num_layers, seq_len=20)
        snap = manager.snapshot(kv)
        rolled = manager.rollback(kv, snap)
        assert len(rolled) == num_layers
        assert all(len(layer) == 2 for layer in rolled)

    def test_snapshot_different_sizes(self, manager):
        kv10 = self._make_kv_cache(num_layers=2, seq_len=10)
        kv100 = self._make_kv_cache(num_layers=2, seq_len=100)
        snap10 = manager.snapshot(kv10)
        snap100 = manager.snapshot(kv100)
        assert snap10.seq_len == 10
        assert snap100.seq_len == 100
```

**Step 2: Run test to verify it fails**

Run: `conda run -n WFCLLM pytest tests/watermark/test_kv_cache.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'wfcllm.watermark.kv_cache'`

**Step 3: Write minimal implementation**

Create `wfcllm/watermark/kv_cache.py`:

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
        # All layers have same seq_len; read from first layer's key tensor
        seq_len = past_key_values[0][0].shape[2]
        return CacheSnapshot(seq_len=seq_len)

    def rollback(
        self, past_key_values: tuple, snapshot: CacheSnapshot
    ) -> tuple:
        """Truncate KV-Cache to the snapshot's sequence length.

        Returns a new tuple of truncated (key, value) pairs.
        """
        target_len = snapshot.seq_len
        return tuple(
            (k[:, :, :target_len, :], v[:, :, :target_len, :])
            for k, v in past_key_values
        )
```

**Step 4: Run test to verify it passes**

Run: `conda run -n WFCLLM pytest tests/watermark/test_kv_cache.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add wfcllm/watermark/kv_cache.py tests/watermark/test_kv_cache.py
git commit -m "feat(watermark): add KVCacheManager for snapshot and rollback"
```

---

### Task 6: StatementInterceptor — incremental AST parsing

**Files:**
- Create: `wfcllm/watermark/interceptor.py`
- Test: `tests/watermark/test_interceptor.py`

**Context:** The interceptor accumulates decoded tokens and uses Tree-sitter to detect when a complete, error-free statement block closes. It compares the AST state before and after each new token. It uses `SIMPLE_STATEMENT_TYPES` and `COMPOUND_STATEMENT_TYPES` from `wfcllm/common/ast_parser.py`. Uses Tree-sitter incremental parsing for efficiency.

**Step 1: Write the failing test**

Create `tests/watermark/test_interceptor.py`:

```python
"""Tests for wfcllm.watermark.interceptor."""

import pytest
from wfcllm.watermark.interceptor import StatementInterceptor, InterceptEvent


class TestInterceptEvent:
    def test_event_fields(self):
        e = InterceptEvent(
            block_text="x = 1",
            block_type="simple",
            node_type="expression_statement",
            parent_node_type="module",
            token_start_idx=0,
            token_count=3,
        )
        assert e.block_text == "x = 1"
        assert e.block_type == "simple"


class TestStatementInterceptor:
    @pytest.fixture
    def interceptor(self):
        return StatementInterceptor()

    def test_no_event_on_partial_tokens(self, interceptor):
        """Feeding partial statement shouldn't trigger."""
        assert interceptor.feed_token("x") is None
        assert interceptor.feed_token(" ") is None
        assert interceptor.feed_token("=") is None
        assert interceptor.feed_token(" ") is None

    def test_simple_assignment_triggers(self, interceptor):
        """A complete 'x = 1\\n' should trigger a simple block event."""
        tokens = ["x", " ", "=", " ", "1", "\n"]
        events = []
        for tok in tokens:
            event = interceptor.feed_token(tok)
            if event is not None:
                events.append(event)
        assert len(events) >= 1
        assert events[0].block_type == "simple"
        assert "x" in events[0].block_text and "1" in events[0].block_text

    def test_multiple_statements(self, interceptor):
        """Two complete statements should trigger two events."""
        code = "x = 1\ny = 2\n"
        events = []
        for ch in code:
            event = interceptor.feed_token(ch)
            if event is not None:
                events.append(event)
        assert len(events) >= 2

    def test_compound_statement_triggers(self, interceptor):
        """A complete for loop should trigger compound event."""
        code = "for i in range(10):\n    x = i\n"
        events = []
        for ch in code:
            event = interceptor.feed_token(ch)
            if event is not None:
                events.append(event)
        # Should have at least a simple event (x = i) and possibly compound
        simple_events = [e for e in events if e.block_type == "simple"]
        assert len(simple_events) >= 1

    def test_event_has_parent_node_type(self, interceptor):
        """Events should include parent node type for topology hashing."""
        code = "x = 1\n"
        events = []
        for ch in code:
            event = interceptor.feed_token(ch)
            if event is not None:
                events.append(event)
        assert len(events) >= 1
        assert events[0].parent_node_type is not None

    def test_reset_clears_state(self, interceptor):
        """After reset, interceptor should start fresh."""
        interceptor.feed_token("x")
        interceptor.feed_token(" ")
        interceptor.reset()
        assert interceptor._accumulated == ""

    def test_syntax_error_no_false_trigger(self, interceptor):
        """Incomplete/malformed code should not trigger false positives."""
        tokens = ["def", " ", "foo", "("]
        events = []
        for tok in tokens:
            event = interceptor.feed_token(tok)
            if event is not None:
                events.append(event)
        assert len(events) == 0

    def test_token_tracking(self, interceptor):
        """Events should track token indices."""
        code = "x = 1\n"
        events = []
        for i, ch in enumerate(code):
            event = interceptor.feed_token(ch)
            if event is not None:
                events.append(event)
        if events:
            assert events[0].token_count > 0
```

**Step 2: Run test to verify it fails**

Run: `conda run -n WFCLLM pytest tests/watermark/test_interceptor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'wfcllm.watermark.interceptor'`

**Step 3: Write minimal implementation**

Create `wfcllm/watermark/interceptor.py`:

```python
"""Incremental AST parsing for statement block interception."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from wfcllm.common.ast_parser import (
    COMPOUND_STATEMENT_TYPES,
    SIMPLE_STATEMENT_TYPES,
    PythonParser,
)


@dataclass
class InterceptEvent:
    """Emitted when a complete statement block is detected."""

    block_text: str
    block_type: Literal["simple", "compound"]
    node_type: str
    parent_node_type: str | None
    token_start_idx: int
    token_count: int


class StatementInterceptor:
    """Detect statement block closures by incremental Tree-sitter parsing."""

    def __init__(self):
        self._parser = PythonParser()
        self._accumulated = ""
        self._token_idx = 0
        self._prev_blocks: list[_BlockInfo] = []
        self._prev_tree = None

    def feed_token(self, token_text: str) -> InterceptEvent | None:
        """Feed a new token; return event if a new block completed."""
        self._accumulated += token_text
        self._token_idx += 1

        # Parse current accumulated text
        tree = self._parser.parse(self._accumulated)
        current_blocks = self._extract_blocks(tree.root_node)

        event = self._detect_new_block(current_blocks)
        self._prev_blocks = current_blocks
        self._prev_tree = tree
        return event

    def reset(self):
        """Clear all accumulated state."""
        self._accumulated = ""
        self._token_idx = 0
        self._prev_blocks = []
        self._prev_tree = None

    def _extract_blocks(self, root) -> list[_BlockInfo]:
        """Walk AST and collect all error-free statement blocks."""
        blocks: list[_BlockInfo] = []
        self._walk(root, parent_type=None, blocks=blocks)
        return blocks

    def _walk(self, node, parent_type: str | None, blocks: list[_BlockInfo]):
        if node.type in SIMPLE_STATEMENT_TYPES | COMPOUND_STATEMENT_TYPES:
            if not node.has_error:
                text = self._accumulated[node.start_byte : node.end_byte]
                blocks.append(
                    _BlockInfo(
                        text=text,
                        node_type=node.type,
                        parent_type=parent_type or "module",
                        start_byte=node.start_byte,
                        end_byte=node.end_byte,
                        is_compound=node.type in COMPOUND_STATEMENT_TYPES,
                    )
                )
        for child in node.children:
            child_parent = node.type if node.type in COMPOUND_STATEMENT_TYPES | {"module"} else parent_type
            self._walk(child, child_parent, blocks)

    def _detect_new_block(self, current: list[_BlockInfo]) -> InterceptEvent | None:
        """Compare current blocks with previous, find newly completed ones."""
        prev_keys = {(b.node_type, b.start_byte, b.end_byte) for b in self._prev_blocks}

        for block in current:
            key = (block.node_type, block.start_byte, block.end_byte)
            if key not in prev_keys:
                return InterceptEvent(
                    block_text=block.text,
                    block_type="compound" if block.is_compound else "simple",
                    node_type=block.node_type,
                    parent_node_type=block.parent_type,
                    token_start_idx=max(0, self._token_idx - len(block.text)),
                    token_count=len(block.text),
                )
        return None


@dataclass
class _BlockInfo:
    """Internal representation of a detected block."""

    text: str
    node_type: str
    parent_type: str
    start_byte: int
    end_byte: int
    is_compound: bool
```

Note: `_accumulated[node.start_byte:node.end_byte]` works because Tree-sitter operates on byte offsets and we parsed from the string directly. The `PythonParser.parse()` encodes to UTF-8 internally, so byte offsets from tree-sitter correspond to the UTF-8 encoding. For the interceptor, we need to use the UTF-8 encoded bytes to extract text. The actual implementation should use:

```python
text = self._accumulated.encode("utf-8")[node.start_byte:node.end_byte].decode("utf-8")
```

**Step 4: Run test to verify it passes**

Run: `conda run -n WFCLLM pytest tests/watermark/test_interceptor.py -v`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add wfcllm/watermark/interceptor.py tests/watermark/test_interceptor.py
git commit -m "feat(watermark): add StatementInterceptor for block detection"
```

---

### Task 7: WatermarkGenerator — core generation loop

**Files:**
- Create: `wfcllm/watermark/generator.py`
- Test: `tests/watermark/test_generator.py`

**Context:** The generator is the orchestrator. It uses `model.forward()` for token-by-token generation, feeds each decoded token to the interceptor, and when a block closes: computes entropy, derives (v, t), verifies projection, and rolls back + retries if verification fails. This is the most complex module.

Note: The test for this module requires loading a real (small) model. Tests should use `Salesforce/codet5-small` or the smallest available CausalLM. Mark these tests with `@pytest.mark.slow` to allow skipping in fast CI.

**Step 1: Write the failing test**

Create `tests/watermark/test_generator.py`:

```python
"""Tests for wfcllm.watermark.generator."""

import pytest
import torch
from unittest.mock import MagicMock, patch
from wfcllm.watermark.generator import WatermarkGenerator, GenerateResult
from wfcllm.watermark.config import WatermarkConfig


class TestGenerateResult:
    def test_result_fields(self):
        r = GenerateResult(
            code="x = 1",
            total_blocks=1,
            embedded_blocks=1,
            failed_blocks=0,
            fallback_blocks=0,
        )
        assert r.code == "x = 1"
        assert r.total_blocks == 1


class TestWatermarkGeneratorUnit:
    """Unit tests with mock model — no GPU required."""

    @pytest.fixture
    def config(self):
        return WatermarkConfig(
            secret_key="test-key",
            max_new_tokens=50,
            encoder_device="cpu",
        )

    @pytest.fixture
    def mock_components(self):
        """Create mock model, tokenizer, encoder."""
        model = MagicMock()
        tokenizer = MagicMock()
        encoder = MagicMock()
        encoder_tokenizer = MagicMock()

        # Mock tokenizer encode
        tokenizer.encode = MagicMock(return_value=[1, 2, 3])
        tokenizer.decode = MagicMock(side_effect=lambda ids, **kw: "x = 1\n")
        tokenizer.eos_token_id = 2

        return model, tokenizer, encoder, encoder_tokenizer

    def test_generator_init(self, config, mock_components):
        model, tokenizer, encoder, enc_tok = mock_components
        gen = WatermarkGenerator(
            model=model,
            tokenizer=tokenizer,
            encoder=encoder,
            encoder_tokenizer=enc_tok,
            config=config,
        )
        assert gen._config == config

    def test_generate_result_type(self, config, mock_components):
        """generate() should return GenerateResult."""
        model, tokenizer, encoder, enc_tok = mock_components

        # Mock model.forward to return logits and kv-cache, then EOS
        vocab_size = 100
        logits = torch.zeros(1, 1, vocab_size)
        logits[0, 0, tokenizer.eos_token_id] = 10.0  # Force EOS
        past_kv = tuple(
            (torch.randn(1, 4, 3, 32), torch.randn(1, 4, 3, 32))
            for _ in range(2)
        )
        mock_output = MagicMock()
        mock_output.logits = logits
        mock_output.past_key_values = past_kv
        model.return_value = mock_output

        tokenizer.encode = MagicMock(return_value=[1, 2, 3])
        tokenizer.decode = MagicMock(return_value="")

        gen = WatermarkGenerator(
            model=model,
            tokenizer=tokenizer,
            encoder=encoder,
            encoder_tokenizer=enc_tok,
            config=config,
        )
        result = gen.generate("Write a function")
        assert isinstance(result, GenerateResult)
        assert isinstance(result.code, str)
```

**Step 2: Run test to verify it fails**

Run: `conda run -n WFCLLM pytest tests/watermark/test_generator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'wfcllm.watermark.generator'`

**Step 3: Write minimal implementation**

Create `wfcllm/watermark/generator.py`:

```python
"""Watermark-embedded code generation using custom token-by-token loop."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from wfcllm.watermark.config import WatermarkConfig
from wfcllm.watermark.entropy import NodeEntropyEstimator
from wfcllm.watermark.interceptor import StatementInterceptor
from wfcllm.watermark.keying import WatermarkKeying
from wfcllm.watermark.kv_cache import KVCacheManager
from wfcllm.watermark.verifier import ProjectionVerifier


@dataclass
class GenerateResult:
    """Result of watermark-embedded generation."""

    code: str
    total_blocks: int
    embedded_blocks: int
    failed_blocks: int
    fallback_blocks: int


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

        self._interceptor = StatementInterceptor()
        self._entropy_est = NodeEntropyEstimator()
        self._keying = WatermarkKeying(config.secret_key, config.encoder_embed_dim)
        self._verifier = ProjectionVerifier(
            encoder, encoder_tokenizer, device=config.encoder_device
        )
        self._cache_mgr = KVCacheManager()

    def generate(self, prompt: str) -> GenerateResult:
        """Generate code with watermark embedding.

        Args:
            prompt: The input prompt for code generation.

        Returns:
            GenerateResult with generated code and embedding statistics.
        """
        device = next(self._model.parameters()).device

        # Tokenize prompt
        input_ids = torch.tensor(
            [self._tokenizer.encode(prompt)], dtype=torch.long, device=device
        )

        past_kv = None
        generated_ids: list[int] = []
        generated_text = ""

        # Statistics
        total_blocks = 0
        embedded_blocks = 0
        failed_blocks = 0
        fallback_blocks = 0

        # Pending fallback regions
        pending_fallbacks: list[str] = []

        self._interceptor.reset()
        eos_id = self._config.eos_token_id or self._tokenizer.eos_token_id

        for _ in range(self._config.max_new_tokens):
            # Forward pass
            output = self._model(
                input_ids=input_ids,
                past_key_values=past_kv,
                use_cache=True,
            )
            logits = output.logits[:, -1, :]
            past_kv = output.past_key_values

            # Sample next token
            next_id = self._sample_token(logits)

            if next_id == eos_id:
                break

            generated_ids.append(next_id)
            token_text = self._tokenizer.decode([next_id], skip_special_tokens=True)
            generated_text += token_text

            # Feed to interceptor
            event = self._interceptor.feed_token(token_text)

            if event is not None and event.block_type == "simple":
                total_blocks += 1

                # Compute entropy and margin
                block_entropy = self._entropy_est.estimate_block_entropy(
                    event.block_text
                )
                margin = self._entropy_est.compute_margin(
                    block_entropy, self._config
                )

                # Derive watermark target
                v, t = self._keying.derive(
                    event.parent_node_type or "module", event.node_type
                )

                # Verify projection
                result = self._verifier.verify(event.block_text, v, t, margin)

                if result.passed:
                    embedded_blocks += 1
                else:
                    # Rejection sampling with KV-Cache rollback
                    snapshot = self._cache_mgr.snapshot(past_kv)
                    success = False

                    for _ in range(self._config.max_retries):
                        # Rollback
                        past_kv = self._cache_mgr.rollback(past_kv, snapshot)
                        # Remove generated tokens for this block
                        rollback_count = event.token_count
                        if rollback_count > 0 and rollback_count <= len(generated_ids):
                            generated_ids = generated_ids[:-rollback_count]
                            generated_text = self._tokenizer.decode(
                                generated_ids, skip_special_tokens=True
                            )

                        # Re-generate the block
                        regen_ids, regen_text, past_kv = self._regenerate_block(
                            past_kv, device, rollback_count
                        )
                        generated_ids.extend(regen_ids)
                        generated_text += regen_text

                        # Re-verify
                        result = self._verifier.verify(regen_text, v, t, margin)
                        if result.passed:
                            embedded_blocks += 1
                            success = True
                            break

                    if not success:
                        failed_blocks += 1
                        pending_fallbacks.append(event.block_text)

            elif event is not None and event.block_type == "compound":
                if self._config.enable_fallback and pending_fallbacks:
                    # Attempt macro-level fallback on compound block
                    total_blocks += 1
                    block_entropy = self._entropy_est.estimate_block_entropy(
                        event.block_text
                    )
                    margin = self._entropy_est.compute_margin(
                        block_entropy, self._config
                    )
                    v, t = self._keying.derive(
                        event.parent_node_type or "module", event.node_type
                    )
                    result = self._verifier.verify(event.block_text, v, t, margin)
                    if result.passed:
                        fallback_blocks += 1
                        pending_fallbacks.clear()

            # Prepare next input
            input_ids = torch.tensor([[next_id]], dtype=torch.long, device=device)

        return GenerateResult(
            code=generated_text,
            total_blocks=total_blocks,
            embedded_blocks=embedded_blocks,
            failed_blocks=failed_blocks,
            fallback_blocks=fallback_blocks,
        )

    def _sample_token(self, logits: torch.Tensor) -> int:
        """Sample a token from logits with temperature, top-k, top-p."""
        logits = logits.squeeze(0).float()

        # Temperature
        if self._config.temperature > 0:
            logits = logits / self._config.temperature

        # Top-k
        if self._config.top_k > 0:
            top_k = min(self._config.top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k).values[-1]
            logits[indices_to_remove] = float("-inf")

        # Top-p (nucleus)
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

    def _regenerate_block(
        self, past_kv, device, num_tokens: int
    ) -> tuple[list[int], str, tuple]:
        """Re-generate a fixed number of tokens from the rollback point."""
        regen_ids: list[int] = []
        # Use the last token in KV-cache context to start
        # We need to feed a dummy start token — use the token just before the block
        seq_len = past_kv[0][0].shape[2]
        # Generate from cache state
        input_ids = torch.tensor(
            [[self._tokenizer.eos_token_id or 0]], dtype=torch.long, device=device
        )

        for _ in range(num_tokens):
            output = self._model(
                input_ids=input_ids,
                past_key_values=past_kv,
                use_cache=True,
            )
            logits = output.logits[:, -1, :]
            past_kv = output.past_key_values
            next_id = self._sample_token(logits)
            regen_ids.append(next_id)
            input_ids = torch.tensor([[next_id]], dtype=torch.long, device=device)

        regen_text = self._tokenizer.decode(regen_ids, skip_special_tokens=True)
        return regen_ids, regen_text, past_kv
```

**Step 4: Run test to verify it passes**

Run: `conda run -n WFCLLM pytest tests/watermark/test_generator.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add wfcllm/watermark/generator.py tests/watermark/test_generator.py
git commit -m "feat(watermark): add WatermarkGenerator with core generation loop"
```

---

### Task 8: Update `__init__.py` and run full test suite

**Files:**
- Modify: `wfcllm/watermark/__init__.py`

**Step 1: Update public API exports**

Update `wfcllm/watermark/__init__.py`:

```python
"""Generation-time watermark embedding module."""

from wfcllm.watermark.config import WatermarkConfig
from wfcllm.watermark.entropy import NodeEntropyEstimator
from wfcllm.watermark.generator import GenerateResult, WatermarkGenerator
from wfcllm.watermark.interceptor import InterceptEvent, StatementInterceptor
from wfcllm.watermark.keying import WatermarkKeying
from wfcllm.watermark.kv_cache import CacheSnapshot, KVCacheManager
from wfcllm.watermark.verifier import ProjectionVerifier, VerifyResult

__all__ = [
    "WatermarkConfig",
    "NodeEntropyEstimator",
    "WatermarkGenerator",
    "GenerateResult",
    "StatementInterceptor",
    "InterceptEvent",
    "WatermarkKeying",
    "KVCacheManager",
    "CacheSnapshot",
    "ProjectionVerifier",
    "VerifyResult",
]
```

**Step 2: Run full watermark test suite**

Run: `conda run -n WFCLLM pytest tests/watermark/ -v`
Expected: All tests PASS

**Step 3: Run entire project test suite**

Run: `conda run -n WFCLLM pytest tests/ -v`
Expected: All tests PASS (no regressions)

**Step 4: Commit**

```bash
git add wfcllm/watermark/__init__.py
git commit -m "feat(watermark): complete Phase 2 module with public API exports"
```
