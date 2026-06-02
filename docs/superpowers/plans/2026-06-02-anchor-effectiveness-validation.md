# Anchor Effectiveness Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first-stage offline diagnostic harness that validates whether deterministic code anchors improve semantic LSH region diversity, valid-hit balance, and retry selection before any production AO-LSH generation changes are attempted.

**Architecture:** Keep production watermark behavior unchanged while adding reusable AO-LSH math under `wfcllm.watermark` and a diagnostic package under `wfcllm.evaluation.anchor_validation`. The diagnostic package builds candidate pools from repeated JSONL candidate artifacts, constructs prompt-free and prompt-aware anchors, computes M0-M8 region metrics, simulates retry selection, and writes stop/go summaries under `data/diagnostics/anchor_validation/`.

**Tech Stack:** Python 3.11, PyTorch tensor math, existing `wfcllm` parser/LSH/keying/evaluation modules, argparse scripts, JSONL artifacts, pytest, offline local encoder assets.

---

## Scope Check

This plan implements the spec's first execution gate only: R001 through R004, corresponding to execution milestones M0-M3 in `docs/experiment/2026-06-02-anchor-effectiveness-validation-plan.md`. It still computes partition methods M0-M8 inside those diagnostic milestones.

Do not implement end-to-end generation-time AO-LSH in this plan. The spec says "先不要直接改生产水印逻辑" and requires diagnostic stop/go evidence first. If R002-R004 pass, create a separate follow-up plan for production config, `ProjectionVerifier`, `SemanticChannel`, `BlockScorer`, extraction metadata, and end-to-end S0-S4 experiments.

Use @superpowers:test-driven-development for every task below and @superpowers:verification-before-completion before claiming completion.

## File Map

### Create

- `wfcllm/watermark/anchor_lsh.py`: Pure tensor helpers for anchored projection, signatures, margins, pairwise Hamming distance, and deterministic random anchors. This is shared by diagnostics and the later production AO-LSH plan.
- `wfcllm/evaluation/anchor_validation/__init__.py`: Package exports for validation dataclasses and runner entry points.
- `wfcllm/evaluation/anchor_validation/schema.py`: Frozen dataclasses for candidate contexts, candidate blocks, method IDs, metric rows, simulation rows, and stop/go summary rows. Candidate contexts include non-target source context so M3/M5/M6 actually test context anchors.
- `wfcllm/evaluation/anchor_validation/io.py`: UTF-8 JSONL load/write helpers for candidate pools, region metrics, selection simulation, and summaries.
- `wfcllm/evaluation/anchor_validation/candidate_generation.py`: Optional offline candidate generator for R001/R002 when full candidate-pool artifacts are unavailable. It samples K block-level candidates across temperatures from local prompts/reference or seed completions, writes explicit per-block candidate rows, and leaves production watermark generation untouched.
- `wfcllm/evaluation/anchor_validation/pool_builder.py`: Builds candidate-pool contexts from explicit per-block candidate rows, or conservatively from repeated whole-program JSONL rows only when a masked non-target context fingerprint matches. It must reject ambiguous whole-program grouping instead of silently grouping by ordinal alone.
- `wfcllm/evaluation/anchor_validation/anchors.py`: Builds slot/context/skeleton/prompt-aware anchor text and masks skeletons deterministically without using `secret_key`.
- `wfcllm/evaluation/anchor_validation/embedding.py`: Embedding provider protocol, deterministic hash provider for tests/smoke runs, and encoder-backed provider for real diagnostics.
- `wfcllm/evaluation/anchor_validation/metrics.py`: Entropy, collapse ratio, effective region count, Hamming diversity, valid-hit balance, bootstrap confidence intervals, and aggregation helpers.
- `wfcllm/evaluation/anchor_validation/selection.py`: Offline retry-budget simulation and z-proxy computation from candidate signatures.
- `wfcllm/evaluation/anchor_validation/summary.py`: R001-R004 aggregation and stop/go evidence, including paired deltas, random-anchor gap, SeqMark oracle gain ratio, low-entropy/node-type stratification, valid-hit balance, key-wise variance, retry B=4/B=8 improvement, fallback, and quality-proxy checks.
- `wfcllm/evaluation/anchor_validation/runner.py`: Orchestrates pool building, embedding, M0-M8 metric computation, multi-key valid-hit balance, retry simulation, and stop/go summary writing.
- `scripts/anchor_validation.py`: CLI for `generate-pool`, `build-pool`, and `run-diagnostics`.
- `tests/watermark/test_anchor_lsh.py`
- `tests/evaluation/anchor_validation/__init__.py`
- `tests/evaluation/anchor_validation/test_schema_io.py`
- `tests/evaluation/anchor_validation/test_candidate_generation.py`
- `tests/evaluation/anchor_validation/test_pool_builder.py`
- `tests/evaluation/anchor_validation/test_anchors.py`
- `tests/evaluation/anchor_validation/test_embedding.py`
- `tests/evaluation/anchor_validation/test_metrics.py`
- `tests/evaluation/anchor_validation/test_selection.py`
- `tests/evaluation/anchor_validation/test_summary.py`
- `tests/evaluation/anchor_validation/test_runner.py`
- `tests/integration/test_anchor_validation_cli.py`
- `docs/experiment/anchor-validation-runbook.md`

### Modify

- `wfcllm/watermark/lsh_space.py`: Add a read-only `planes` property and delegate existing `sign`/`min_margin` internals through reusable helpers where safe. Do not change existing return values.
- `scripts/evaluate.py`: Optional only if the implementation chooses to expose the new diagnostics as a subcommand. Prefer standalone `scripts/anchor_validation.py` first to avoid unnecessary CLI churn.

### Existing Files To Read Before Editing

- `CLAUDE.md`
- `AGENTS.md`
- `docs/experiment/2026-06-02-anchor-effectiveness-validation-plan.md`
- `docs/design/2026-06-02-casd-wfcllm-repair-plan.md`
- `wfcllm/watermark/lsh_space.py`
- `wfcllm/watermark/keying.py`
- `wfcllm/watermark/verifier.py`
- `wfcllm/lang/python/parser.py`
- `wfcllm/evaluation/benchmark.py`
- `scripts/evaluate.py`
- `tests/watermark/test_lsh_space.py`
- `tests/watermark/test_verifier.py`
- `tests/extract/test_scorer.py`

## Artifact Contracts

- Candidate pools are written to `data/diagnostics/anchor_validation/candidate_pools.jsonl`.
- Region metrics are written to `data/diagnostics/anchor_validation/region_metrics.jsonl`.
- Retry simulation rows are written to `data/diagnostics/anchor_validation/selection_simulation.jsonl`.
- Stop/go summaries are written to `data/diagnostics/anchor_validation/anchor_validation_summary.json`.
- These are diagnostic artifacts and should not be committed unless the user explicitly asks for tracked fixtures.
- Never include `secret_key` in diagnostic JSON/JSONL output. Store a `key_id` such as `key-00`, not the key string.

## Task 1: Add Reusable AO-LSH Tensor Utilities

**Files:**
- Create: `wfcllm/watermark/anchor_lsh.py`
- Modify: `wfcllm/watermark/lsh_space.py`
- Test: `tests/watermark/test_anchor_lsh.py`
- Test: `tests/watermark/test_lsh_space.py`

- [ ] **Step 1: Write failing AO-LSH utility tests**

```python
from __future__ import annotations

import pytest
import torch

from wfcllm.watermark.anchor_lsh import (
    hamming_distance,
    pairwise_hamming_diversity,
    project_planes_orthogonal,
    random_anchor,
    residual_signature,
    sign_with_planes,
)
from wfcllm.watermark.lsh_space import LSHSpace


def test_project_planes_orthogonal_removes_anchor_direction():
    planes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    anchor = torch.tensor([1.0, 0.0])

    projected = project_planes_orthogonal(planes, anchor)

    assert torch.allclose(projected[0], torch.zeros(2), atol=1e-6)
    assert torch.allclose(projected[1], torch.tensor([0.0, 1.0]), atol=1e-6)


def test_anchored_signature_is_invariant_to_removed_direction():
    planes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    anchor = torch.tensor([1.0, 0.0])
    projected = project_planes_orthogonal(planes, anchor)

    sig_a = sign_with_planes(torch.tensor([10.0, 1.0]), projected)
    sig_b = sign_with_planes(torch.tensor([-10.0, 1.0]), projected)

    assert sig_a == sig_b


def test_pairwise_hamming_diversity_normalizes_by_signature_width():
    signatures = [(0, 0), (0, 1), (1, 1)]
    assert pairwise_hamming_diversity(signatures) == pytest.approx(2 / 3)


def test_residual_signature_subtracts_seqmark_center():
    planes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    center = torch.tensor([10.0, 0.0])

    signature = residual_signature(
        torch.tensor([9.0, 1.0]),
        center=center,
        planes=planes,
    )

    assert signature == (0, 1)


def test_random_anchor_is_deterministic_and_unit_norm():
    a = random_anchor(secret_key="k", context_id="ctx-1", method="random", embed_dim=4)
    b = random_anchor(secret_key="k", context_id="ctx-1", method="random", embed_dim=4)
    c = random_anchor(secret_key="k", context_id="ctx-2", method="random", embed_dim=4)

    assert torch.allclose(a, b)
    assert not torch.allclose(a, c)
    assert torch.linalg.vector_norm(a).item() == pytest.approx(1.0)


def test_lsh_space_exposes_planes_as_defensive_copy():
    space = LSHSpace(secret_key="k", embed_dim=4, d=2)

    planes = space.planes
    planes[0, 0] = planes[0, 0] + 10.0

    assert not torch.allclose(planes, space.planes)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_anchor_lsh.py tests/watermark/test_lsh_space.py -v`

Expected: FAIL with missing `wfcllm.watermark.anchor_lsh` and missing `LSHSpace.planes`.

- [ ] **Step 3: Implement `wfcllm/watermark/anchor_lsh.py`**

```python
from __future__ import annotations

import hashlib
import hmac

import torch
import torch.nn.functional as F


def normalize_nonzero(vectors: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    norms = torch.linalg.vector_norm(vectors.float(), dim=dim, keepdim=True)
    normalized = vectors.float() / norms.clamp_min(eps)
    return torch.where(norms > eps, normalized, torch.zeros_like(normalized))


def project_planes_orthogonal(
    planes: torch.Tensor,
    anchor: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    anchor = anchor.float().flatten()
    planes = planes.float()
    denom = torch.dot(anchor, anchor) + eps
    if denom.item() <= eps:
        return normalize_nonzero(planes, dim=1, eps=eps)
    coeff = (planes @ anchor).unsqueeze(1) / denom
    projected = planes - coeff * anchor.unsqueeze(0)
    return normalize_nonzero(projected, dim=1, eps=eps)


def sign_with_planes(u: torch.Tensor, planes: torch.Tensor) -> tuple[int, ...]:
    u_norm = F.normalize(u.float().flatten().unsqueeze(0), dim=1)
    dots = (planes.float() @ u_norm.T).squeeze(1)
    return tuple((dots > 0).int().tolist())


def min_margin_with_planes(u: torch.Tensor, planes: torch.Tensor) -> float:
    u_norm = F.normalize(u.float().flatten().unsqueeze(0), dim=1)
    dots = (planes.float() @ u_norm.T).squeeze(1)
    return float(dots.abs().min().item())


def anchored_signature(
    u: torch.Tensor,
    planes: torch.Tensor,
    anchor: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[int, ...]:
    return sign_with_planes(u, project_planes_orthogonal(planes, anchor, eps=eps))


def residual_signature(
    u: torch.Tensor,
    center: torch.Tensor,
    planes: torch.Tensor,
) -> tuple[int, ...]:
    return sign_with_planes(u.float().flatten() - center.float().flatten(), planes)


def random_anchor(
    secret_key: str,
    context_id: str,
    method: str,
    embed_dim: int,
) -> torch.Tensor:
    key = secret_key.encode("utf-8")
    message = f"{method}:{context_id}".encode("utf-8")
    digest = hmac.new(key, message, hashlib.sha256).digest()
    seed = int.from_bytes(digest[:8], "big")
    gen = torch.Generator()
    gen.manual_seed(seed)
    return normalize_nonzero(torch.randn(embed_dim, generator=gen), dim=0)


def hamming_distance(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    if len(left) != len(right):
        raise ValueError("signatures must have the same width")
    return sum(1 for a, b in zip(left, right) if a != b)


def pairwise_hamming_diversity(signatures: list[tuple[int, ...]]) -> float:
    if len(signatures) < 2:
        return 0.0
    width = len(signatures[0])
    if width == 0:
        return 0.0
    total = 0.0
    pairs = 0
    for i, left in enumerate(signatures):
        for right in signatures[i + 1:]:
            total += hamming_distance(left, right) / width
            pairs += 1
    return total / pairs if pairs else 0.0
```

- [ ] **Step 4: Add a defensive `planes` property to `LSHSpace`**

```python
    @property
    def planes(self) -> torch.Tensor:
        """Return a defensive copy of the normalized LSH hyperplanes."""
        return self._planes.clone()
```

- [ ] **Step 5: Re-run AO-LSH and existing LSH tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_anchor_lsh.py tests/watermark/test_lsh_space.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add wfcllm/watermark/anchor_lsh.py wfcllm/watermark/lsh_space.py tests/watermark/test_anchor_lsh.py tests/watermark/test_lsh_space.py
git commit -m "feat: add anchored lsh utilities"
```

## Task 2: Add Anchor Validation Schemas and JSONL IO

**Files:**
- Create: `wfcllm/evaluation/anchor_validation/__init__.py`
- Create: `wfcllm/evaluation/anchor_validation/schema.py`
- Create: `wfcllm/evaluation/anchor_validation/io.py`
- Create: `tests/evaluation/anchor_validation/__init__.py`
- Test: `tests/evaluation/anchor_validation/test_schema_io.py`

- [ ] **Step 1: Write failing schema and IO tests**

```python
from __future__ import annotations

from wfcllm.evaluation.anchor_validation.io import (
    load_candidate_contexts,
    write_candidate_contexts,
)
from wfcllm.evaluation.anchor_validation.schema import CandidateBlock, CandidateContext


def test_candidate_context_jsonl_roundtrip(tmp_path):
    path = tmp_path / "candidate_pools.jsonl"
    context = CandidateContext(
        context_id="humaneval:0:1",
        dataset="humaneval",
        task_id="HumanEval/0",
        prompt="def f(x):\n",
        function_signature="def f(x):",
        ast_path=("function_definition", "return_statement"),
        node_type="return_statement",
        parent_node_type="function_definition",
        block_ordinal=1,
        context_hash="ctxhash",
        context_before="def f(x):\n",
        context_after="",
        masked_parent_context="def f(x):\n    <TARGET_BLOCK>",
        import_and_helper_signatures=("import math", "def helper(v):"),
        temperature=0.2,
        candidates=(
            CandidateBlock(candidate_id="c0", block_text="return x + 1", rank=0),
            CandidateBlock(candidate_id="c1", block_text="return 1 + x", rank=1),
        ),
    )

    write_candidate_contexts(path, [context])
    loaded = load_candidate_contexts(path)

    assert loaded == [context]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/evaluation/anchor_validation/test_schema_io.py -v`

Expected: FAIL with missing `wfcllm.evaluation.anchor_validation`.

- [ ] **Step 3: Implement frozen dataclasses in `schema.py`**

```python
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class AnchorMethod(StrEnum):
    VANILLA = "vanilla"
    RANDOM = "random"
    SLOT = "slot"
    CONTEXT = "context"
    SKELETON = "skeleton"
    SLOT_CONTEXT = "slot_context"
    SLOT_CONTEXT_SKELETON = "slot_context_skeleton"
    PROMPT_AWARE = "prompt_aware"
    SEQMARK_ORACLE = "seqmark_oracle"


@dataclass(frozen=True)
class CandidateBlock:
    candidate_id: str
    block_text: str
    rank: int
    syntax_valid: bool = True
    parse_valid: bool = True
    quality: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateContext:
    context_id: str
    dataset: str
    task_id: str
    prompt: str
    function_signature: str
    ast_path: tuple[str, ...]
    node_type: str
    parent_node_type: str
    block_ordinal: int
    context_hash: str
    temperature: float | None
    candidates: tuple[CandidateBlock, ...]
    context_before: str = ""
    context_after: str = ""
    masked_parent_context: str = ""
    import_and_helper_signatures: tuple[str, ...] = ()


@dataclass(frozen=True)
class RegionMetricRow:
    context_id: str
    dataset: str
    task_id: str
    method: str
    projection_key_id: str | None
    key_id: str | None
    gamma: float | None
    candidate_count: int
    normalized_entropy: float
    collapse_ratio: float
    effective_region_count: float
    hamming_diversity: float
    node_type: str | None = None
    valid_hit_rate: float | None = None
    gamma_deviation: float | None = None


@dataclass(frozen=True)
class SelectionSimulationRow:
    context_id: str
    method: str
    key_id: str
    gamma: float
    retry_budget: int
    selected_candidate_id: str
    selected_rank: int
    hit_acquired: bool
    fallback: bool
    z_proxy: float
    quality: dict[str, Any] = field(default_factory=dict)


def dataclass_to_jsonable(value: Any) -> dict[str, Any]:
    payload = asdict(value)
    if "ast_path" in payload:
        payload["ast_path"] = list(payload["ast_path"])
    return payload
```

- [ ] **Step 4: Implement JSONL helpers in `io.py`**

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, TypeVar

from wfcllm.evaluation.anchor_validation.schema import (
    CandidateBlock,
    CandidateContext,
    dataclass_to_jsonable,
)


T = TypeVar("T")


def write_jsonl(path: Path, rows: Iterable[object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            payload = row if isinstance(row, dict) else dataclass_to_jsonable(row)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_candidate_contexts(path: Path, contexts: Iterable[CandidateContext]) -> Path:
    return write_jsonl(path, contexts)


def load_candidate_contexts(path: Path) -> list[CandidateContext]:
    contexts: list[CandidateContext] = []
    for payload in read_jsonl(path):
        candidates = tuple(CandidateBlock(**item) for item in payload["candidates"])
        contexts.append(CandidateContext(
            context_id=str(payload["context_id"]),
            dataset=str(payload["dataset"]),
            task_id=str(payload["task_id"]),
            prompt=str(payload.get("prompt", "")),
            function_signature=str(payload.get("function_signature", "")),
            ast_path=tuple(str(part) for part in payload.get("ast_path", [])),
            node_type=str(payload["node_type"]),
            parent_node_type=str(payload.get("parent_node_type", "module")),
            block_ordinal=int(payload["block_ordinal"]),
            context_hash=str(payload.get("context_hash", "")),
            context_before=str(payload.get("context_before", "")),
            context_after=str(payload.get("context_after", "")),
            masked_parent_context=str(payload.get("masked_parent_context", "")),
            import_and_helper_signatures=tuple(
                str(part)
                for part in payload.get("import_and_helper_signatures", [])
            ),
            temperature=(
                float(payload["temperature"])
                if payload.get("temperature") is not None
                else None
            ),
            candidates=candidates,
        ))
    return contexts
```

- [ ] **Step 5: Re-run schema and IO tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/evaluation/anchor_validation/test_schema_io.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add wfcllm/evaluation/anchor_validation/__init__.py wfcllm/evaluation/anchor_validation/schema.py wfcllm/evaluation/anchor_validation/io.py tests/evaluation/anchor_validation/__init__.py tests/evaluation/anchor_validation/test_schema_io.py
git commit -m "feat: add anchor validation schemas"
```

## Task 3: Generate or Build Diagnostic Candidate Pools

**Files:**
- Create: `wfcllm/evaluation/anchor_validation/anchors.py` with `mask_code_skeleton` only; Task 4 extends this file with anchor text builders.
- Create: `wfcllm/evaluation/anchor_validation/candidate_generation.py`
- Create: `wfcllm/evaluation/anchor_validation/pool_builder.py`
- Test: `tests/evaluation/anchor_validation/test_candidate_generation.py`
- Test: `tests/evaluation/anchor_validation/test_pool_builder.py`

- [ ] **Step 1: Write failing candidate-generation tests**

```python
from __future__ import annotations

from wfcllm.evaluation.anchor_validation.candidate_generation import (
    GenerationContextSource,
    build_block_completion_prompt,
    extract_generation_contexts,
    generate_candidate_rows,
    replace_target_block,
)


def test_build_block_completion_prompt_contains_masked_context_not_secret():
    source = GenerationContextSource(
        dataset="humaneval",
        task_id="HumanEval/0",
        prompt="def f(x):\n",
        source_code="def f(x):\n    y = x + 1\n    return y\n",
    )
    context = extract_generation_contexts(source, max_contexts=1)[0]

    prompt = build_block_completion_prompt(context, secret_key="do-not-leak")

    assert "<TARGET_BLOCK>" in prompt
    assert "do-not-leak" not in prompt
    assert context.block_text not in prompt


def test_generate_candidate_rows_uses_temperature_sweep_and_k_budget():
    source = GenerationContextSource(
        dataset="humaneval",
        task_id="HumanEval/0",
        prompt="def f(x):\n",
        source_code="def f(x):\n    y = x + 1\n    return y\n",
    )

    def sampler(prompt: str, temperature: float, sample_index: int) -> str:
        return "y = 1 + x" if sample_index == 0 else "y = x + 1"

    rows = generate_candidate_rows(
        sources=(source,),
        sampler=sampler,
        temperatures=(0.2, 0.4),
        candidates_per_temperature=2,
        max_contexts_per_source=1,
    )

    assert len(rows) == 4
    assert {row["temperature"] for row in rows} == {0.2, 0.4}
    assert all(row["candidate_context_id"] for row in rows)
    assert all(row["context_hash"] for row in rows)


def test_generate_candidate_rows_filters_invalid_and_preserves_indentation():
    source = GenerationContextSource(
        dataset="humaneval",
        task_id="HumanEval/0",
        prompt="def f(x):\n",
        source_code="def f(x):\n    y = x + 1\n    return y\n",
    )

    def sampler(prompt: str, temperature: float, sample_index: int) -> str:
        return "not valid python:" if sample_index == 0 else "y = 1 + x"

    rows = generate_candidate_rows(
        sources=(source,),
        sampler=sampler,
        temperatures=(0.2,),
        candidates_per_temperature=2,
        max_contexts_per_source=1,
    )

    assert len(rows) == 1
    assert rows[0]["block_text"] == "    y = 1 + x"
    assert rows[0]["syntax_valid"] is True
    assert rows[0]["parse_valid"] is True
```

- [ ] **Step 2: Run candidate-generation tests to verify they fail**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/evaluation/anchor_validation/test_candidate_generation.py -v`

Expected: FAIL with missing `candidate_generation`.

- [ ] **Step 3: Implement `mask_code_skeleton` in `anchors.py`**

```python
from __future__ import annotations

import io
import keyword
import tokenize


def mask_code_skeleton(source: str) -> str:
    tokens: list[str] = []
    sentinel = "__WFCLLM_TARGET_BLOCK__"
    source = source.replace("<TARGET_BLOCK>", sentinel)
    stream = io.StringIO(source)
    try:
        generated = tokenize.generate_tokens(stream.readline)
        for token in generated:
            token_type = token.type
            token_text = token.string
            if token_type == tokenize.NAME and token_text == sentinel:
                tokens.append("<TARGET_BLOCK>")
            elif token_type == tokenize.NAME and not keyword.iskeyword(token_text):
                tokens.append("<NAME>")
            elif token_type == tokenize.NUMBER:
                tokens.append("<NUMBER>")
            elif token_type == tokenize.STRING:
                tokens.append("<STRING>")
            elif token_type in {tokenize.ENCODING, tokenize.ENDMARKER, tokenize.NL, tokenize.NEWLINE}:
                continue
            elif token_type in {tokenize.INDENT, tokenize.DEDENT}:
                continue
            else:
                tokens.append(token_text)
    except tokenize.TokenError:
        return source.strip()
    return " ".join(tokens).replace("( ", "(").replace(" )", ")").strip()
```

- [ ] **Step 4: Implement candidate-generation helpers**

```python
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Callable

from wfcllm.evaluation.anchor_validation.anchors import mask_code_skeleton
from wfcllm.lang.python.parser import StatementBlock, extract_statement_blocks


Sampler = Callable[[str, float, int], str]


@dataclass(frozen=True)
class GenerationContextSource:
    dataset: str
    task_id: str
    prompt: str
    source_code: str


@dataclass(frozen=True)
class BlockGenerationContext:
    candidate_context_id: str
    dataset: str
    task_id: str
    prompt: str
    source_code: str
    block_text: str
    context_hash: str
    context_before: str
    context_after: str
    masked_parent_context: str
    function_signature: str
    ast_path: tuple[str, ...]
    node_type: str
    parent_node_type: str
    block_ordinal: int
    import_and_helper_signatures: tuple[str, ...]


def extract_generation_contexts(
    source: GenerationContextSource,
    max_contexts: int | None = None,
) -> list[BlockGenerationContext]:
    blocks = extract_statement_blocks(source.source_code)
    block_by_id = {block.block_id: block for block in blocks}
    simple_blocks = [block for block in blocks if block.block_type == "simple"]
    contexts: list[BlockGenerationContext] = []
    for ordinal, block in enumerate(simple_blocks):
        parts = _context_parts(source.source_code, block, blocks, block_by_id)
        context_id = f"{source.dataset}:{source.task_id}:{ordinal}:{parts['context_hash'][:12]}"
        contexts.append(BlockGenerationContext(
            candidate_context_id=context_id,
            dataset=source.dataset,
            task_id=source.task_id,
            prompt=source.prompt,
            source_code=source.source_code,
            block_text=block.source,
            context_hash=parts["context_hash"],
            context_before=parts["context_before"],
            context_after=parts["context_after"],
            masked_parent_context=parts["masked_parent_context"],
            function_signature=_function_signature(source.prompt),
            ast_path=_ast_path(block, blocks),
            node_type=block.node_type,
            parent_node_type=_parent_node_type(block, block_by_id),
            block_ordinal=ordinal,
            import_and_helper_signatures=tuple(_import_and_helper_signatures(blocks)),
        ))
        if max_contexts is not None and len(contexts) >= max_contexts:
            break
    return contexts


def build_block_completion_prompt(
    context: BlockGenerationContext,
    secret_key: str | None = None,
) -> str:
    return (
        "Complete only the Python statement block that replaces <TARGET_BLOCK>. "
        "Return only the replacement block, without surrounding code.\n\n"
        f"{context.context_before}    <TARGET_BLOCK>\n{context.context_after}"
    )


def generate_candidate_rows(
    sources: tuple[GenerationContextSource, ...],
    sampler: Sampler,
    temperatures: tuple[float, ...] = (0.2, 0.4, 0.7),
    candidates_per_temperature: int = 16,
    max_contexts_per_source: int | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in sources:
        contexts = extract_generation_contexts(source, max_contexts=max_contexts_per_source)
        for context in contexts:
            completion_prompt = build_block_completion_prompt(context)
            for temperature in temperatures:
                for sample_index in range(candidates_per_temperature):
                    raw_candidate = sampler(completion_prompt, temperature, sample_index).strip("\n")
                    candidate = indent_candidate_block(context, raw_candidate)
                    full_code = replace_target_block(context, candidate)
                    syntax_valid, parse_valid = _candidate_is_parseable(full_code)
                    if not syntax_valid or not parse_valid:
                        continue
                    rows.append({
                        "candidate_context_id": context.candidate_context_id,
                        "candidate_id": f"{context.candidate_context_id}:{temperature}:{sample_index}",
                        "task_id": context.task_id,
                        "dataset": context.dataset,
                        "prompt": context.prompt,
                        "function_signature": context.function_signature,
                        "ast_path": list(context.ast_path),
                        "block_text": candidate,
                        "context_hash": context.context_hash,
                        "context_before": context.context_before,
                        "context_after": context.context_after,
                        "masked_parent_context": context.masked_parent_context,
                        "import_and_helper_signatures": list(context.import_and_helper_signatures),
                        "node_type": context.node_type,
                        "parent_node_type": context.parent_node_type,
                        "block_ordinal": context.block_ordinal,
                        "rank": sample_index,
                        "temperature": temperature,
                        "syntax_valid": syntax_valid,
                        "parse_valid": parse_valid,
                        "quality": {
                            "syntax_valid": syntax_valid,
                            "parse_valid": parse_valid,
                            "block_length": len(candidate),
                        },
                    })
    return rows


def replace_target_block(context: BlockGenerationContext, candidate_block: str) -> str:
    return context.context_before + candidate_block.rstrip() + "\n" + context.context_after


def indent_candidate_block(context: BlockGenerationContext, candidate_block: str) -> str:
    original_lines = context.block_text.splitlines()
    first_original = original_lines[0] if original_lines else ""
    indent = first_original[: len(first_original) - len(first_original.lstrip())]
    lines = candidate_block.splitlines() or [candidate_block]
    indented: list[str] = []
    for line in lines:
        if not line.strip():
            indented.append("")
        elif line.startswith(indent):
            indented.append(line)
        else:
            indented.append(indent + line.lstrip())
    return "\n".join(indented)


def build_hf_sampler(model, tokenizer, max_new_tokens: int = 64) -> Sampler:
    def sample(prompt: str, temperature: float, sample_index: int) -> str:
        import torch

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        output_ids = model.generate(
            input_ids=input_ids,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=getattr(tokenizer, "eos_token_id", None),
        )
        new_ids = output_ids[0, input_ids.shape[1]:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        return text.splitlines()[0] if text.splitlines() else text.strip()

    return sample


def _candidate_is_parseable(source_code: str) -> tuple[bool, bool]:
    try:
        compile(source_code, "<anchor-candidate>", "exec")
        syntax_valid = True
    except SyntaxError:
        syntax_valid = False
    try:
        extract_statement_blocks(source_code)
        parse_valid = True
    except Exception:
        parse_valid = False
    return syntax_valid, parse_valid
```

Implement `_context_parts`, `_parent_node_type`, `_function_signature`, `_ast_path`, and `_import_and_helper_signatures` by reusing the same helper logic from `pool_builder.py`; keep them private and deterministic.

- [ ] **Step 5: Write failing pool-builder tests**

```python
from __future__ import annotations

from wfcllm.evaluation.anchor_validation.pool_builder import build_candidate_contexts_from_records


def test_build_candidate_contexts_groups_repeated_task_blocks_by_ordinal():
    records = [
        {
            "id": "HumanEval/0",
            "dataset": "humaneval",
            "prompt": "def add_one(x):\n",
            "generated_code": "    y = x + 1\n    return y\n",
            "candidate_index": 0,
            "temperature": 0.2,
        },
        {
            "id": "HumanEval/0",
            "dataset": "humaneval",
            "prompt": "def add_one(x):\n",
            "generated_code": "    y = 1 + x\n    return y\n",
            "candidate_index": 1,
            "temperature": 0.2,
        },
    ]

    contexts = build_candidate_contexts_from_records(records, min_candidates=2)

    assert contexts
    first = contexts[0]
    assert first.dataset == "humaneval"
    assert first.task_id == "HumanEval/0"
    assert first.function_signature == "def add_one(x):"
    assert len(first.candidates) == 2
    assert first.parent_node_type == "function_definition"
    assert first.context_hash
    assert "<TARGET_BLOCK>" in first.masked_parent_context
    assert "return y" in first.context_after
    assert "y = 1 + x" not in first.context_before
    assert "y = 1 + x" not in first.context_after
    assert "y = 1 + x" not in first.masked_parent_context
```

Add this ambiguity guard test in the same file:

```python
def test_build_candidate_contexts_does_not_mix_changed_surrounding_contexts():
    records = [
        {
            "id": "HumanEval/0",
            "dataset": "humaneval",
            "prompt": "def add_one(x):\n",
            "generated_code": "    y = x + 1\n    return y\n",
            "candidate_index": 0,
        },
        {
            "id": "HumanEval/0",
            "dataset": "humaneval",
            "prompt": "def add_one(x):\n",
            "generated_code": "    debug = x\n    y = x + 1\n    return y\n",
            "candidate_index": 1,
        },
    ]

    assert build_candidate_contexts_from_records(records, min_candidates=2) == []
```

- [ ] **Step 6: Run test to verify it fails**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/evaluation/anchor_validation/test_pool_builder.py -v`

Expected: FAIL with missing `pool_builder`.

- [ ] **Step 7: Implement artifact-backed pool construction**

```python
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from typing import Any

from wfcllm.evaluation.anchor_validation.anchors import mask_code_skeleton
from wfcllm.evaluation.anchor_validation.schema import CandidateBlock, CandidateContext
from wfcllm.lang.python.parser import StatementBlock, extract_statement_blocks


def build_candidate_contexts_from_records(
    records: list[dict[str, Any]],
    min_candidates: int = 2,
    max_contexts_per_task: int | None = None,
) -> list[CandidateContext]:
    explicit_contexts = _build_explicit_candidate_contexts(records, min_candidates)
    if explicit_contexts:
        return explicit_contexts

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        task_id = str(record.get("id", ""))
        if task_id:
            grouped[task_id].append(record)

    contexts: list[CandidateContext] = []
    for task_id, task_records in sorted(grouped.items()):
        per_context: dict[str, list[CandidateBlock]] = defaultdict(list)
        context_examples: dict[str, CandidateContext] = {}

        for rank, record in enumerate(task_records):
            prompt = str(record.get("prompt", ""))
            generated = str(record.get("generated_code", ""))
            full_code = prompt + generated if prompt else generated
            try:
                blocks = extract_statement_blocks(full_code)
            except Exception:
                continue
            block_by_id = {block.block_id: block for block in blocks}
            simple_blocks = [block for block in blocks if block.block_type == "simple"]
            for ordinal, block in enumerate(simple_blocks):
                parent_type = _parent_node_type(block, block_by_id)
                context_parts = _context_parts(full_code, block, blocks, block_by_id)
                context_hash = context_parts["context_hash"]
                source_hash = sha256(block.source.encode("utf-8")).hexdigest()[:12]
                candidate = CandidateBlock(
                    candidate_id=f"{task_id}:{ordinal}:{rank}:{source_hash}",
                    block_text=block.source,
                    rank=int(record.get("candidate_index", rank)),
                    syntax_valid=True,
                    parse_valid=True,
                    quality={
                        "source_hash": source_hash,
                        "generated_length": len(generated),
                    },
                )
                per_context[context_hash].append(candidate)
                if context_hash not in context_examples:
                    context_examples[context_hash] = CandidateContext(
                        context_id=f"{record.get('dataset', 'unknown')}:{task_id}:{ordinal}:{context_hash[:12]}",
                        dataset=str(record.get("dataset", "unknown")),
                        task_id=task_id,
                        prompt=prompt,
                        function_signature=_function_signature(prompt),
                        ast_path=_ast_path(block, blocks),
                        node_type=block.node_type,
                        parent_node_type=parent_type,
                        block_ordinal=ordinal,
                        context_hash=context_hash,
                        temperature=(
                            float(record["temperature"])
                            if record.get("temperature") is not None
                            else None
                        ),
                        candidates=(),
                        context_before=context_parts["context_before"],
                        context_after=context_parts["context_after"],
                        masked_parent_context=context_parts["masked_parent_context"],
                        import_and_helper_signatures=tuple(_import_and_helper_signatures(blocks)),
                    )

        task_contexts = 0
        for context_hash, candidates in sorted(per_context.items()):
            unique = _dedupe_candidates(candidates)
            if len(unique) < min_candidates:
                continue
            base = context_examples[context_hash]
            contexts.append(CandidateContext(
                context_id=base.context_id,
                dataset=base.dataset,
                task_id=base.task_id,
                prompt=base.prompt,
                function_signature=base.function_signature,
                ast_path=base.ast_path,
                node_type=base.node_type,
                parent_node_type=base.parent_node_type,
                block_ordinal=base.block_ordinal,
                context_hash=base.context_hash,
                temperature=base.temperature,
                candidates=tuple(unique),
                context_before=base.context_before,
                context_after=base.context_after,
                masked_parent_context=base.masked_parent_context,
                import_and_helper_signatures=base.import_and_helper_signatures,
            ))
            task_contexts += 1
            if max_contexts_per_task is not None and task_contexts >= max_contexts_per_task:
                break
    return contexts


def _build_explicit_candidate_contexts(
    records: list[dict[str, Any]],
    min_candidates: int,
) -> list[CandidateContext]:
    if not records or not all("candidate_context_id" in record for record in records):
        return []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["candidate_context_id"])].append(record)

    contexts: list[CandidateContext] = []
    for context_id, rows in sorted(grouped.items()):
        if len(rows) < min_candidates:
            continue
        first = rows[0]
        required = ("block_text", "context_hash", "node_type", "parent_node_type", "block_ordinal")
        missing = [field for field in required if field not in first]
        if missing:
            raise ValueError(f"explicit candidate context {context_id} missing fields: {missing}")
        candidates = tuple(
            CandidateBlock(
                candidate_id=str(row.get("candidate_id", f"{context_id}:{idx}")),
                block_text=str(row["block_text"]),
                rank=int(row.get("rank", row.get("candidate_index", idx))),
                syntax_valid=bool(row.get("syntax_valid", True)),
                parse_valid=bool(row.get("parse_valid", True)),
                quality=dict(row.get("quality", {}) or {}),
            )
            for idx, row in enumerate(rows)
        )
        contexts.append(CandidateContext(
            context_id=context_id,
            dataset=str(first.get("dataset", "unknown")),
            task_id=str(first.get("task_id", first.get("id", ""))),
            prompt=str(first.get("prompt", "")),
            function_signature=str(first.get("function_signature", "")),
            ast_path=tuple(str(part) for part in first.get("ast_path", [])),
            node_type=str(first["node_type"]),
            parent_node_type=str(first["parent_node_type"]),
            block_ordinal=int(first["block_ordinal"]),
            context_hash=str(first["context_hash"]),
            temperature=(
                float(first["temperature"])
                if first.get("temperature") is not None
                else None
            ),
            candidates=candidates,
            context_before=str(first.get("context_before", "")),
            context_after=str(first.get("context_after", "")),
            masked_parent_context=str(first.get("masked_parent_context", "")),
            import_and_helper_signatures=tuple(
                str(part)
                for part in first.get("import_and_helper_signatures", [])
            ),
        ))
    return contexts


def _dedupe_candidates(candidates: list[CandidateBlock]) -> list[CandidateBlock]:
    seen: set[str] = set()
    unique: list[CandidateBlock] = []
    for candidate in sorted(candidates, key=lambda item: item.rank):
        if candidate.block_text in seen:
            continue
        seen.add(candidate.block_text)
        unique.append(candidate)
    return unique


def _parent_node_type(block: StatementBlock, block_by_id: dict[str, StatementBlock]) -> str:
    if block.parent_id is None:
        return "module"
    return block_by_id[block.parent_id].node_type


def _context_parts(
    full_code: str,
    block: StatementBlock,
    blocks: list[StatementBlock],
    block_by_id: dict[str, StatementBlock],
) -> dict[str, str]:
    lines = full_code.splitlines(keepends=True)
    start = max(block.start_line - 1, 0)
    end = max(block.end_line, start + 1)
    context_before = "".join(lines[:start])
    context_after = "".join(lines[end:])
    masked_full_context = context_before + "    <TARGET_BLOCK>\n" + context_after

    parent = block_by_id.get(block.parent_id) if block.parent_id is not None else None
    if parent is not None:
        masked_parent_source = parent.source.replace(block.source, "<TARGET_BLOCK>", 1)
    else:
        masked_parent_source = masked_full_context

    context_fingerprint = "\n".join([
        str(block.start_line),
        block.node_type,
        _parent_node_type(block, block_by_id),
        mask_code_skeleton(masked_full_context),
        mask_code_skeleton(masked_parent_source),
        "\n".join(_import_and_helper_signatures(blocks)),
    ])
    return {
        "context_hash": sha256(context_fingerprint.encode("utf-8")).hexdigest(),
        "context_before": context_before,
        "context_after": context_after,
        "masked_parent_context": mask_code_skeleton(masked_parent_source),
    }


def _import_and_helper_signatures(blocks: list[StatementBlock]) -> list[str]:
    signatures: list[str] = []
    for block in blocks:
        if block.depth != 0:
            continue
        if block.node_type in {"import_statement", "import_from_statement"}:
            signatures.append(block.source.strip())
        elif block.node_type in {"function_definition", "class_definition"}:
            first_line = block.source.splitlines()[0].strip()
            signatures.append(first_line)
    return signatures


def _function_signature(prompt: str) -> str:
    for line in reversed(prompt.splitlines()):
        stripped = line.strip()
        if stripped.startswith("def "):
            return stripped
    return ""


def _ast_path(block: StatementBlock, blocks: list[StatementBlock]) -> tuple[str, ...]:
    by_id = {item.block_id: item for item in blocks}
    path = [block.node_type]
    parent_id = block.parent_id
    while parent_id is not None:
        parent = by_id[parent_id]
        path.append(parent.node_type)
        parent_id = parent.parent_id
    return tuple(reversed(path))
```

- [ ] **Step 8: Add malformed and insufficient-candidate tests**

```python
def test_build_candidate_contexts_accepts_explicit_per_block_rows():
    records = [
        {
            "candidate_context_id": "ctx-1",
            "candidate_id": "c0",
            "task_id": "HumanEval/0",
            "dataset": "humaneval",
            "block_text": "return x",
            "context_hash": "ctxhash",
            "context_before": "def f(x):\n",
            "context_after": "",
            "masked_parent_context": "def f(<NAME>):\n    <TARGET_BLOCK>",
            "node_type": "return_statement",
            "parent_node_type": "function_definition",
            "block_ordinal": 0,
            "rank": 0,
        },
        {
            "candidate_context_id": "ctx-1",
            "candidate_id": "c1",
            "task_id": "HumanEval/0",
            "dataset": "humaneval",
            "block_text": "return x + 1",
            "context_hash": "ctxhash",
            "context_before": "def f(x):\n",
            "context_after": "",
            "masked_parent_context": "def f(<NAME>):\n    <TARGET_BLOCK>",
            "node_type": "return_statement",
            "parent_node_type": "function_definition",
            "block_ordinal": 0,
            "rank": 1,
        },
    ]

    contexts = build_candidate_contexts_from_records(records, min_candidates=2)

    assert len(contexts) == 1
    assert contexts[0].context_id == "ctx-1"
    assert len(contexts[0].candidates) == 2


def test_build_candidate_contexts_skips_slots_below_min_candidates():
    records = [{
        "id": "HumanEval/0",
        "dataset": "humaneval",
        "prompt": "def f(x):\n",
        "generated_code": "    return x\n",
    }]

    assert build_candidate_contexts_from_records(records, min_candidates=2) == []
```

- [ ] **Step 9: Re-run candidate-generation and pool-builder tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/evaluation/anchor_validation/test_candidate_generation.py tests/evaluation/anchor_validation/test_pool_builder.py -v`

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add wfcllm/evaluation/anchor_validation/anchors.py wfcllm/evaluation/anchor_validation/candidate_generation.py wfcllm/evaluation/anchor_validation/pool_builder.py tests/evaluation/anchor_validation/test_candidate_generation.py tests/evaluation/anchor_validation/test_pool_builder.py
git commit -m "feat: build anchor candidate pools"
```

## Task 4: Add Anchor Material Builders and Embedding Providers

**Files:**
- Modify: `wfcllm/evaluation/anchor_validation/anchors.py`
- Create: `wfcllm/evaluation/anchor_validation/embedding.py`
- Test: `tests/evaluation/anchor_validation/test_anchors.py`
- Test: `tests/evaluation/anchor_validation/test_embedding.py`

- [ ] **Step 1: Write failing anchor tests**

```python
from __future__ import annotations

import pytest
import torch

from wfcllm.evaluation.anchor_validation.anchors import build_anchor_text, mask_code_skeleton
from wfcllm.evaluation.anchor_validation.embedding import (
    EmbeddingProvider,
    EncoderEmbeddingProvider,
    HashEmbeddingProvider,
)
from wfcllm.evaluation.anchor_validation.schema import AnchorMethod, CandidateBlock, CandidateContext


def _context() -> CandidateContext:
    return CandidateContext(
        context_id="ctx",
        dataset="humaneval",
        task_id="HumanEval/0",
        prompt="def f(x):\n",
        function_signature="def f(x):",
        ast_path=("function_definition", "return_statement"),
        node_type="return_statement",
        parent_node_type="function_definition",
        block_ordinal=0,
        context_hash="ctxhash",
        context_before="def f(x):\n",
        context_after="",
        masked_parent_context="def f(<NAME>):\n    <TARGET_BLOCK>",
        import_and_helper_signatures=("import math", "def helper(v):"),
        temperature=0.2,
        candidates=(CandidateBlock("c0", "return x + 1", 0),),
    )


def test_mask_code_skeleton_masks_identifiers_and_literals():
    assert mask_code_skeleton("total = x + 42") == "<NAME> = <NAME> + <NUMBER>"


def test_slot_context_anchor_is_prompt_free_and_does_not_include_secret_key():
    text = build_anchor_text(
        AnchorMethod.SLOT_CONTEXT,
        _context(),
        _context().candidates[0],
        secret_key="do-not-leak",
    )

    assert "do-not-leak" not in text
    assert "HumanEval/0" in text
    assert "return x + 1" not in text
    assert "def f(<NAME>):" in text


def test_context_anchor_uses_surrounding_context_not_only_signature():
    slot = build_anchor_text(
        AnchorMethod.SLOT,
        _context(),
        _context().candidates[0],
    )
    context = build_anchor_text(
        AnchorMethod.CONTEXT,
        _context(),
        _context().candidates[0],
    )

    assert context != slot
    assert "<TARGET_BLOCK>" in context


def test_hash_embedding_provider_is_deterministic_and_normalized():
    provider = HashEmbeddingProvider(embed_dim=8)

    left = provider.embed("slot|return")
    right = provider.embed("slot|return")

    assert torch.allclose(left, right)
    assert torch.linalg.vector_norm(left).item() == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/evaluation/anchor_validation/test_anchors.py tests/evaluation/anchor_validation/test_embedding.py -v`

Expected: FAIL with missing `build_anchor_text` and `embedding`.

- [ ] **Step 3: Extend `anchors.py` with deterministic anchor text builders**

Preserve the `mask_code_skeleton` helper created in Task 3. Add the imports and `build_anchor_text` below it.

```python
from __future__ import annotations

from wfcllm.evaluation.anchor_validation.schema import AnchorMethod, CandidateBlock, CandidateContext


def build_anchor_text(
    method: AnchorMethod,
    context: CandidateContext,
    candidate: CandidateBlock | None = None,
    secret_key: str | None = None,
) -> str:
    if method == AnchorMethod.VANILLA:
        return ""
    if method == AnchorMethod.RANDOM:
        return ""
    if method == AnchorMethod.SEQMARK_ORACLE:
        return ""

    parts: list[str] = []
    if method in {
        AnchorMethod.SLOT,
        AnchorMethod.SLOT_CONTEXT,
        AnchorMethod.SLOT_CONTEXT_SKELETON,
        AnchorMethod.PROMPT_AWARE,
    }:
        parts.extend([
            f"dataset={context.dataset}",
            f"task={context.task_id}",
            f"signature={context.function_signature}",
            f"ast_path={'/'.join(context.ast_path)}",
            f"node={context.node_type}",
            f"parent={context.parent_node_type}",
            f"ordinal={context.block_ordinal}",
        ])
    if method in {
        AnchorMethod.CONTEXT,
        AnchorMethod.SLOT_CONTEXT,
        AnchorMethod.SLOT_CONTEXT_SKELETON,
        AnchorMethod.PROMPT_AWARE,
    }:
        parts.extend([
            f"context_hash={context.context_hash}",
            f"context_before={mask_code_skeleton(context.context_before)}",
            f"context_after={mask_code_skeleton(context.context_after)}",
            f"masked_parent={context.masked_parent_context}",
            "imports_helpers=" + "|".join(context.import_and_helper_signatures),
        ])
    if method in {
        AnchorMethod.SKELETON,
        AnchorMethod.SLOT_CONTEXT_SKELETON,
    }:
        if candidate is None:
            raise ValueError("candidate is required for skeleton anchors")
        parts.append(f"skeleton={mask_code_skeleton(candidate.block_text)}")
    if method == AnchorMethod.PROMPT_AWARE:
        parts.append(f"prompt={context.prompt}")
    return "\n".join(parts)
```

- [ ] **Step 4: Implement embedding providers**

```python
from __future__ import annotations

import hashlib
from typing import Protocol

import torch
import torch.nn.functional as F


class EmbeddingProvider(Protocol):
    embed_dim: int

    def embed(self, text: str) -> torch.Tensor:
        ...


class HashEmbeddingProvider:
    def __init__(self, embed_dim: int = 128) -> None:
        self.embed_dim = embed_dim

    def embed(self, text: str) -> torch.Tensor:
        values: list[float] = []
        counter = 0
        while len(values) < self.embed_dim:
            digest = hashlib.sha256(f"{counter}:{text}".encode("utf-8")).digest()
            for byte in digest:
                values.append((byte / 127.5) - 1.0)
                if len(values) == self.embed_dim:
                    break
            counter += 1
        vector = torch.tensor(values, dtype=torch.float32)
        return F.normalize(vector.unsqueeze(0), dim=1).squeeze(0)


class EncoderEmbeddingProvider:
    def __init__(self, encoder, tokenizer, device: str = "cpu", max_length: int = 256) -> None:
        self._encoder = encoder
        self._tokenizer = tokenizer
        self._device = device
        self._max_length = max_length
        self.embed_dim = int(getattr(encoder, "embed_dim", 128))

    @torch.no_grad()
    def embed(self, text: str) -> torch.Tensor:
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length,
        )
        input_ids = inputs["input_ids"].to(self._device)
        attention_mask = inputs["attention_mask"].to(self._device)
        return self._encoder(input_ids, attention_mask).squeeze(0).detach().cpu()
```

- [ ] **Step 5: Re-run anchor and embedding tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/evaluation/anchor_validation/test_anchors.py tests/evaluation/anchor_validation/test_embedding.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add wfcllm/evaluation/anchor_validation/anchors.py wfcllm/evaluation/anchor_validation/embedding.py tests/evaluation/anchor_validation/test_anchors.py tests/evaluation/anchor_validation/test_embedding.py
git commit -m "feat: add anchor builders"
```

## Task 5: Compute Region Metrics and Valid-Hit Balance

**Files:**
- Create: `wfcllm/evaluation/anchor_validation/metrics.py`
- Test: `tests/evaluation/anchor_validation/test_metrics.py`

- [ ] **Step 1: Write failing metric tests**

```python
from __future__ import annotations

import pytest

from wfcllm.evaluation.anchor_validation.metrics import (
    effective_region_count,
    normalized_region_entropy,
    summarize_signature_metrics,
    valid_hit_balance,
)


def test_normalized_region_entropy_is_zero_for_full_collapse():
    signatures = [(0, 0), (0, 0), (0, 0)]
    assert normalized_region_entropy(signatures, region_count=4) == 0.0


def test_normalized_region_entropy_is_one_for_even_two_region_split():
    signatures = [(0,), (1,)]
    assert normalized_region_entropy(signatures, region_count=2) == pytest.approx(1.0)


def test_effective_region_count_matches_exp_entropy():
    signatures = [(0,), (1,)]
    assert effective_region_count(signatures) == pytest.approx(2.0)


def test_valid_hit_balance_reports_gamma_deviation():
    signatures = [(0,), (1,), (1,), (1,)]
    balance = valid_hit_balance(signatures, valid_set=frozenset({(1,)}), gamma=0.5)

    assert balance.hit_rate == pytest.approx(0.75)
    assert balance.gamma_deviation == pytest.approx(0.25)


def test_summarize_signature_metrics_includes_collapse_ratio():
    row = summarize_signature_metrics(
        context_id="ctx",
        dataset="humaneval",
        task_id="HumanEval/0",
        method="vanilla",
        signatures=[(0,), (1,)],
        region_count=2,
        projection_key_id="proj-00",
        key_id=None,
        gamma=None,
        valid_set=None,
    )

    assert row.normalized_entropy == pytest.approx(1.0)
    assert row.collapse_ratio == pytest.approx(0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/evaluation/anchor_validation/test_metrics.py -v`

Expected: FAIL with missing `metrics`.

- [ ] **Step 3: Implement entropy, diversity, and balance helpers**

```python
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from wfcllm.evaluation.anchor_validation.schema import RegionMetricRow
from wfcllm.watermark.anchor_lsh import pairwise_hamming_diversity


@dataclass(frozen=True)
class ValidHitBalance:
    hit_rate: float
    gamma_deviation: float


def region_entropy(signatures: list[tuple[int, ...]]) -> float:
    if not signatures:
        return 0.0
    counts = Counter(signatures)
    total = len(signatures)
    return -sum((count / total) * math.log(count / total) for count in counts.values())


def normalized_region_entropy(signatures: list[tuple[int, ...]], region_count: int) -> float:
    if not signatures:
        return 0.0
    denom = math.log(min(len(signatures), region_count))
    if denom <= 0:
        return 0.0
    return region_entropy(signatures) / denom


def effective_region_count(signatures: list[tuple[int, ...]]) -> float:
    return math.exp(region_entropy(signatures)) if signatures else 0.0


def valid_hit_balance(
    signatures: list[tuple[int, ...]],
    valid_set: frozenset[tuple[int, ...]],
    gamma: float,
) -> ValidHitBalance:
    if not signatures:
        return ValidHitBalance(hit_rate=0.0, gamma_deviation=abs(gamma))
    hits = sum(1 for sig in signatures if sig in valid_set)
    hit_rate = hits / len(signatures)
    return ValidHitBalance(hit_rate=hit_rate, gamma_deviation=abs(hit_rate - gamma))


def summarize_signature_metrics(
    context_id: str,
    dataset: str,
    task_id: str,
    method: str,
    signatures: list[tuple[int, ...]],
    region_count: int,
    projection_key_id: str | None,
    key_id: str | None,
    gamma: float | None,
    valid_set: frozenset[tuple[int, ...]] | None,
    node_type: str | None = None,
) -> RegionMetricRow:
    normalized_entropy = normalized_region_entropy(signatures, region_count)
    balance = (
        valid_hit_balance(signatures, valid_set, gamma)
        if valid_set is not None and gamma is not None
        else None
    )
    return RegionMetricRow(
        context_id=context_id,
        dataset=dataset,
        task_id=task_id,
        method=method,
        projection_key_id=projection_key_id,
        key_id=key_id,
        gamma=gamma,
        candidate_count=len(signatures),
        normalized_entropy=normalized_entropy,
        collapse_ratio=1.0 - normalized_entropy,
        effective_region_count=effective_region_count(signatures),
        hamming_diversity=pairwise_hamming_diversity(signatures),
        node_type=node_type,
        valid_hit_rate=balance.hit_rate if balance else None,
        gamma_deviation=balance.gamma_deviation if balance else None,
    )
```

- [ ] **Step 4: Add bootstrap CI helper tests**

```python
def test_bootstrap_mean_ci_is_deterministic():
    from wfcllm.evaluation.anchor_validation.metrics import bootstrap_mean_ci

    lo, hi = bootstrap_mean_ci([0.1, 0.2, 0.3], iterations=100, seed=7)

    assert 0.1 <= lo <= hi <= 0.3
```

Implementation:

```python
def bootstrap_mean_ci(
    values: list[float],
    iterations: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    import random
    rng = random.Random(seed)
    means: list[float] = []
    n = len(values)
    for _ in range(iterations):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = max(0, int((alpha / 2) * iterations))
    hi_idx = min(iterations - 1, int((1 - alpha / 2) * iterations))
    return means[lo_idx], means[hi_idx]
```

- [ ] **Step 5: Re-run metric tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/evaluation/anchor_validation/test_metrics.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add wfcllm/evaluation/anchor_validation/metrics.py tests/evaluation/anchor_validation/test_metrics.py
git commit -m "feat: add anchor region metrics"
```

## Task 6: Simulate Offline Retry Selection

**Files:**
- Create: `wfcllm/evaluation/anchor_validation/selection.py`
- Test: `tests/evaluation/anchor_validation/test_selection.py`

- [ ] **Step 1: Write failing selection tests**

```python
from __future__ import annotations

from wfcllm.evaluation.anchor_validation.schema import CandidateBlock
from wfcllm.evaluation.anchor_validation.selection import simulate_retry_selection


def test_retry_selection_accepts_first_hit_within_budget():
    candidates = (
        CandidateBlock("c0", "return x", rank=0),
        CandidateBlock("c1", "return x + 1", rank=1),
        CandidateBlock("c2", "return 1 + x", rank=2),
    )
    signatures = {"c0": (0,), "c1": (1,), "c2": (1,)}

    row = simulate_retry_selection(
        context_id="ctx",
        method="slot_context",
        key_id="key-00",
        gamma=0.5,
        retry_budget=2,
        candidates=candidates,
        signatures_by_candidate_id=signatures,
        valid_set=frozenset({(1,)}),
    )

    assert row.selected_candidate_id == "c1"
    assert row.selected_rank == 1
    assert row.hit_acquired is True
    assert row.fallback is False


def test_retry_selection_falls_back_to_rank_zero_when_no_hit():
    candidates = (CandidateBlock("c0", "return x", rank=0),)

    row = simulate_retry_selection(
        context_id="ctx",
        method="vanilla",
        key_id="key-00",
        gamma=0.5,
        retry_budget=1,
        candidates=candidates,
        signatures_by_candidate_id={"c0": (0,)},
        valid_set=frozenset({(1,)}),
    )

    assert row.selected_candidate_id == "c0"
    assert row.hit_acquired is False
    assert row.fallback is True
    assert row.z_proxy < 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/evaluation/anchor_validation/test_selection.py -v`

Expected: FAIL with missing `selection`.

- [ ] **Step 3: Implement retry simulation**

```python
from __future__ import annotations

import math

from wfcllm.evaluation.anchor_validation.schema import CandidateBlock, SelectionSimulationRow


def simulate_retry_selection(
    context_id: str,
    method: str,
    key_id: str,
    gamma: float,
    retry_budget: int,
    candidates: tuple[CandidateBlock, ...],
    signatures_by_candidate_id: dict[str, tuple[int, ...]],
    valid_set: frozenset[tuple[int, ...]],
) -> SelectionSimulationRow:
    ordered = sorted(candidates, key=lambda item: item.rank)
    budgeted = ordered[:max(1, retry_budget)]
    selected = ordered[0]
    hit_acquired = False
    for candidate in budgeted:
        signature = signatures_by_candidate_id[candidate.candidate_id]
        if signature in valid_set:
            selected = candidate
            hit_acquired = True
            break
    fallback = not hit_acquired
    z_proxy = _z_proxy(1 if hit_acquired else 0, gamma)
    return SelectionSimulationRow(
        context_id=context_id,
        method=method,
        key_id=key_id,
        gamma=gamma,
        retry_budget=retry_budget,
        selected_candidate_id=selected.candidate_id,
        selected_rank=selected.rank,
        hit_acquired=hit_acquired,
        fallback=fallback,
        z_proxy=z_proxy,
        quality={
            **dict(selected.quality),
            "syntax_valid": selected.syntax_valid,
            "parse_valid": selected.parse_valid,
        },
    )


def _z_proxy(observed_hit: int, gamma: float) -> float:
    variance = gamma * (1.0 - gamma)
    if variance <= 0.0:
        return 0.0
    return (observed_hit - gamma) / math.sqrt(variance)
```

- [ ] **Step 4: Re-run selection tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/evaluation/anchor_validation/test_selection.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add wfcllm/evaluation/anchor_validation/selection.py tests/evaluation/anchor_validation/test_selection.py
git commit -m "feat: simulate anchor retry selection"
```

## Task 7: Add Stop/Go Summary, Diagnostic Runner, and CLI

**Files:**
- Create: `wfcllm/evaluation/anchor_validation/summary.py`
- Create: `wfcllm/evaluation/anchor_validation/runner.py`
- Create: `scripts/anchor_validation.py`
- Test: `tests/evaluation/anchor_validation/test_summary.py`
- Test: `tests/evaluation/anchor_validation/test_runner.py`
- Test: `tests/integration/test_anchor_validation_cli.py`

- [ ] **Step 1: Write failing runner tests**

```python
from __future__ import annotations

import json

from wfcllm.evaluation.anchor_validation.runner import AnchorValidationConfig, AnchorValidationRunner
from wfcllm.evaluation.anchor_validation.schema import CandidateBlock, CandidateContext
from wfcllm.evaluation.anchor_validation.io import write_candidate_contexts


def test_runner_writes_metrics_simulation_and_summary(tmp_path):
    pool_path = tmp_path / "candidate_pools.jsonl"
    output_dir = tmp_path / "out"
    write_candidate_contexts(pool_path, [
        CandidateContext(
            context_id="ctx",
            dataset="humaneval",
            task_id="HumanEval/0",
            prompt="def f(x):\n",
            function_signature="def f(x):",
            ast_path=("function_definition", "return_statement"),
            node_type="return_statement",
            parent_node_type="function_definition",
            block_ordinal=0,
            context_hash="ctxhash",
            context_before="def f(x):\n",
            context_after="",
            masked_parent_context="def f(<NAME>):\n    <TARGET_BLOCK>",
            import_and_helper_signatures=(),
            temperature=0.2,
            candidates=(
                CandidateBlock("c0", "return x", 0),
                CandidateBlock("c1", "return x + 1", 1),
            ),
        )
    ])

    runner = AnchorValidationRunner(AnchorValidationConfig(
        pool_path=pool_path,
        output_dir=output_dir,
        secret_keys=("k0", "k1"),
        gammas=(0.5,),
        methods=("vanilla", "random", "slot_context", "seqmark_oracle"),
        retry_budgets=(1, 2),
        lsh_d=2,
        embed_dim=8,
        embedding_mode="hash",
    ))

    result = runner.run()

    assert result.metrics_path.exists()
    assert result.selection_path.exists()
    assert result.summary_path.exists()
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["meta"]["context_count"] == 1
    assert "go_no_go" in summary
```

- [ ] **Step 2: Write failing stop/go summary tests**

```python
from __future__ import annotations

import pytest

from wfcllm.evaluation.anchor_validation.schema import RegionMetricRow, SelectionSimulationRow
from wfcllm.evaluation.anchor_validation.summary import build_anchor_validation_summary


def _metric(
    context_id: str,
    method: str,
    entropy: float,
    key_id: str | None = None,
    gamma_deviation: float | None = None,
    projection_key_id: str = "proj-00",
):
    return RegionMetricRow(
        context_id=context_id,
        dataset="humaneval",
        task_id=context_id,
        method=method,
        projection_key_id=projection_key_id,
        key_id=key_id,
        gamma=0.5 if key_id else None,
        candidate_count=16,
        normalized_entropy=entropy,
        collapse_ratio=1.0 - entropy,
        effective_region_count=2.0,
        hamming_diversity=0.5,
        valid_hit_rate=0.5 if key_id else None,
        gamma_deviation=gamma_deviation,
    )


def test_summary_computes_required_anchor_gate_fields():
    metrics = [
        _metric("ctx1", "vanilla", 0.20),
        _metric("ctx1", "random", 0.23),
        _metric("ctx1", "slot_context", 0.36),
        _metric("ctx1", "seqmark_oracle", 0.50),
        _metric("ctx1", "slot_context", 0.36, key_id="key-00", gamma_deviation=0.02),
        _metric("ctx1", "slot_context", 0.36, key_id="key-01", gamma_deviation=0.03),
    ]
    selection = [
        SelectionSimulationRow("ctx1", "vanilla", "key-00", 0.5, 4, "c0", 0, False, True, -1.0),
        SelectionSimulationRow("ctx1", "slot_context", "key-00", 0.5, 4, "c1", 1, True, False, 1.0),
        SelectionSimulationRow("ctx1", "slot_context", "key-00", 0.5, 8, "c1", 1, True, False, 1.0),
    ]

    summary = build_anchor_validation_summary(
        metrics,
        selection,
        context_count=1,
        methods=("vanilla", "random", "slot_context", "seqmark_oracle"),
    )

    evidence = summary["go_no_go"]["evidence"]
    assert evidence["paired_entropy_delta"]["slot_context_vs_vanilla"]["mean"] == pytest.approx(0.16)
    assert evidence["random_anchor_gap"]["slot_context_minus_random"]["mean"] == pytest.approx(0.13)
    assert evidence["seqmark_oracle_gain_ratio"]["slot_context"] == pytest.approx(0.533333, rel=1e-5)
    assert evidence["valid_hit_balance"]["slot_context"]["max_delta_gamma"] == pytest.approx(0.03)
    assert evidence["retry"]["slot_context"]["budget_4_hit_acquisition"] == pytest.approx(1.0)


def test_summary_averages_geometry_across_projection_keys():
    row_a = _metric("ctx1", "vanilla", 0.20, projection_key_id="proj-00")
    row_b = _metric("ctx1", "vanilla", 0.40, projection_key_id="proj-01")
    slot = _metric("ctx1", "slot_context", 0.50)

    summary = build_anchor_validation_summary(
        [row_a, row_b, slot],
        [],
        context_count=1,
        methods=("vanilla", "slot_context"),
    )

    assert summary["summary"]["mean_entropy_by_method"]["vanilla"] == pytest.approx(0.30)
```

- [ ] **Step 3: Write failing CLI integration tests**

```python
from __future__ import annotations

import json
import subprocess


def test_anchor_validation_cli_build_pool_and_run_diagnostics(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        "\n".join([
            json.dumps({
                "id": "HumanEval/0",
                "dataset": "humaneval",
                "prompt": "def f(x):\n",
                "generated_code": "    return x\n",
                "candidate_index": 0,
            }),
            json.dumps({
                "id": "HumanEval/0",
                "dataset": "humaneval",
                "prompt": "def f(x):\n",
                "generated_code": "    return x + 1\n",
                "candidate_index": 1,
            }),
        ]) + "\n",
        encoding="utf-8",
    )
    pool = tmp_path / "candidate_pools.jsonl"
    out = tmp_path / "diag"

    build = subprocess.run(
        [
            "python", "scripts/anchor_validation.py", "build-pool",
            "--input-jsonl", str(candidates),
            "--output", str(pool),
            "--min-candidates", "2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert build.returncode == 0, build.stderr
    assert pool.exists()

    run = subprocess.run(
        [
            "python", "scripts/anchor_validation.py", "run-diagnostics",
            "--pool", str(pool),
            "--output-dir", str(out),
            "--embedding-mode", "hash",
            "--embed-dim", "8",
            "--lsh-d", "2",
            "--secret-key", "k0",
            "--methods", "vanilla", "random", "slot_context", "seqmark_oracle",
            "--gammas", "0.5",
            "--retry-budgets", "1", "2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert (out / "anchor_validation_summary.json").exists()


def test_anchor_validation_cli_generate_pool_echo_mode(tmp_path):
    source = tmp_path / "sources.jsonl"
    source.write_text(
        json.dumps({
            "id": "HumanEval/0",
            "dataset": "humaneval",
            "prompt": "def f(x):\n",
            "source_code": "def f(x):\n    y = x + 1\n    return y\n",
        }) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "per_block_candidates.jsonl"

    result = subprocess.run(
        [
            "python", "scripts/anchor_validation.py", "generate-pool",
            "--source-jsonl", str(source),
            "--output", str(output),
            "--sampler-mode", "echo",
            "--temperatures", "0.2",
            "--candidates-per-temperature", "1",
            "--max-contexts-per-source", "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output.exists()
```

- [ ] **Step 4: Run summary, runner, and CLI tests to verify they fail**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/evaluation/anchor_validation/test_summary.py tests/evaluation/anchor_validation/test_runner.py tests/integration/test_anchor_validation_cli.py -v`

Expected: FAIL with missing summary, runner, and script modules.

- [ ] **Step 5: Implement stop/go summary aggregation**

```python
from __future__ import annotations

from collections import defaultdict
from statistics import variance
from typing import Any

from wfcllm.evaluation.anchor_validation.metrics import bootstrap_mean_ci
from wfcllm.evaluation.anchor_validation.schema import RegionMetricRow, SelectionSimulationRow


def build_anchor_validation_summary(
    metrics_rows: list[RegionMetricRow],
    selection_rows: list[SelectionSimulationRow],
    context_count: int,
    methods: tuple[str, ...],
) -> dict[str, Any]:
    evidence = {
        "paired_entropy_delta": _paired_entropy_delta(metrics_rows, baseline="vanilla"),
        "random_anchor_gap": _random_anchor_gap(metrics_rows),
        "seqmark_oracle_gain_ratio": _seqmark_oracle_gain_ratio(metrics_rows),
        "valid_hit_balance": _valid_hit_balance(metrics_rows),
        "key_wise_variance": _key_wise_variance(metrics_rows),
        "retry": _retry_summary(selection_rows),
        "low_entropy": _low_entropy_summary(metrics_rows),
        "node_type": _node_type_summary(metrics_rows),
    }
    return {
        "meta": {
            "context_count": context_count,
            "methods": list(methods),
        },
        "summary": {
            "mean_entropy_by_method": _mean_entropy_by_method(metrics_rows),
            "mean_hit_acquisition_by_method": {
                method: values["overall_hit_acquisition"]
                for method, values in evidence["retry"].items()
            },
        },
        "go_no_go": {
            "first_stage_passed": _passes_first_stage(evidence),
            "end_to_end_followup_allowed": _passes_first_stage(evidence),
            "evidence": evidence,
        },
    }


def _unkeyed(rows: list[RegionMetricRow]) -> list[RegionMetricRow]:
    return [row for row in rows if row.key_id is None]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_entropy_by_method(rows: list[RegionMetricRow]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in _unkeyed(rows):
        grouped[row.method].append(row.normalized_entropy)
    return {method: _mean(values) for method, values in sorted(grouped.items())}


def _context_method_means(rows: list[RegionMetricRow]) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in _unkeyed(rows):
        grouped[(row.context_id, row.method)].append(row.normalized_entropy)
    return {key: _mean(values) for key, values in grouped.items()}


def _paired_entropy_delta(rows: list[RegionMetricRow], baseline: str) -> dict[str, dict[str, float | list[float]]]:
    by_context = _context_method_means(rows)
    contexts = {row.context_id for row in _unkeyed(rows)}
    methods = {row.method for row in _unkeyed(rows) if row.method != baseline}
    result: dict[str, dict[str, float | list[float]]] = {}
    for method in sorted(methods):
        deltas = [
            by_context[(context_id, method)] - by_context[(context_id, baseline)]
            for context_id in contexts
            if (context_id, method) in by_context and (context_id, baseline) in by_context
        ]
        ci = bootstrap_mean_ci(deltas, iterations=1000, seed=7)
        result[f"{method}_vs_{baseline}"] = {
            "mean": _mean(deltas),
            "median": sorted(deltas)[len(deltas) // 2] if deltas else 0.0,
            "win_rate": _mean([1.0 if delta > 0 else 0.0 for delta in deltas]),
            "bootstrap_ci_95": [ci[0], ci[1]],
        }
    return result


def _random_anchor_gap(rows: list[RegionMetricRow]) -> dict[str, dict[str, float]]:
    paired = _paired_entropy_delta(rows, baseline="random")
    return {
        key.replace("_vs_random", "_minus_random"): value
        for key, value in paired.items()
        if key.startswith(("slot", "context", "skeleton", "prompt"))
    }


def _seqmark_oracle_gain_ratio(rows: list[RegionMetricRow]) -> dict[str, float]:
    by_context = _context_method_means(rows)
    contexts = {row.context_id for row in _unkeyed(rows)}
    methods = {row.method for row in _unkeyed(rows)} - {"vanilla", "seqmark_oracle"}
    ratios: dict[str, float] = {}
    for method in sorted(methods):
        values: list[float] = []
        for context_id in contexts:
            vanilla = by_context.get((context_id, "vanilla"))
            oracle = by_context.get((context_id, "seqmark_oracle"))
            current = by_context.get((context_id, method))
            if vanilla is None or oracle is None or current is None:
                continue
            denom = oracle - vanilla
            if denom > 0:
                values.append((current - vanilla) / denom)
        ratios[method] = _mean(values)
    return ratios


def _valid_hit_balance(rows: list[RegionMetricRow]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.key_id is not None and row.gamma_deviation is not None:
            grouped[row.method].append(row.gamma_deviation)
    return {
        method: {
            "mean_delta_gamma": _mean(values),
            "max_delta_gamma": max(values) if values else 0.0,
        }
        for method, values in sorted(grouped.items())
    }


def _key_wise_variance(rows: list[RegionMetricRow]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.key_id is not None and row.valid_hit_rate is not None:
            grouped[row.method].append(row.valid_hit_rate)
    return {
        method: variance(values) if len(values) > 1 else 0.0
        for method, values in sorted(grouped.items())
    }


def _retry_summary(rows: list[SelectionSimulationRow]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[SelectionSimulationRow]] = defaultdict(list)
    for row in rows:
        grouped[row.method].append(row)
    result: dict[str, dict[str, float]] = {}
    for method, method_rows in sorted(grouped.items()):
        values = {
            "overall_hit_acquisition": _mean([1.0 if row.hit_acquired else 0.0 for row in method_rows]),
            "overall_fallback_rate": _mean([1.0 if row.fallback else 0.0 for row in method_rows]),
            "overall_z_proxy": _mean([row.z_proxy for row in method_rows]),
            "mean_selected_rank": _mean([float(row.selected_rank) for row in method_rows]),
        }
        for budget in (4, 8):
            budget_rows = [row for row in method_rows if row.retry_budget == budget]
            values[f"budget_{budget}_hit_acquisition"] = _mean([1.0 if row.hit_acquired else 0.0 for row in budget_rows])
            values[f"budget_{budget}_fallback_rate"] = _mean([1.0 if row.fallback else 0.0 for row in budget_rows])
            values[f"budget_{budget}_z_proxy"] = _mean([row.z_proxy for row in budget_rows])
            values[f"budget_{budget}_mean_selected_rank"] = _mean([float(row.selected_rank) for row in budget_rows])
        values.update(_quality_proxy_means(method_rows))
        result[method] = values
    return result


def _quality_proxy_means(rows: list[SelectionSimulationRow]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.quality.items():
            if isinstance(value, bool):
                grouped[f"quality_{key}"].append(1.0 if value else 0.0)
            elif isinstance(value, (int, float)):
                grouped[f"quality_{key}"].append(float(value))
    return {key: _mean(values) for key, values in sorted(grouped.items())}


def _low_entropy_summary(rows: list[RegionMetricRow]) -> dict[str, float]:
    baseline = [
        row.normalized_entropy
        for row in _unkeyed(rows)
        if row.method == "vanilla"
    ]
    if not baseline:
        return {}
    threshold = sorted(baseline)[max(0, int(0.25 * (len(baseline) - 1)))]
    low_contexts = {
        row.context_id
        for row in _unkeyed(rows)
        if row.method == "vanilla" and row.normalized_entropy <= threshold
    }
    return _mean_entropy_by_method([
        row for row in rows if row.context_id in low_contexts
    ])


def _node_type_summary(rows: list[RegionMetricRow]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[RegionMetricRow]] = defaultdict(list)
    for row in _unkeyed(rows):
        if row.node_type is not None:
            grouped[row.node_type].append(row)
    return {
        node_type: _mean_entropy_by_method(node_rows)
        for node_type, node_rows in sorted(grouped.items())
    }


def _passes_first_stage(evidence: dict[str, Any]) -> bool:
    deterministic = ("slot_context", "slot_context_skeleton")
    paired = evidence["paired_entropy_delta"]
    random_gap = evidence["random_anchor_gap"]
    oracle_ratio = evidence["seqmark_oracle_gain_ratio"]
    balance = evidence["valid_hit_balance"]
    retry = evidence["retry"]
    low_entropy = evidence["low_entropy"]
    node_type = evidence["node_type"]
    for method in deterministic:
        if paired.get(f"{method}_vs_vanilla", {}).get("mean", 0.0) < 0.10:
            continue
        if low_entropy.get(method, 0.0) - low_entropy.get("vanilla", 0.0) < 0.15:
            continue
        node_type_wins = sum(
            1
            for values in node_type.values()
            if values.get(method, 0.0) - values.get("vanilla", 0.0) >= 0.05
        )
        if node_type_wins < 2:
            continue
        if random_gap.get(f"{method}_minus_random", {}).get("mean", 0.0) < 0.05:
            continue
        if oracle_ratio.get(method, 0.0) < 0.50:
            continue
        if balance.get(method, {}).get("max_delta_gamma", 1.0) > 0.05:
            continue
        method_retry_gain = max(
            retry.get(method, {}).get("budget_4_hit_acquisition", 0.0),
            retry.get(method, {}).get("budget_8_hit_acquisition", 0.0),
        )
        vanilla_retry_gain = max(
            retry.get("vanilla", {}).get("budget_4_hit_acquisition", 0.0),
            retry.get("vanilla", {}).get("budget_8_hit_acquisition", 0.0),
        )
        random_retry_gain = max(
            retry.get("random", {}).get("budget_4_hit_acquisition", 0.0),
            retry.get("random", {}).get("budget_8_hit_acquisition", 0.0),
        )
        if method_retry_gain <= vanilla_retry_gain:
            continue
        if method_retry_gain <= random_retry_gain:
            continue
        if retry.get(method, {}).get("overall_z_proxy", -999.0) <= retry.get("vanilla", {}).get("overall_z_proxy", -999.0):
            continue
        if retry.get(method, {}).get("overall_fallback_rate", 1.0) > retry.get("vanilla", {}).get("overall_fallback_rate", 1.0) + 0.05:
            continue
        if not _quality_non_regression(retry, method, baseline="vanilla"):
            continue
        if retry.get(method, {}).get("mean_selected_rank", 999.0) > 8.0:
            continue
        return True
    return False


def _quality_non_regression(
    retry: dict[str, dict[str, float]],
    method: str,
    baseline: str,
) -> bool:
    for quality_key in ("quality_syntax_valid", "quality_parse_valid"):
        method_value = retry.get(method, {}).get(quality_key)
        baseline_value = retry.get(baseline, {}).get(quality_key)
        if method_value is not None and baseline_value is not None:
            if method_value < baseline_value - 0.01:
                return False
    return True
```

- [ ] **Step 6: Implement runner dataclasses and method signature computation**

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from wfcllm.evaluation.anchor_validation.anchors import build_anchor_text
from wfcllm.evaluation.anchor_validation.embedding import (
    EmbeddingProvider,
    EncoderEmbeddingProvider,
    HashEmbeddingProvider,
)
from wfcllm.evaluation.anchor_validation.io import load_candidate_contexts, write_jsonl
from wfcllm.evaluation.anchor_validation.metrics import summarize_signature_metrics
from wfcllm.evaluation.anchor_validation.schema import AnchorMethod
from wfcllm.evaluation.anchor_validation.selection import simulate_retry_selection
from wfcllm.evaluation.anchor_validation.summary import build_anchor_validation_summary
from wfcllm.watermark.adaptive_gamma.schedule import quantize_gamma
from wfcllm.watermark.anchor_lsh import (
    anchored_signature,
    random_anchor,
    residual_signature,
    sign_with_planes,
)
from wfcllm.watermark.keying import WatermarkKeying
from wfcllm.watermark.lsh_space import LSHSpace


@dataclass(frozen=True)
class AnchorValidationConfig:
    pool_path: Path
    output_dir: Path
    secret_keys: tuple[str, ...]
    gammas: tuple[float, ...]
    methods: tuple[str, ...]
    retry_budgets: tuple[int, ...]
    lsh_d: int = 3
    embed_dim: int = 128
    embedding_mode: str = "hash"
    encoder_model_path: str = "data/models/codet5-base"
    encoder_checkpoint: Path | None = None
    encoder_device: str = "cpu"
    max_length: int = 256
    use_ordinal_keying: bool = True


@dataclass(frozen=True)
class AnchorValidationResult:
    metrics_path: Path
    selection_path: Path
    summary_path: Path


class AnchorValidationRunner:
    def __init__(self, config: AnchorValidationConfig) -> None:
        self._config = config

    def run(self) -> AnchorValidationResult:
        contexts = load_candidate_contexts(self._config.pool_path)
        provider = _build_embedding_provider(self._config)
        metrics_rows = []
        selection_rows = []

        for context in contexts:
            block_embeddings = {
                candidate.candidate_id: provider.embed(candidate.block_text)
                for candidate in context.candidates
            }
            oracle_anchor = _mean_embedding(tuple(block_embeddings.values()))
            for method_name in self._config.methods:
                method = AnchorMethod(method_name)
                for key_index, secret_key in enumerate(self._config.secret_keys):
                    key_id = f"key-{key_index:02d}"
                    lsh = LSHSpace(secret_key=secret_key, embed_dim=provider.embed_dim, d=self._config.lsh_d)
                    signatures = _signatures_for_method(
                        method=method,
                        context=context,
                        block_embeddings=block_embeddings,
                        provider=provider,
                        planes=lsh.planes,
                        secret_key=secret_key,
                        oracle_anchor=oracle_anchor,
                    )
                    signature_list = [signatures[c.candidate_id] for c in context.candidates]
                    region_count = 2 ** self._config.lsh_d
                    metrics_rows.append(summarize_signature_metrics(
                        context_id=context.context_id,
                        dataset=context.dataset,
                        task_id=context.task_id,
                        method=method.value,
                        signatures=signature_list,
                        region_count=region_count,
                        projection_key_id=key_id,
                        key_id=None,
                        gamma=None,
                        valid_set=None,
                        node_type=context.node_type,
                    ))
                    for gamma in self._config.gammas:
                        gamma_resolution = quantize_gamma(gamma, self._config.lsh_d)
                        keying = WatermarkKeying(secret_key, self._config.lsh_d)
                        ordinal = (
                            context.block_ordinal
                            if self._config.use_ordinal_keying
                            else None
                        )
                        valid_set = keying.derive(
                            context.parent_node_type,
                            k=gamma_resolution.k,
                            ordinal=ordinal,
                        )
                        metrics_rows.append(summarize_signature_metrics(
                            context_id=context.context_id,
                            dataset=context.dataset,
                            task_id=context.task_id,
                            method=method.value,
                            signatures=signature_list,
                            region_count=region_count,
                            projection_key_id=key_id,
                            key_id=key_id,
                            gamma=gamma_resolution.gamma_effective,
                            valid_set=valid_set,
                            node_type=context.node_type,
                        ))
                        for budget in self._config.retry_budgets:
                            selection_rows.append(simulate_retry_selection(
                                context_id=context.context_id,
                                method=method.value,
                                key_id=key_id,
                                gamma=gamma_resolution.gamma_effective,
                                retry_budget=budget,
                                candidates=context.candidates,
                                signatures_by_candidate_id=signatures,
                                valid_set=valid_set,
                            ))

        self._config.output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = write_jsonl(self._config.output_dir / "region_metrics.jsonl", metrics_rows)
        selection_path = write_jsonl(self._config.output_dir / "selection_simulation.jsonl", selection_rows)
        summary_payload = build_anchor_validation_summary(
            metrics_rows,
            selection_rows,
            context_count=len(contexts),
            methods=tuple(self._config.methods),
        )
        summary_payload["meta"].update({
            "pool_path": str(self._config.pool_path),
            "gammas": list(self._config.gammas),
            "retry_budgets": list(self._config.retry_budgets),
            "embedding_mode": self._config.embedding_mode,
            "use_ordinal_keying": self._config.use_ordinal_keying,
        })
        summary_path = self._write_summary(summary_payload)
        return AnchorValidationResult(metrics_path, selection_path, summary_path)
```

Helper code:

```python
def _build_embedding_provider(config: AnchorValidationConfig) -> EmbeddingProvider:
    if config.embedding_mode == "hash":
        return HashEmbeddingProvider(embed_dim=config.embed_dim)
    if config.embedding_mode != "encoder":
        raise ValueError(f"unsupported embedding mode: {config.embedding_mode}")

    import torch
    from transformers import AutoTokenizer

    from wfcllm.encoder.config import EncoderConfig
    from wfcllm.encoder.model import SemanticEncoder

    enc_config = EncoderConfig(
        model_name=config.encoder_model_path,
        embed_dim=config.embed_dim,
    )
    encoder = SemanticEncoder(config=enc_config)
    if config.encoder_checkpoint is not None:
        checkpoint = torch.load(config.encoder_checkpoint, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        encoder.load_state_dict(state_dict)
    encoder = encoder.to(config.encoder_device)
    tokenizer = AutoTokenizer.from_pretrained(config.encoder_model_path)
    return EncoderEmbeddingProvider(
        encoder=encoder,
        tokenizer=tokenizer,
        device=config.encoder_device,
        max_length=config.max_length,
    )


def _mean_embedding(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
    if not values:
        raise ValueError("at least one candidate embedding is required")
    return torch.stack(values).mean(dim=0)


def _signatures_for_method(
    method: AnchorMethod,
    context,
    block_embeddings: dict[str, torch.Tensor],
    provider,
    planes: torch.Tensor,
    secret_key: str,
    oracle_anchor: torch.Tensor,
) -> dict[str, tuple[int, ...]]:
    signatures: dict[str, tuple[int, ...]] = {}
    for candidate in context.candidates:
        u = block_embeddings[candidate.candidate_id]
        if method == AnchorMethod.VANILLA:
            signature = sign_with_planes(u, planes)
        elif method == AnchorMethod.RANDOM:
            anchor = random_anchor(secret_key, context.context_id, method.value, provider.embed_dim)
            signature = anchored_signature(u, planes, anchor)
        elif method == AnchorMethod.SEQMARK_ORACLE:
            signature = residual_signature(u, center=oracle_anchor, planes=planes)
        else:
            anchor_text = build_anchor_text(method, context, candidate)
            anchor = provider.embed(anchor_text)
            signature = anchored_signature(u, planes, anchor)
        signatures[candidate.candidate_id] = signature
    return signatures
```

Summary write code:

```python
def _write_summary(self, payload: dict) -> Path:
    path = self._config.output_dir / "anchor_validation_summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
```

- [ ] **Step 7: Implement `scripts/anchor_validation.py` CLI**

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.evaluation.anchor_validation.candidate_generation import (
    GenerationContextSource,
    build_hf_sampler,
    generate_candidate_rows,
)
from wfcllm.evaluation.anchor_validation.io import read_jsonl, write_candidate_contexts
from wfcllm.evaluation.anchor_validation.pool_builder import build_candidate_contexts_from_records
from wfcllm.evaluation.anchor_validation.runner import AnchorValidationConfig, AnchorValidationRunner


def _cmd_generate_pool(args: argparse.Namespace) -> int:
    sources = _load_generation_sources(tuple(Path(path) for path in args.source_jsonl))
    if args.sampler_mode == "echo":
        sampler = lambda prompt, temperature, sample_index: "pass"
    else:
        if not args.lm_model_path:
            raise ValueError("--lm-model-path is required with --sampler-mode hf")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.lm_model_path)
        model = AutoModelForCausalLM.from_pretrained(args.lm_model_path, device_map=args.device_map)
        sampler = build_hf_sampler(model, tokenizer, max_new_tokens=args.max_new_tokens)

    rows = generate_candidate_rows(
        sources=tuple(sources),
        sampler=sampler,
        temperatures=tuple(float(value) for value in args.temperatures),
        candidates_per_temperature=args.candidates_per_temperature,
        max_contexts_per_source=args.max_contexts_per_source,
    )
    from wfcllm.evaluation.anchor_validation.io import write_jsonl

    write_jsonl(Path(args.output), rows)
    print(f"[anchor-validation] wrote {len(rows)} per-block candidate rows to {args.output}")
    return 0


def _load_generation_sources(paths: tuple[Path, ...]) -> list[GenerationContextSource]:
    sources: list[GenerationContextSource] = []
    for path in paths:
        for row in read_jsonl(path):
            prompt = str(row.get("prompt", ""))
            generated = str(row.get("generated_code", row.get("solution", "")))
            source_code = str(row.get("source_code", prompt + generated))
            sources.append(GenerationContextSource(
                dataset=str(row.get("dataset", "unknown")),
                task_id=str(row.get("id", row.get("task_id", ""))),
                prompt=prompt,
                source_code=source_code,
            ))
    return sources


def _cmd_build_pool(args: argparse.Namespace) -> int:
    records = []
    for path in args.input_jsonl:
        records.extend(read_jsonl(Path(path)))
    contexts = build_candidate_contexts_from_records(
        records,
        min_candidates=args.min_candidates,
        max_contexts_per_task=args.max_contexts_per_task,
    )
    write_candidate_contexts(Path(args.output), contexts)
    print(f"[anchor-validation] wrote {len(contexts)} contexts to {args.output}")
    return 0


def _cmd_run_diagnostics(args: argparse.Namespace) -> int:
    config = AnchorValidationConfig(
        pool_path=Path(args.pool),
        output_dir=Path(args.output_dir),
        secret_keys=tuple(args.secret_key),
        gammas=tuple(float(value) for value in args.gammas),
        methods=tuple(args.methods),
        retry_budgets=tuple(int(value) for value in args.retry_budgets),
        lsh_d=args.lsh_d,
        embed_dim=args.embed_dim,
        embedding_mode=args.embedding_mode,
        encoder_model_path=args.encoder_model_path,
        encoder_checkpoint=Path(args.encoder_checkpoint) if args.encoder_checkpoint else None,
        encoder_device=args.encoder_device,
        max_length=args.max_length,
        use_ordinal_keying=not args.legacy_parent_keying,
    )
    result = AnchorValidationRunner(config).run()
    print(f"[anchor-validation] metrics: {result.metrics_path}")
    print(f"[anchor-validation] selection: {result.selection_path}")
    print(f"[anchor-validation] summary: {result.summary_path}")
    return 0
```

Parser shape:

```python
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Anchor effectiveness diagnostics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-pool")
    generate.add_argument("--source-jsonl", nargs="+", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--sampler-mode", choices=["hf", "echo"], default="hf")
    generate.add_argument("--lm-model-path", default=None)
    generate.add_argument("--device-map", default="auto")
    generate.add_argument("--max-new-tokens", type=int, default=64)
    generate.add_argument("--temperatures", nargs="+", default=["0.2", "0.4", "0.7"])
    generate.add_argument("--candidates-per-temperature", type=int, default=16)
    generate.add_argument("--max-contexts-per-source", type=int, default=None)
    generate.set_defaults(func=_cmd_generate_pool)

    build = subparsers.add_parser("build-pool")
    build.add_argument("--input-jsonl", nargs="+", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--min-candidates", type=int, default=2)
    build.add_argument("--max-contexts-per-task", type=int, default=None)
    build.set_defaults(func=_cmd_build_pool)

    run = subparsers.add_parser("run-diagnostics")
    run.add_argument("--pool", required=True)
    run.add_argument("--output-dir", default="data/diagnostics/anchor_validation")
    run.add_argument("--embedding-mode", choices=["hash", "encoder"], default="hash")
    run.add_argument("--embed-dim", type=int, default=128)
    run.add_argument("--encoder-model-path", default="data/models/codet5-base")
    run.add_argument("--encoder-checkpoint", default=None)
    run.add_argument("--encoder-device", default="cpu")
    run.add_argument("--max-length", type=int, default=256)
    run.add_argument("--lsh-d", type=int, default=3)
    run.add_argument("--secret-key", nargs="+", required=True)
    run.add_argument(
        "--legacy-parent-keying",
        action="store_true",
        help="derive valid sets from parent node type only; default uses block ordinal",
    )
    run.add_argument("--methods", nargs="+", required=True)
    run.add_argument("--gammas", nargs="+", default=["0.5"])
    run.add_argument("--retry-budgets", nargs="+", default=["1", "4", "8", "16"])
    run.set_defaults(func=_cmd_run_diagnostics)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 8: Re-run summary, runner, and CLI tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/evaluation/anchor_validation/test_summary.py tests/evaluation/anchor_validation/test_runner.py tests/integration/test_anchor_validation_cli.py -v`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add wfcllm/evaluation/anchor_validation/summary.py wfcllm/evaluation/anchor_validation/runner.py scripts/anchor_validation.py tests/evaluation/anchor_validation/test_summary.py tests/evaluation/anchor_validation/test_runner.py tests/integration/test_anchor_validation_cli.py
git commit -m "feat: add anchor validation runner"
```

## Task 8: Add Runbook and Final Verification

**Files:**
- Create: `docs/experiment/anchor-validation-runbook.md`
- Modify: `docs/experiment/2026-06-02-anchor-effectiveness-validation-plan.md` only if a short "Implementation entry point" pointer is useful and does not rewrite the spec.

- [ ] **Step 1: Write the runbook**

Create `docs/experiment/anchor-validation-runbook.md` with this structure:

````markdown
# Anchor Validation Runbook

## Scope

This runbook executes R001-R004 from `docs/experiment/2026-06-02-anchor-effectiveness-validation-plan.md`.
It does not enable production AO-LSH.

## Build Candidate Pool From Explicit Per-Block Candidates

Preferred input rows contain `candidate_context_id`, `block_text`, `context_hash`, `context_before`, `context_after`, `masked_parent_context`, `import_and_helper_signatures`, `node_type`, `parent_node_type`, and `block_ordinal`.

## Generate Candidates When Full Pools Are Unavailable

Use this path for R001/R002 when existing cap artifacts do not include K candidates per context. `source_seed_contexts.jsonl` should contain parseable `source_code` rows from reference solutions or seed completions.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
conda run -n WFCLLM python scripts/anchor_validation.py generate-pool \
  --source-jsonl data/diagnostics/anchor_validation/source_seed_contexts.jsonl \
  --output data/diagnostics/anchor_validation/per_block_candidates.jsonl \
  --sampler-mode hf \
  --lm-model-path data/models/deepseek-coder-7b-base \
  --temperatures 0.2 0.4 0.7 \
  --candidates-per-temperature 16 \
  --max-contexts-per-source 3
```

```bash
python scripts/anchor_validation.py build-pool \
  --input-jsonl data/diagnostics/anchor_validation/per_block_candidates.jsonl \
  --output data/diagnostics/anchor_validation/candidate_pools.jsonl \
  --min-candidates 8
```

## Conservative Whole-Program Fallback

Repeated whole-program candidate rows are accepted only when the masked non-target context fingerprint matches. Ambiguous rows are skipped rather than grouped by ordinal alone.

```bash
python scripts/anchor_validation.py build-pool \
  --input-jsonl data/watermarked/candidate_*.jsonl \
  --output data/diagnostics/anchor_validation/candidate_pools_from_programs.jsonl \
  --min-candidates 8 \
  --max-contexts-per-task 3
```

## Smoke Diagnostics With Hash Embeddings

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM python scripts/anchor_validation.py run-diagnostics \
  --pool data/diagnostics/anchor_validation/candidate_pools.jsonl \
  --output-dir data/diagnostics/anchor_validation/hash_smoke \
  --embedding-mode hash \
  --embed-dim 128 \
  --lsh-d 3 \
  --secret-key anchor-key-00 anchor-key-01 anchor-key-02 anchor-key-03 anchor-key-04 anchor-key-05 anchor-key-06 anchor-key-07 anchor-key-08 anchor-key-09 \
  --methods vanilla random slot context skeleton slot_context slot_context_skeleton prompt_aware seqmark_oracle \
  --gammas 0.25 0.5 0.75 \
  --retry-budgets 1 4 8 16
```

## Real Diagnostics

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
conda run -n WFCLLM python scripts/anchor_validation.py run-diagnostics \
  --pool data/diagnostics/anchor_validation/candidate_pools.jsonl \
  --output-dir data/diagnostics/anchor_validation/encoder_r001 \
  --embedding-mode encoder \
  --encoder-model-path data/models/codet5-base \
  --encoder-device cpu \
  --embed-dim 128 \
  --lsh-d 3 \
  --secret-key anchor-key-00 anchor-key-01 anchor-key-02 anchor-key-03 anchor-key-04 anchor-key-05 anchor-key-06 anchor-key-07 anchor-key-08 anchor-key-09 \
  --methods vanilla random slot context skeleton slot_context slot_context_skeleton prompt_aware seqmark_oracle \
  --gammas 0.25 0.5 0.75 \
  --retry-budgets 1 4 8 16
```

Keep `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `HF_DATASETS_OFFLINE=1` unless the user explicitly asks for online behavior.

## Stop/Go

Continue to end-to-end AO-LSH only if the summary shows deterministic anchors beating vanilla and random controls, valid-hit balance within the spec threshold, and retry simulation improvement at B=4 or B=8.
````

- [ ] **Step 2: Run all anchor validation tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_anchor_lsh.py tests/evaluation/anchor_validation/ tests/integration/test_anchor_validation_cli.py -v`

Expected: PASS.

- [ ] **Step 3: Run affected existing semantic tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_lsh_space.py tests/watermark/test_verifier.py tests/extract/test_scorer.py -v`

Expected: PASS.

- [ ] **Step 4: Run syntax smoke validation**

Run: `conda run -n WFCLLM python -m compileall wfcllm run.py scripts tools`

Expected: PASS with no compile errors.

- [ ] **Step 5: Check git diff for artifact leakage**

Run: `git status --short`

Expected: Only planned source, test, script, and doc files are modified or untracked. No `data/diagnostics/anchor_validation/*` outputs should be staged.

- [ ] **Step 6: Commit**

```bash
git add wfcllm/watermark/anchor_lsh.py wfcllm/watermark/lsh_space.py wfcllm/evaluation/anchor_validation scripts/anchor_validation.py tests/watermark/test_anchor_lsh.py tests/evaluation/anchor_validation tests/integration/test_anchor_validation_cli.py docs/experiment/anchor-validation-runbook.md
git commit -m "feat: add anchor effectiveness diagnostics"
```

## Post-Implementation Smoke Commands

Use these after all tasks are complete:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_anchor_lsh.py tests/evaluation/anchor_validation/ tests/integration/test_anchor_validation_cli.py -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_lsh_space.py tests/watermark/test_verifier.py tests/extract/test_scorer.py -v
conda run -n WFCLLM python -m compileall wfcllm run.py scripts tools
```

Expected: all pytest targets PASS, and compileall reports no syntax errors.

## Follow-Up Plan Trigger

Create a new implementation plan for end-to-end AO-LSH only after diagnostic artifacts support the GO criteria from the spec:

- M5 or M6 normalized entropy improves over M0 by at least `+0.10`.
- Low-entropy contexts improve by at least `+0.15`.
- Deterministic anchors reach at least `50%` of SeqMark oracle gain.
- Deterministic anchors beat random anchor by at least `+0.05` normalized entropy or a significant bootstrap CI.
- Valid-hit balance satisfies `Delta_gamma <= 0.05`.
- Offline retry simulation improves hit acquisition without lowering quality proxy.

The follow-up plan should modify production surfaces: `wfcllm.watermark.config`, `wfcllm.watermark.verifier`, `wfcllm.watermark.semantic_channel`, `wfcllm.watermark.orchestrator`, `wfcllm.extract.scorer`, `wfcllm.extract.detector`, `wfcllm.watermark.pipeline`, `wfcllm.cli.config_resolver`, and the relevant integration tests.
