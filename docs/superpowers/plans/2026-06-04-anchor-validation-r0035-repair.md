# Anchor Validation R003.5 Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the anchor validation harness so R003.5 can separate CodeT5 anchor signal, valid-set/keying balance skew, and pool coverage limits without changing production watermark behavior.

**Architecture:** Keep all changes inside `wfcllm.evaluation.anchor_validation` and `scripts/anchor_validation.py`. Add `primary_method` diagnostic evidence as an additive summary layer while preserving production `first_stage_passed`; add balance-skew diagnostics as an additive `anchor_diagnostics.balance_skew` object; add two CodeT5 comment anchor variants through the existing `AnchorMethod`/`build_anchor_text`/runner cache path; then run two remote R003.5 experiments using the existing default ordinal keying and existing `--legacy-parent-keying` option.

**Tech Stack:** Python 3.11/3.13, pytest, PyTorch tensors, existing WFCLLM anchor validation JSON/JSONL artifacts, Conda env `WFCLLM`, offline HuggingFace assets.

---

## Scope Guard

Do not change production watermark generation or extraction behavior. This plan only modifies diagnostic validation code under `wfcllm/evaluation/anchor_validation/`, the standalone CLI `scripts/anchor_validation.py`, nearby tests, and optionally a runbook note.

Do not regenerate candidate pools in this plan. R003.5 must start from:

```text
data/diagnostics/anchor_validation/candidate_pools_r002.jsonl
```

Do not treat R003.5 as a production GO even if a primary method improves. The R002 pool has only 39 contexts and 3 node-type categories, so production `first_stage_passed` must remain constrained by data-quality gates.

## File Map

### Modify

- `wfcllm/evaluation/anchor_validation/schema.py`
  - Add `codet5_comment_minimal` and `codet5_comment_contextual` to `AnchorMethod`.
- `wfcllm/evaluation/anchor_validation/anchors.py`
  - Add two CodeT5 comment anchor builders.
  - Wire builders through `build_anchor_text`.
  - Ensure new methods require candidate blocks and do not leak `secret_key` or raw candidate text.
- `wfcllm/evaluation/anchor_validation/runner.py`
  - Add `AnchorValidationConfig.primary_method`.
  - Pass `primary_method` into summary building.
  - Ensure runner cache treats the new methods as candidate-specific.
- `wfcllm/evaluation/anchor_validation/summary.py`
  - Add additive `primary_method` evidence and gates.
  - Parameterize gate helpers by method without changing production data-quality semantics.
  - Add `anchor_diagnostics.balance_skew`.
  - Add signed delta and distribution metrics derived from existing keyed metric rows.
- `scripts/anchor_validation.py`
  - Add `run-diagnostics --primary-method`.
- `tests/evaluation/anchor_validation/test_anchors.py`
  - Test new CodeT5 comment variants.
  - Test real R002-like contextual parseability with representative contexts.
- `tests/evaluation/anchor_validation/test_runner.py`
  - Test `primary_method` is passed into summary output through the runner.
  - Test missing primary method fails clearly.
- `tests/evaluation/anchor_validation/test_summary.py`
  - Test primary method evidence/gates and baseline validation.
  - Test balance skew signed metrics and distribution aggregation.
- `tests/integration/test_anchor_validation_cli.py`
  - Test `--primary-method` and new method names through CLI.
- `docs/experiment/anchor-validation-runbook.md`
  - Optional: add R003.5 ordinal and parent-only commands after tests pass.

### Do Not Modify

- `wfcllm/watermark/*` production runtime behavior.
- `wfcllm/extract/*` extraction runtime behavior.
- Persisted existing field names in `anchor_validation_summary.json`, `region_metrics.jsonl`, or `selection_simulation.jsonl`.

## Task 1: Add Primary Method Diagnostic Evidence

**Files:**
- Modify: `wfcllm/evaluation/anchor_validation/runner.py`
- Modify: `wfcllm/evaluation/anchor_validation/summary.py`
- Modify: `scripts/anchor_validation.py`
- Test: `tests/evaluation/anchor_validation/test_summary.py`
- Test: `tests/evaluation/anchor_validation/test_runner.py`
- Test: `tests/integration/test_anchor_validation_cli.py`

- [ ] **Step 1: Write failing summary tests for primary-method evidence**

Add these tests to `tests/evaluation/anchor_validation/test_summary.py`:

```python
def test_summary_reports_primary_method_evidence_without_changing_data_quality_gate():
    metrics = [
        _metric("ctx1", "vanilla", 0.20),
        _metric("ctx1", "random", 0.25),
        _metric("ctx1", "codet5_comment_anchor", 0.40),
        _metric("ctx1", "seqmark_oracle", 0.70),
        _metric(
            "ctx1",
            "codet5_comment_anchor",
            0.40,
            key_id="key-00",
            gamma_deviation=0.10,
        ),
    ]
    selection = [
        _selection("ctx1", "vanilla", 4, hit=False, fallback=True),
        _selection("ctx1", "random", 4, hit=False, fallback=True),
        _selection("ctx1", "codet5_comment_anchor", 4, hit=True, fallback=False),
    ]
    pool_quality = {
        "context_count": 39,
        "candidates_per_context": {"median": 20},
        "node_type_distribution": {
            "expression_statement": 24,
            "import_from_statement": 12,
            "return_statement": 3,
        },
        "parse_valid_rate": 1.0,
    }

    summary = build_anchor_validation_summary(
        metrics,
        selection,
        context_count=39,
        methods=("vanilla", "random", "codet5_comment_anchor", "seqmark_oracle"),
        pool_quality=pool_quality,
        primary_method="codet5_comment_anchor",
    )

    assert summary["meta"]["primary_method"] == "codet5_comment_anchor"
    assert summary["go_no_go"]["primary_method"] == "codet5_comment_anchor"
    assert summary["go_no_go"]["first_stage_passed"] is False
    assert summary["go_no_go"]["gates"]["data_quality_gate"]["passed"] is False

    evidence = summary["go_no_go"]["primary_method_evidence"]
    assert evidence["vs_vanilla"]["mean"] == pytest.approx(0.20)
    assert evidence["minus_random"]["mean"] == pytest.approx(0.15)
    assert evidence["oracle_gap"]["gain_ratio"] == pytest.approx(0.40)
    assert evidence["valid_hit_balance"]["mean_delta_gamma"] == pytest.approx(0.10)
    assert evidence["retry"]["budget_4_hit_acquisition"] == pytest.approx(1.0)
```

Add this validation test:

```python
def test_summary_rejects_missing_primary_method():
    metrics = [
        _metric("ctx1", "vanilla", 0.20),
        _metric("ctx1", "random", 0.25),
        _metric("ctx1", "seqmark_oracle", 0.70),
    ]

    with pytest.raises(ValueError, match="primary_method .* is not present"):
        build_anchor_validation_summary(
            metrics,
            [],
            context_count=1,
            methods=("vanilla", "random", "seqmark_oracle"),
            primary_method="codet5_comment_anchor",
        )
```

Add this baseline validation test:

```python
def test_summary_rejects_primary_method_without_required_baselines():
    metrics = [
        _metric("ctx1", "vanilla", 0.20),
        _metric("ctx1", "codet5_comment_anchor", 0.40),
    ]

    with pytest.raises(ValueError, match="required baseline methods"):
        build_anchor_validation_summary(
            metrics,
            [],
            context_count=1,
            methods=("vanilla", "codet5_comment_anchor"),
            primary_method="codet5_comment_anchor",
        )
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/evaluation/anchor_validation/test_summary.py -v
```

Expected: FAIL because `build_anchor_validation_summary` does not accept `primary_method` and does not emit `primary_method_evidence`.

- [ ] **Step 3: Implement primary_method in summary**

In `wfcllm/evaluation/anchor_validation/summary.py`, change the signature:

```python
def build_anchor_validation_summary(
    metrics_rows: list[RegionMetricRow],
    selection_rows: list[SelectionSimulationRow],
    context_count: int,
    methods: tuple[str, ...],
    pool_quality: dict[str, Any] | None = None,
    empirical_gamma_rows: list[dict[str, Any]] | None = None,
    method_oracle_agreement: dict[str, dict[str, Any]] | None = None,
    primary_method: str = "role_aware_slot_context",
) -> dict[str, Any]:
```

Add helpers near existing summary helpers:

```python
_REQUIRED_PRIMARY_BASELINES = {"vanilla", "random", "seqmark_oracle"}


def _validate_primary_method(
    methods: tuple[str, ...],
    rows: list[RegionMetricRow],
    primary_method: str,
) -> None:
    method_set = set(methods)
    if primary_method not in method_set:
        raise ValueError(f"primary_method {primary_method!r} is not present in methods")
    unkeyed_methods = {row.method for row in _unkeyed(rows)}
    missing = sorted(_REQUIRED_PRIMARY_BASELINES - unkeyed_methods)
    if missing:
        raise ValueError(f"required baseline methods missing for primary_method evidence: {missing}")
    if primary_method not in unkeyed_methods:
        raise ValueError(f"primary_method {primary_method!r} has no unkeyed metric rows")
```

Add primary evidence helper:

```python
def _primary_method_evidence(
    evidence: dict[str, Any],
    primary_method: str,
) -> dict[str, Any]:
    return {
        "vs_vanilla": evidence["paired_entropy_delta"].get(
            f"{primary_method}_vs_vanilla",
            {},
        ),
        "minus_random": evidence["random_anchor_gap"].get(
            f"{primary_method}_minus_random",
            {},
        ),
        "oracle_gap": evidence["oracle_gap"].get(primary_method, {}),
        "gain_ratio": evidence["seqmark_oracle_gain_ratio"].get(primary_method, 0.0),
        "valid_hit_balance": evidence["valid_hit_balance"].get(primary_method, {}),
        "retry": evidence["retry"].get(primary_method, {}),
    }
```

Update the build function after evidence is created:

```python
    _validate_primary_method(methods, metrics_rows, primary_method)
    primary_evidence = _primary_method_evidence(evidence, primary_method)
    gates = _layered_gates(
        evidence,
        pool_quality or {"context_count": context_count},
        primary_method=primary_method,
    )
    primary_method_gates = _primary_method_gates(evidence, primary_method)
```

Update returned payload:

```python
        "meta": {
            "context_count": context_count,
            "methods": list(methods),
            "primary_method": primary_method,
        },
        ...
        "go_no_go": {
            "first_stage_passed": first_stage_passed,
            "end_to_end_followup_allowed": first_stage_passed,
            "primary_method": primary_method,
            "primary_method_evidence": primary_evidence,
            "primary_method_gates": primary_method_gates,
            "gates": gates,
            "evidence": evidence,
        },
```

- [ ] **Step 4: Parameterize gate helpers without changing data-quality semantics**

Change `_layered_gates`:

```python
def _layered_gates(
    evidence: dict[str, Any],
    pool_quality: dict[str, Any],
    primary_method: str,
) -> dict[str, dict[str, Any]]:
    return {
        "data_quality_gate": _data_quality_gate(pool_quality),
        "anchor_signal_gate": _anchor_signal_gate(evidence, primary_method),
        "vanilla_improvement_gate": _vanilla_improvement_gate(evidence, primary_method),
        "balance_gate": _balance_gate(evidence, primary_method),
        "retry_gate": _retry_gate(evidence, primary_method),
    }
```

Change `_anchor_signal_gate`:

```python
def _anchor_signal_gate(evidence: dict[str, Any], primary_method: str) -> dict[str, Any]:
    reasons: list[str] = []
    key = f"{primary_method}_minus_random"
    delta = evidence["random_anchor_gap"].get(key, {})
    ci = delta.get("bootstrap_ci_95", [0.0, 0.0])
    ci_lower = float(ci[0]) if ci else 0.0
    win_rate = float(delta.get("win_rate", 0.0))
    if ci_lower <= 0.0:
        reasons.append(f"{key} CI lower {ci_lower:.4f} <= 0")
    if win_rate < 0.60:
        reasons.append(f"{key} win_rate {win_rate:.4f} < 0.60")
    return _gate(not reasons, reasons)
```

Change `_vanilla_improvement_gate`:

```python
def _vanilla_improvement_gate(evidence: dict[str, Any], primary_method: str) -> dict[str, Any]:
    reasons: list[str] = []
    key = f"{primary_method}_vs_vanilla"
    delta = evidence["paired_entropy_delta"].get(key, {})
    ci = delta.get("bootstrap_ci_95", [0.0, 0.0])
    mean_delta = float(delta.get("mean", 0.0))
    ci_lower = float(ci[0]) if ci else 0.0
    gain_ratio = float(evidence["oracle_gap"].get(primary_method, {}).get("gain_ratio", 0.0))
    if mean_delta <= 0.03:
        reasons.append(f"{key} mean {mean_delta:.4f} <= 0.03")
    if ci_lower < 0.0:
        reasons.append(f"{key} CI lower {ci_lower:.4f} < 0")
    if gain_ratio < 0.15:
        reasons.append(f"gain_ratio {gain_ratio:.4f} < 0.15")
    return _gate(not reasons, reasons)
```

Change `_balance_gate`:

```python
def _balance_gate(evidence: dict[str, Any], primary_method: str) -> dict[str, Any]:
    reasons: list[str] = []
    balance = evidence["valid_hit_balance"].get(primary_method, {})
    max_delta = float(balance.get("max_delta_gamma", 1.0))
    mean_delta = float(balance.get("mean_delta_gamma", 1.0))
    if max_delta > 0.35:
        reasons.append(f"max_delta_gamma {max_delta:.4f} > 0.35")
    if mean_delta > 0.15:
        reasons.append(f"mean_delta_gamma {mean_delta:.4f} > 0.15")
    return _gate(not reasons, reasons)
```

Change `_retry_gate`:

```python
def _retry_gate(evidence: dict[str, Any], primary_method: str) -> dict[str, Any]:
    reasons: list[str] = []
    retry = evidence["retry"]
    method = retry.get(primary_method, {})
    vanilla = retry.get("vanilla", {})
    random = retry.get("random", {})
    method_hit = float(method.get("budget_4_hit_acquisition", 0.0))
    vanilla_hit = float(vanilla.get("budget_4_hit_acquisition", 0.0))
    random_hit = float(random.get("budget_4_hit_acquisition", 0.0))
    fallback = float(method.get("budget_4_fallback_rate", method.get("overall_fallback_rate", 1.0)))
    if method_hit <= vanilla_hit:
        reasons.append(f"budget_4_hit_acquisition {method_hit:.4f} <= vanilla {vanilla_hit:.4f}")
    if method_hit <= random_hit:
        reasons.append(f"budget_4_hit_acquisition {method_hit:.4f} <= random {random_hit:.4f}")
    if fallback > 0.15:
        reasons.append(f"fallback_rate {fallback:.4f} > 0.15")
    return _gate(not reasons, reasons)
```

Add diagnostic-only gates:

```python
def _primary_method_gates(
    evidence: dict[str, Any],
    primary_method: str,
    epsilon: float = 0.002,
) -> dict[str, dict[str, Any]]:
    return {
        "diagnostic_positive": _primary_diagnostic_positive_gate(
            evidence,
            primary_method,
            epsilon,
        ),
        "gate_positive": _primary_gate_positive_gate(evidence, primary_method),
    }
```

Diagnostic positive:

```python
def _primary_diagnostic_positive_gate(
    evidence: dict[str, Any],
    primary_method: str,
    epsilon: float,
) -> dict[str, Any]:
    reasons: list[str] = []
    vs = evidence["paired_entropy_delta"].get(f"{primary_method}_vs_vanilla", {})
    rnd = evidence["random_anchor_gap"].get(f"{primary_method}_minus_random", {})
    vs_ci = vs.get("bootstrap_ci_95", [0.0, 0.0])
    vs_mean = float(vs.get("mean", 0.0))
    vs_lower = float(vs_ci[0]) if vs_ci else 0.0
    rnd_mean = float(rnd.get("mean", 0.0))
    if vs_mean <= 0.0:
        reasons.append(f"{primary_method}_vs_vanilla mean {vs_mean:.4f} <= 0")
    if vs_lower < -epsilon:
        reasons.append(f"{primary_method}_vs_vanilla CI lower {vs_lower:.4f} < -{epsilon:g}")
    if rnd_mean <= 0.0:
        reasons.append(f"{primary_method}_minus_random mean {rnd_mean:.4f} <= 0")
    return _gate(not reasons, reasons)
```

Gate positive:

```python
def _primary_gate_positive_gate(evidence: dict[str, Any], primary_method: str) -> dict[str, Any]:
    reasons: list[str] = []
    vs = evidence["paired_entropy_delta"].get(f"{primary_method}_vs_vanilla", {})
    rnd = evidence["random_anchor_gap"].get(f"{primary_method}_minus_random", {})
    vs_ci = vs.get("bootstrap_ci_95", [0.0, 0.0])
    rnd_ci = rnd.get("bootstrap_ci_95", [0.0, 0.0])
    vs_lower = float(vs_ci[0]) if vs_ci else 0.0
    rnd_lower = float(rnd_ci[0]) if rnd_ci else 0.0
    rnd_win = float(rnd.get("win_rate", 0.0))
    vs_win = float(vs.get("win_rate", 0.0))
    if vs_lower < 0.0:
        reasons.append(f"{primary_method}_vs_vanilla CI lower {vs_lower:.4f} < 0")
    if rnd_lower < 0.0:
        reasons.append(f"{primary_method}_minus_random CI lower {rnd_lower:.4f} < 0")
    if rnd_win < 0.60:
        reasons.append(f"{primary_method}_minus_random win_rate {rnd_win:.4f} < 0.60")
    if vs_win < 0.60:
        reasons.append(f"{primary_method}_vs_vanilla win_rate {vs_win:.4f} < 0.60")
    return _gate(not reasons, reasons)
```

- [ ] **Step 5: Wire runner and CLI**

In `AnchorValidationConfig`, add:

```python
    primary_method: str = "role_aware_slot_context"
```

In runner summary call:

```python
            primary_method=self._config.primary_method,
```

In `scripts/anchor_validation.py`, pass:

```python
        primary_method=args.primary_method,
```

Add parser argument:

```python
    run.add_argument("--primary-method", default="role_aware_slot_context")
```

- [ ] **Step 6: Add runner and CLI tests**

In `tests/evaluation/anchor_validation/test_runner.py`, add a config that includes:

```python
methods=("vanilla", "random", "codet5_comment_anchor", "seqmark_oracle")
primary_method="codet5_comment_anchor"
```

Assert:

```python
assert summary["meta"]["primary_method"] == "codet5_comment_anchor"
assert summary["go_no_go"]["primary_method"] == "codet5_comment_anchor"
assert "primary_method_evidence" in summary["go_no_go"]
```

In `tests/integration/test_anchor_validation_cli.py`, add to an existing run:

```python
"--primary-method",
"codet5_comment_anchor",
```

and include `codet5_comment_anchor` in `--methods`. Assert summary meta:

```python
assert summary["meta"]["primary_method"] == "codet5_comment_anchor"
```

- [ ] **Step 7: Run tests**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/evaluation/anchor_validation/test_summary.py tests/evaluation/anchor_validation/test_runner.py tests/integration/test_anchor_validation_cli.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add wfcllm/evaluation/anchor_validation/summary.py wfcllm/evaluation/anchor_validation/runner.py scripts/anchor_validation.py tests/evaluation/anchor_validation/test_summary.py tests/evaluation/anchor_validation/test_runner.py tests/integration/test_anchor_validation_cli.py
git commit -m "fix: add primary anchor diagnostics"
```

## Task 2: Add Balance Skew Diagnostics

**Files:**
- Modify: `wfcllm/evaluation/anchor_validation/summary.py`
- Test: `tests/evaluation/anchor_validation/test_summary.py`

- [ ] **Step 1: Write failing balance skew tests**

Add to `tests/evaluation/anchor_validation/test_summary.py`:

```python
def test_summary_reports_signed_balance_skew_distribution():
    metrics = [
        _metric("ctx1", "vanilla", 0.20),
        _metric("ctx1", "random", 0.25),
        _metric("ctx1", "codet5_comment_anchor", 0.40),
        _metric("ctx1", "seqmark_oracle", 0.70),
        _metric(
            "ctx1",
            "codet5_comment_anchor",
            0.40,
            key_id="key-00",
            gamma_deviation=0.25,
            candidate_count=4,
            block_ordinal=0,
            node_type="return_statement",
        ),
        _metric(
            "ctx2",
            "codet5_comment_anchor",
            0.30,
            key_id="key-01",
            gamma_deviation=0.75,
            candidate_count=20,
            block_ordinal=8,
            node_type="expression_statement",
        ),
    ]
    # Patch explicit values after dataclass creation because _metric uses valid_hit_rate=0.5.
    metrics[-2] = RegionMetricRow(
        **{**metrics[-2].__dict__, "gamma": 0.25, "valid_hit_rate": 0.50, "gamma_deviation": 0.25}
    )
    metrics[-1] = RegionMetricRow(
        **{**metrics[-1].__dict__, "gamma": 0.75, "valid_hit_rate": 0.00, "gamma_deviation": 0.75}
    )

    summary = build_anchor_validation_summary(
        metrics,
        [],
        context_count=2,
        methods=("vanilla", "random", "codet5_comment_anchor", "seqmark_oracle"),
        primary_method="codet5_comment_anchor",
    )

    skew = summary["anchor_diagnostics"]["balance_skew"]
    method = skew["by_method"]["codet5_comment_anchor"]
    assert method["row_count"] == 2.0
    assert method["mean_abs_delta"] == pytest.approx(0.50)
    assert method["median_abs_delta"] == pytest.approx(0.75)
    assert method["p90_abs_delta"] == pytest.approx(0.75)
    assert method["fraction_abs_delta_ge_0.50"] == pytest.approx(0.50)
    assert method["fraction_abs_delta_ge_0.75"] == pytest.approx(0.50)
    assert method["mean_signed_delta"] == pytest.approx(-0.25)
    assert skew["top_extreme_contexts"][0]["context_id"] == "ctx2"
    assert skew["top_extreme_contexts"][0]["signed_gamma_delta"] == pytest.approx(-0.75)
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/evaluation/anchor_validation/test_summary.py::test_summary_reports_signed_balance_skew_distribution -v
```

Expected: FAIL because `balance_skew` is missing.

- [ ] **Step 3: Implement balance skew rows**

In `summary.py`, add helper:

```python
def _balance_skew(rows: list[RegionMetricRow]) -> dict[str, Any]:
    keyed_rows = [
        row for row in rows
        if row.key_id is not None and row.gamma is not None and row.valid_hit_rate is not None
    ]
    detail_rows = [_balance_skew_detail(row) for row in keyed_rows]
    return {
        "by_method": _balance_skew_group(detail_rows, lambda row: row["method"]),
        "by_gamma": _balance_skew_group(detail_rows, lambda row: f"{row['target_gamma']:g}"),
        "by_key_id": _balance_skew_group(detail_rows, lambda row: row["key_id"]),
        "by_node_type": _balance_skew_group(detail_rows, lambda row: row.get("node_type") or "unknown"),
        "by_block_ordinal_bucket": _balance_skew_group(
            detail_rows,
            lambda row: _block_ordinal_bucket(row.get("block_ordinal")) or "unknown",
        ),
        "by_candidate_count_bucket": _balance_skew_group(
            detail_rows,
            lambda row: _candidate_count_bucket(int(row["candidate_count"])),
        ),
        "top_extreme_contexts": sorted(
            detail_rows,
            key=lambda row: row["abs_gamma_delta"],
            reverse=True,
        )[:10],
    }
```

Add detail conversion:

```python
def _balance_skew_detail(row: RegionMetricRow) -> dict[str, Any]:
    empirical = float(row.valid_hit_rate or 0.0)
    target = float(row.gamma or 0.0)
    signed = empirical - target
    return {
        "context_id": row.context_id,
        "dataset": row.dataset,
        "task_id": row.task_id,
        "method": row.method,
        "target_gamma": target,
        "empirical_hit_rate": empirical,
        "signed_gamma_delta": signed,
        "abs_gamma_delta": abs(signed),
        "candidate_count": float(row.candidate_count),
        "node_type": row.node_type,
        "block_ordinal": row.block_ordinal,
        "key_id": row.key_id,
        "projection_key_id": row.projection_key_id,
    }
```

Add distribution helper:

```python
def _balance_skew_group(
    detail_rows: list[dict[str, Any]],
    key_fn,
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        grouped[str(key_fn(row))].append(row)
    return {
        key: _balance_skew_stats(rows)
        for key, rows in sorted(grouped.items())
    }
```

Add stats:

```python
def _balance_skew_stats(rows: list[dict[str, Any]]) -> dict[str, float]:
    abs_values = sorted(float(row["abs_gamma_delta"]) for row in rows)
    signed_values = [float(row["signed_gamma_delta"]) for row in rows]
    return {
        "row_count": float(len(rows)),
        "mean_abs_delta": _mean(abs_values),
        "median_abs_delta": _percentile_nearest(abs_values, 0.50),
        "p90_abs_delta": _percentile_nearest(abs_values, 0.90),
        "p95_abs_delta": _percentile_nearest(abs_values, 0.95),
        "fraction_abs_delta_ge_0.50": _mean([1.0 if value >= 0.50 else 0.0 for value in abs_values]),
        "fraction_abs_delta_ge_0.75": _mean([1.0 if value >= 0.75 else 0.0 for value in abs_values]),
        "mean_signed_delta": _mean(signed_values),
    }
```

Add percentile:

```python
def _percentile_nearest(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, round(q * (len(values) - 1))))
    return values[index]
```

Wire into `build_anchor_diagnostics`:

```python
        "balance_skew": _balance_skew(rows),
```

- [ ] **Step 4: Run tests**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/evaluation/anchor_validation/test_summary.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add wfcllm/evaluation/anchor_validation/summary.py tests/evaluation/anchor_validation/test_summary.py
git commit -m "fix: add balance skew diagnostics"
```

## Task 3: Add Two CodeT5 Comment Anchor Variants

**Files:**
- Modify: `wfcllm/evaluation/anchor_validation/schema.py`
- Modify: `wfcllm/evaluation/anchor_validation/anchors.py`
- Modify: `wfcllm/evaluation/anchor_validation/runner.py`
- Test: `tests/evaluation/anchor_validation/test_anchors.py`
- Test: `tests/evaluation/anchor_validation/test_runner.py`
- Test: `tests/integration/test_anchor_validation_cli.py`

- [ ] **Step 1: Write failing anchor tests**

Add to `tests/evaluation/anchor_validation/test_anchors.py`:

```python
def test_codet5_comment_minimal_is_short_parseable_and_secret_free():
    context = _context_for_node("return_statement", "return x + 1")
    text = build_anchor_text(
        AnchorMethod.CODET5_COMMENT_MINIMAL,
        context,
        context.candidates[0],
        secret_key="do-not-leak",
    )

    assert "do-not-leak" not in text
    assert "ctxhash" not in text
    assert "return x + 1" not in text
    assert "# wfcllm:" in text
    assert "return_statement" in text
    assert "ordinal_3" in text
    assert "return None" in text
    _assert_parseable_anchor(text)
```

Add contextual test:

```python
def test_codet5_comment_contextual_keeps_context_after_parseable():
    context = _context_for_node(
        "expression_statement",
        "total += x",
        context_before="def f(x):\n    total = 0\n",
        context_after="    return total\n",
    )
    text = build_anchor_text(
        AnchorMethod.CODET5_COMMENT_CONTEXTUAL,
        context,
        context.candidates[0],
        secret_key="do-not-leak",
    )

    assert "do-not-leak" not in text
    assert "total += x" not in text
    assert "def f(x):" in text
    assert "# wfcllm:" in text
    assert "_ = None" in text
    assert "return total" in text
    assert text.index("# wfcllm:") < text.index("_ = None")
    assert text.index("_ = None") < text.index("return total")
    _assert_parseable_anchor(text)
```

Add import-from coverage:

```python
def test_new_codet5_comment_variants_handle_import_from_statement():
    context = _context_for_node(
        "import_from_statement",
        "from .utils import helper",
        context_before="",
        context_after="",
        parent_node_type="module",
    )

    for method in (
        AnchorMethod.CODET5_COMMENT_MINIMAL,
        AnchorMethod.CODET5_COMMENT_CONTEXTUAL,
    ):
        text = build_anchor_text(method, context, context.candidates[0])
        assert "from .utils import helper" in text
        _assert_parseable_anchor(text)
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/evaluation/anchor_validation/test_anchors.py -v
```

Expected: FAIL because enum values do not exist.

- [ ] **Step 3: Add enum values**

In `wfcllm/evaluation/anchor_validation/schema.py`:

```python
    CODET5_COMMENT_MINIMAL = "codet5_comment_minimal"
    CODET5_COMMENT_CONTEXTUAL = "codet5_comment_contextual"
```

- [ ] **Step 4: Implement builders in anchors.py**

Add the methods to `_CODET5_CANDIDATE_ANCHOR_METHODS`:

```python
    AnchorMethod.CODET5_COMMENT_MINIMAL,
    AnchorMethod.CODET5_COMMENT_CONTEXTUAL,
```

Add builders:

```python
def _build_codet5_comment_minimal_anchor(
    context: CandidateContext,
    candidate: CandidateBlock,
) -> str:
    skeleton = _skeleton_for_node(context.node_type, candidate.block_text)
    metadata = (
        "# wfcllm: "
        f"{_metadata_token(context.node_type)} "
        f"ordinal_{context.block_ordinal}"
    )
    return _format_code_lines(
        [
            *_context_prefix_lines(context),
            _indent_like_context(context.context_before, metadata),
            _indent_like_context(context.context_before, skeleton),
        ]
    ).strip()
```

Add contextual builder:

```python
def _build_codet5_comment_contextual_anchor(
    context: CandidateContext,
    candidate: CandidateBlock,
) -> str:
    skeleton = _skeleton_for_node(context.node_type, candidate.block_text)
    metadata = (
        "# wfcllm: "
        f"{_metadata_token(context.node_type)} "
        f"ordinal_{context.block_ordinal} "
        f"parent_{_metadata_token(context.parent_node_type)}"
    )
    return _format_code_lines(
        [
            *_context_prefix_lines(context),
            _indent_like_context(context.context_before, metadata),
            _indent_like_context(context.context_before, skeleton),
            *_nonempty_lines((context.context_after,)),
        ]
    ).strip()
```

Wire dispatch before legacy methods:

```python
    if method == AnchorMethod.CODET5_COMMENT_MINIMAL:
        return _build_codet5_comment_minimal_anchor(context, candidate)
    if method == AnchorMethod.CODET5_COMMENT_CONTEXTUAL:
        return _build_codet5_comment_contextual_anchor(context, candidate)
```

- [ ] **Step 5: Update runner cache granularity**

In `_anchor_cache_candidate_id`, add:

```python
        AnchorMethod.CODET5_COMMENT_MINIMAL,
        AnchorMethod.CODET5_COMMENT_CONTEXTUAL,
```

This prevents candidate-specific skeleton anchors from being reused across candidates.

- [ ] **Step 6: Add CLI integration coverage**

In `tests/integration/test_anchor_validation_cli.py`, add both methods to a run:

```python
"codet5_comment_minimal",
"codet5_comment_contextual",
```

Assert command returns 0 and summary exists.

- [ ] **Step 7: Run targeted tests**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/evaluation/anchor_validation/test_anchors.py tests/evaluation/anchor_validation/test_runner.py tests/integration/test_anchor_validation_cli.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add wfcllm/evaluation/anchor_validation/schema.py wfcllm/evaluation/anchor_validation/anchors.py wfcllm/evaluation/anchor_validation/runner.py tests/evaluation/anchor_validation/test_anchors.py tests/evaluation/anchor_validation/test_runner.py tests/integration/test_anchor_validation_cli.py
git commit -m "feat: add codet5 comment anchor variants"
```

## Task 4: Add R003.5 Runbook Commands

**Files:**
- Modify: `docs/experiment/anchor-validation-runbook.md`

- [ ] **Step 1: Add R003.5 section**

Append this section to `docs/experiment/anchor-validation-runbook.md`:

```markdown
## R003.5 CodeT5 Primary Anchor and Keying Balance Diagnostic

R003.5 uses the existing R002 candidate pool. Do not regenerate candidates.

The objective is diagnostic, not production go/no-go:

- `codet5_comment_anchor` is the primary text anchor.
- `candidate_centroid_oracle` and `context_centroid_oracle` measure headroom.
- Run A uses default ordinal keying.
- Run B uses `--legacy-parent-keying`.
- `first_stage_passed` remains constrained by the data-quality gate.

### R003.5A: Ordinal Keying

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
/root/miniconda3/bin/conda run -n WFCLLM python scripts/anchor_validation.py run-diagnostics \
  --pool data/diagnostics/anchor_validation/candidate_pools_r002.jsonl \
  --output-dir data/diagnostics/anchor_validation/encoder_r0035_codet5_primary_ordinal \
  --embedding-mode encoder \
  --encoder-model-path data/models/codet5-base \
  --encoder-checkpoint data/models/encoder/best_model.pt \
  --encoder-device cuda \
  --embed-dim 128 \
  --lsh-d 3 \
  --secret-key anchor-key-00 anchor-key-01 anchor-key-02 anchor-key-03 anchor-key-04 \
  --methods vanilla random seqmark_oracle context_centroid_oracle candidate_centroid_oracle role_aware_slot_context role_aware_slot_context_skeleton codet5_comment_anchor codet5_comment_minimal codet5_comment_contextual \
  --primary-method codet5_comment_anchor \
  --gammas 0.25 0.5 0.75 \
  --retry-budgets 1 4 8
```

### R003.5B: Parent-Only Keying

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
/root/miniconda3/bin/conda run -n WFCLLM python scripts/anchor_validation.py run-diagnostics \
  --pool data/diagnostics/anchor_validation/candidate_pools_r002.jsonl \
  --output-dir data/diagnostics/anchor_validation/encoder_r0035_codet5_primary_parent_only \
  --embedding-mode encoder \
  --encoder-model-path data/models/codet5-base \
  --encoder-checkpoint data/models/encoder/best_model.pt \
  --encoder-device cuda \
  --embed-dim 128 \
  --lsh-d 3 \
  --secret-key anchor-key-00 anchor-key-01 anchor-key-02 anchor-key-03 anchor-key-04 \
  --methods vanilla random seqmark_oracle context_centroid_oracle candidate_centroid_oracle role_aware_slot_context role_aware_slot_context_skeleton codet5_comment_anchor codet5_comment_minimal codet5_comment_contextual \
  --primary-method codet5_comment_anchor \
  --legacy-parent-keying \
  --gammas 0.25 0.5 0.75 \
  --retry-budgets 1 4 8
```

### Interpretation

Treat R003.5 as diagnostic even if primary-method gates improve. The R002 pool has
39 contexts and 3 node-type categories, so the production data-quality gate should
still block end-to-end AO-LSH follow-up.

Use `go_no_go.primary_method_evidence`, `go_no_go.primary_method_gates`, and
`anchor_diagnostics.balance_skew` to compare ordinal versus parent-only keying.
Prefer mean/median/p90/fraction-extreme skew over a single `max_delta_gamma`.
```

- [ ] **Step 2: Verify docs diff**

Run:

```bash
git diff -- docs/experiment/anchor-validation-runbook.md
```

Expected: R003.5 section only.

- [ ] **Step 3: Commit**

```bash
git add docs/experiment/anchor-validation-runbook.md
git commit -m "docs: add r0035 anchor validation runbook"
```

## Task 5: Final Local Verification

**Files:**
- Verify all modified files.

- [ ] **Step 1: Run anchor validation tests**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/evaluation/anchor_validation/ tests/integration/test_anchor_validation_cli.py -v
```

Expected: PASS.

- [ ] **Step 2: Run compatibility tests**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_lsh_space.py tests/watermark/test_verifier.py tests/extract/test_scorer.py -v
```

Expected: PASS.

- [ ] **Step 3: Run compileall**

Run:

```bash
conda run -n WFCLLM python -m compileall wfcllm run.py scripts tools
```

Expected: PASS.

- [ ] **Step 4: Check diff**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors. Only intended tracked files modified; local diagnostic artifacts remain untracked and uncommitted.

## Task 6: Remote R003.5 Execution

**Files:**
- Remote outputs only under `~/autodl-tmp/WFCLLM/data/diagnostics/anchor_validation/`.

- [ ] **Step 1: Push branch**

Run locally after commits:

```bash
git push -u origin codex-anchor-effectiveness-validation
```

Expected: branch updates on GitHub.

- [ ] **Step 2: Pull on the new server**

Use the current server:

```bash
ssh -p 16311 root@connect.bjb2.seetacloud.com "cd ~/autodl-tmp/WFCLLM && git pull --ff-only"
```

Expected: fast-forward to the new commit.

- [ ] **Step 3: Run R003.5A ordinal keying**

Run:

```bash
ssh -p 16311 root@connect.bjb2.seetacloud.com "screen -dmS anchor_r0035_ordinal bash -lc 'cd ~/autodl-tmp/WFCLLM && export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1; /root/miniconda3/bin/conda run -n WFCLLM python scripts/anchor_validation.py run-diagnostics --pool data/diagnostics/anchor_validation/candidate_pools_r002.jsonl --output-dir data/diagnostics/anchor_validation/encoder_r0035_codet5_primary_ordinal --embedding-mode encoder --encoder-model-path data/models/codet5-base --encoder-checkpoint data/models/encoder/best_model.pt --encoder-device cuda --embed-dim 128 --lsh-d 3 --secret-key anchor-key-00 anchor-key-01 anchor-key-02 anchor-key-03 anchor-key-04 --methods vanilla random seqmark_oracle context_centroid_oracle candidate_centroid_oracle role_aware_slot_context role_aware_slot_context_skeleton codet5_comment_anchor codet5_comment_minimal codet5_comment_contextual --primary-method codet5_comment_anchor --gammas 0.25 0.5 0.75 --retry-budgets 1 4 8 2>&1 | tee data/diagnostics/anchor_validation/encoder_r0035_codet5_primary_ordinal.log'"
```

Expected: detached screen session starts.

- [ ] **Step 4: Run R003.5B parent-only keying after A completes**

Run:

```bash
ssh -p 16311 root@connect.bjb2.seetacloud.com "screen -dmS anchor_r0035_parent bash -lc 'cd ~/autodl-tmp/WFCLLM && export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1; /root/miniconda3/bin/conda run -n WFCLLM python scripts/anchor_validation.py run-diagnostics --pool data/diagnostics/anchor_validation/candidate_pools_r002.jsonl --output-dir data/diagnostics/anchor_validation/encoder_r0035_codet5_primary_parent_only --embedding-mode encoder --encoder-model-path data/models/codet5-base --encoder-checkpoint data/models/encoder/best_model.pt --encoder-device cuda --embed-dim 128 --lsh-d 3 --secret-key anchor-key-00 anchor-key-01 anchor-key-02 anchor-key-03 anchor-key-04 --methods vanilla random seqmark_oracle context_centroid_oracle candidate_centroid_oracle role_aware_slot_context role_aware_slot_context_skeleton codet5_comment_anchor codet5_comment_minimal codet5_comment_contextual --primary-method codet5_comment_anchor --legacy-parent-keying --gammas 0.25 0.5 0.75 --retry-budgets 1 4 8 2>&1 | tee data/diagnostics/anchor_validation/encoder_r0035_codet5_primary_parent_only.log'"
```

Expected: detached screen session starts.

- [ ] **Step 5: Summarize R003.5 outputs**

Run:

```bash
ssh -p 16311 root@connect.bjb2.seetacloud.com "cd ~/autodl-tmp/WFCLLM && /root/miniconda3/bin/conda run -n WFCLLM python -c 'exec(\"\"\"import json
for run in [\"encoder_r0035_codet5_primary_ordinal\", \"encoder_r0035_codet5_primary_parent_only\"]:
    p = f\"data/diagnostics/anchor_validation/{run}/anchor_validation_summary.json\"
    s = json.load(open(p, encoding=\"utf-8\"))
    print(\"RUN\", run)
    print(\"first_stage_passed\", s[\"go_no_go\"][\"first_stage_passed\"])
    print(\"primary_method\", s[\"go_no_go\"].get(\"primary_method\"))
    print(\"primary_method_evidence\", json.dumps(s[\"go_no_go\"].get(\"primary_method_evidence\", {}), ensure_ascii=False, indent=2)[:2000])
    print(\"primary_method_gates\", json.dumps(s[\"go_no_go\"].get(\"primary_method_gates\", {}), ensure_ascii=False, indent=2))
    skew = s[\"anchor_diagnostics\"].get(\"balance_skew\", {})
    print(\"balance by method\", json.dumps(skew.get(\"by_method\", {}), ensure_ascii=False, indent=2)[:2500])
    print()
\"\"\")'"
```

Expected: ordinal and parent-only summaries print primary evidence and balance skew comparison.

## Success Criteria for This Plan

This plan is complete when:

- `--primary-method` works and validates missing methods/baselines.
- Production `first_stage_passed` remains data-quality gated.
- `go_no_go.primary_method_evidence` and `primary_method_gates` are present.
- `anchor_diagnostics.balance_skew` includes signed delta and distribution metrics.
- `codet5_comment_minimal` and `codet5_comment_contextual` run through CLI and tests.
- Local targeted tests and compileall pass.
- R003.5 ordinal and parent-only runs complete on the new server.
- The final analysis separates:
  - CodeT5 text-anchor signal,
  - valid-set/keying/gamma balance,
  - candidate-pool coverage,
  - encoder/oracle headroom.

