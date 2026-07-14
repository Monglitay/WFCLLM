from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from wfcllm.diagnostics.candidate_probe_v4 import (
    CandidateRecord,
    ProbeSecret,
    load_candidate_ledger,
    load_probe_contexts,
    canonical_structural_unit,
    derive_projection_bits,
    derive_target_bits,
    exact_evidence_mismatches,
    extract_structural_units,
    margin_replay,
    parse_serialized_context,
    probe_shape_isolated,
    score_structural_unit,
    score_structural_code,
    selection_capacity,
    structural_pool_capacity,
    summarize_margin_rows,
    validate_candidate_pool,
    write_public_probe_artifact,
)


def _record(task_id: str, attempt_index: int, code: str | None = None) -> CandidateRecord:
    final_code = code or f"def f_{task_id.replace('/', '_')}():\n    return {attempt_index}\n"
    return CandidateRecord(
        task_id=task_id,
        attempt_index=attempt_index,
        final_code=final_code,
        final_code_sha256=hashlib.sha256(final_code.encode("utf-8")).hexdigest(),
    )


def test_canonical_structural_unit_normalizes_formatting_and_comments() -> None:
    left = canonical_structural_unit(
        role="Assign|FunctionDef|body",
        previous="value = source + 1",
        current="result = value * 2",
    )
    right = canonical_structural_unit(
        role="Assign|FunctionDef|body",
        previous="value=source+1  # same program",
        current="result=value*2",
    )

    assert left.representation == right.representation
    assert left.representation_sha256 == right.representation_sha256


def test_canonical_structural_unit_normalizes_scoped_identifier_rename() -> None:
    left = canonical_structural_unit(
        role="Return|FunctionDef|body",
        previous="total = left_value + right_value",
        current="return total / 2",
    )
    right = canonical_structural_unit(
        role="Return|FunctionDef|body",
        previous="sum_value = first + second",
        current="return sum_value / 2",
    )

    assert left.representation == right.representation
    assert left.unit_id == right.unit_id


def test_canonical_structural_unit_preserves_semantic_operator_difference() -> None:
    add = canonical_structural_unit(
        role="Return|FunctionDef|body",
        previous="<BOS>",
        current="return left + right",
    )
    subtract = canonical_structural_unit(
        role="Return|FunctionDef|body",
        previous="<BOS>",
        current="return left - right",
    )

    assert add.representation != subtract.representation
    assert add.unit_id != subtract.unit_id


def test_canonical_structural_unit_rejects_global_ordinal_and_parse_failure() -> None:
    with pytest.raises(TypeError, match="global_ordinal"):
        canonical_structural_unit(  # type: ignore[call-arg]
            role="Return|FunctionDef|body",
            previous="<BOS>",
            current="return 1",
            global_ordinal=7,
        )
    with pytest.raises(ValueError, match="parse"):
        canonical_structural_unit(
            role="Return|FunctionDef|body",
            previous="<BOS>",
            current="return (",
        )


def test_projection_and_target_use_distinct_v4_domains() -> None:
    secret = ProbeSecret.from_material_for_test(b"p" * 32)
    message = b"same-public-message"

    projection = derive_projection_bits(secret, message, bit_count=32)
    target = derive_target_bits(secret, message, bit_count=32)

    assert projection != target
    assert repr(secret) == "ProbeSecret(<redacted>)"
    assert len(projection) == len(target) == 32


def test_structural_evidence_contains_all_strict_discrete_fields() -> None:
    secret = ProbeSecret.from_material_for_test(b"e" * 32)
    unit = canonical_structural_unit(
        role="Return|FunctionDef|body",
        previous="value = source + 1",
        current="return value",
    )

    evidence = score_structural_unit(unit, secret, bit_count=16)

    assert evidence.unit_id == unit.unit_id
    assert evidence.context_sha256 == unit.context_sha256
    assert evidence.representation == unit.representation_bytes
    assert evidence.quantized_values == unit.representation_bytes
    assert evidence.erasure_mask == (False,) * 16
    assert evidence.matches == sum(
        left == right
        for left, right in zip(evidence.signature_bits, evidence.target_bits, strict=True)
    )
    assert evidence.numerator == 2 * evidence.matches - evidence.denominator


def test_exact_evidence_mismatches_names_erasure_and_representation_fields() -> None:
    secret = ProbeSecret.from_material_for_test(b"m" * 32)
    unit = canonical_structural_unit(
        role="Return|FunctionDef|body",
        previous="<BOS>",
        current="return 1",
    )
    evidence = score_structural_unit(unit, secret, bit_count=8)
    changed_mask = evidence.with_erasure_mask((True,) + evidence.erasure_mask[1:])
    changed_representation = evidence.with_representation(
        (evidence.representation[0] ^ 1,) + evidence.representation[1:]
    )

    assert exact_evidence_mismatches(evidence, evidence) == ()
    assert exact_evidence_mismatches(evidence, changed_mask) == ("erasure_mask",)
    assert exact_evidence_mismatches(evidence, changed_representation) == (
        "representation",
        "quantized_values",
    )


def test_margin_replay_erases_ties_and_requires_exact_masks() -> None:
    replay = margin_replay(
        reference_dots=(31, 30, -31, -30, 100),
        candidate_dots=(32, 31, -32, -29, -100),
        absolute_dot_bound=30,
    )

    assert replay.reference_erasure_mask == (False, True, False, True, False)
    assert replay.candidate_erasure_mask == (False, False, False, True, False)
    assert replay.erasure_mask_mismatch_count == 1
    assert replay.signature_mismatch_count == 1
    assert replay.exact is False


def test_candidate_pool_requires_exact_retry_cardinality_order_and_hash() -> None:
    records = tuple(_record("HumanEval/0", index) for index in range(20))

    assert validate_candidate_pool(records, retry=20) == records
    with pytest.raises(ValueError, match="exactly 20"):
        validate_candidate_pool(records[:-1], retry=20)
    with pytest.raises(ValueError, match="attempt indices"):
        validate_candidate_pool(records[:-1] + (_record("HumanEval/0", 21),), retry=20)
    malformed = CandidateRecord(
        task_id="HumanEval/0",
        attempt_index=0,
        final_code="return 1",
        final_code_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="SHA-256"):
        validate_candidate_pool((malformed,), retry=1)


def test_selection_capacity_preserves_pool_and_reports_outlier_share() -> None:
    records = tuple(
        _record(task_id, index)
        for task_id in ("HumanEval/0", "HumanEval/1")
        for index in range(20)
    )

    summary = selection_capacity(
        records,
        retry=20,
        score=lambda record: float(
            record.attempt_index
            if record.task_id.endswith("0")
            else int(record.attempt_index == 1)
        ),
    )

    assert summary.task_count == 2
    assert summary.candidate_count == 40
    assert summary.positive_delta_tasks == 2
    assert summary.mean_delta == pytest.approx(10.0)
    assert summary.maximum_delta_share == pytest.approx(0.95)
    assert summary.selected_attempts == (19, 1)


def test_serialized_context_parser_and_loader_verify_public_hash(tmp_path: Path) -> None:
    serialized = (
        "WFCLLM_DYNAMIC_SEMANTIC_CONTEXT_V3\n"
        "role=Return|FunctionDef|body\n"
        "previous=value = source + 1\n"
        "current=return value"
    )
    role, previous, current = parse_serialized_context(serialized)
    assert (role, previous, current) == (
        "Return|FunctionDef|body",
        "value = source + 1",
        "return value",
    )
    context_path = tmp_path / "contexts.jsonl"
    row = {
        "case_id": "control-0",
        "category": "control",
        "context_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "role": role,
        "serialized": serialized,
        "token_count": 7,
    }
    context_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    contexts = load_probe_contexts(context_path)

    assert len(contexts) == 1
    assert contexts[0].case_id == "control-0"
    assert contexts[0].previous == previous
    assert contexts[0].current == current
    row["context_sha256"] = "0" * 64
    context_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="context SHA-256"):
        load_probe_contexts(context_path)


def test_candidate_ledger_loader_keeps_only_final_code_contract(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.jsonl"
    rows = []
    for index in range(2):
        record = _record("HumanEval/0", index)
        rows.append(
            {
                "id": record.task_id,
                "attempt_index": record.attempt_index,
                "final_code": record.final_code,
                "final_code_sha256": record.final_code_sha256,
                "prompt": "must not enter returned probe records",
                "unit_ids": ["generation-ledger-must-not-enter"],
            }
        )
    ledger_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    records = load_candidate_ledger(
        ledger_path,
        retry=2,
        allowed_task_ids=("HumanEval/0",),
    )

    assert records == tuple(_record("HumanEval/0", index) for index in range(2))
    assert not hasattr(records[0], "prompt")
    with pytest.raises(ValueError, match="allowed task IDs"):
        load_candidate_ledger(
            ledger_path,
            retry=2,
            allowed_task_ids=("HumanEval/1",),
        )


def test_margin_summary_recomputes_masks_from_raw_projection_dots() -> None:
    rows = (
        {
            "condition": {"condition_id": "reference-000"},
            "discrete": {
                "reference_projection_dots": [100, -100, 30],
                "candidate_projection_dots": [99, -101, 31],
            },
        },
        {
            "condition": {"condition_id": "batch-size-002"},
            "discrete": {
                "reference_projection_dots": [100, -100, 30],
                "candidate_projection_dots": [-100, -99, 30],
            },
        },
    )

    summary = summarize_margin_rows(rows, absolute_dot_bound=30)

    assert summary.row_count == 2
    assert summary.bit_count == 6
    assert summary.erased_reference_bits == 2
    assert summary.erasure_mask_mismatch_rows == 1
    assert summary.signature_mismatch_rows == 1
    assert summary.exact_rows == 0


def test_public_probe_writer_rejects_key_confirmation_metadata(tmp_path: Path) -> None:
    output_path = tmp_path / "probe.json"
    write_public_probe_artifact(
        output_path,
        {
            "artifact_type": "candidate_probe",
            "encoder_checkpoint_sha256": "a" * 64,
            "secret_metadata_included": False,
            "candidate_results": [],
        },
    )
    parsed = json.loads(output_path.read_text(encoding="utf-8"))
    assert parsed["schema_version"] == "wfcllm-v4-candidate-probe/v1"
    assert parsed["secret_metadata_included"] is False
    with pytest.raises(ValueError, match="forbidden secret metadata"):
        write_public_probe_artifact(output_path, {"key_fingerprint": "confirming-value"})


def test_candidate_probe_cli_exposes_required_frozen_inputs() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/wfcllm_v4_candidate_probe.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--contexts" in completed.stdout
    assert "--root-cause-matrix" in completed.stdout
    assert "--candidate-ledger" in completed.stdout
    assert "--diagnostic-key-file" in completed.stdout
    assert "--output" in completed.stdout


def test_structural_extraction_handles_function_arguments_and_nested_statements() -> None:
    code = """
def total(values):
    running = 0
    for value in values:
        if value > 0:
            running += value
    return running
"""

    extraction = extract_structural_units(code)

    assert extraction.parse_ok is True
    assert extraction.erasure_counts == {}
    assert len(extraction.units) == 5
    assert len({unit.unit_id for unit in extraction.units}) == 5
    roles = {unit.role for unit in extraction.units}
    assert "For|FunctionDef|body" in roles
    assert "If|For|body" in roles
    assert "AugAssign|If|body" in roles
    compound_units = {
        unit.role: unit.representation for unit in extraction.units if unit.role.startswith(("For|", "If|"))
    }
    assert "AugAssign" not in compound_units["For|FunctionDef|body"]
    assert "AugAssign" not in compound_units["If|For|body"]
    malformed = extract_structural_units("def broken(:\n")
    assert malformed.parse_ok is False
    assert malformed.units == ()
    assert malformed.erasure_counts == {"parse_failure": 1}


def test_structural_code_score_is_final_code_only_and_replays_exactly() -> None:
    secret = ProbeSecret.from_material_for_test(b"s" * 32)
    left = score_structural_code(
        "def f(value):\n    result = value + 1\n    return result\n",
        secret,
        bit_count=16,
        minimum_independent_units=2,
    )
    right = score_structural_code(
        "def f(source): # renamed\n    output=source+1\n    return output\n",
        secret,
        bit_count=16,
        minimum_independent_units=2,
    )

    assert left.evidence == right.evidence
    assert left.numerator == right.numerator
    assert left.denominator == right.denominator
    assert left.eligible is True
    assert left.exact_mismatches(right) == ()


def test_structural_pool_probe_preserves_all_raw_candidates() -> None:
    records = tuple(
        _record(task_id, index)
        for task_id in ("HumanEval/0", "HumanEval/1")
        for index in range(2)
    )
    secret = ProbeSecret.from_material_for_test(b"c" * 32)

    result = structural_pool_capacity(
        records,
        secret,
        retry=2,
        bit_count=8,
        minimum_independent_units=1,
    )

    assert result.capacity.task_count == 2
    assert result.capacity.candidate_count == 4
    assert result.input_pool_sha256 == result.output_pool_sha256
    assert result.candidate_pool_match_rate == 1.0
    assert result.r3_input_fields == ("final_code",)
    assert result.eligible_task_count == 2


def test_candidate_probe_cli_runs_public_b_c_d_probe_without_neural_runtime(
    tmp_path: Path,
) -> None:
    serialized = (
        "WFCLLM_DYNAMIC_SEMANTIC_CONTEXT_V3\n"
        "role=Return|FunctionDef|body\n"
        "previous=<BOS>\n"
        "current=return 1"
    )
    contexts_path = tmp_path / "contexts.jsonl"
    contexts_path.write_text(
        json.dumps(
            {
                "case_id": "synthetic-0",
                "category": "synthetic",
                "context_sha256": hashlib.sha256(
                    serialized.encode("utf-8")
                ).hexdigest(),
                "role": "Return|FunctionDef|body",
                "serialized": serialized,
                "token_count": 4,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(
        json.dumps(
            {
                "maximum_observed_absolute_difference_by_stage": {
                    "projection_dots": 30
                },
                "bound_scope": "empirical_only",
            }
        ),
        encoding="utf-8",
    )
    margin_path = tmp_path / "margin.jsonl"
    margin_path.write_text(
        json.dumps(
            {
                "condition": {"condition_id": "batch-size-002"},
                "discrete": {
                    "reference_projection_dots": [100, -100, 30],
                    "candidate_projection_dots": [99, -101, 31],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    ledger_path = tmp_path / "ledger.jsonl"
    records = tuple(_record("HumanEval/0", index) for index in range(2))
    ledger_path.write_text(
        "".join(
            json.dumps(
                {
                    "id": item.task_id,
                    "attempt_index": item.attempt_index,
                    "final_code": item.final_code,
                    "final_code_sha256": item.final_code_sha256,
                }
            )
            + "\n"
            for item in records
        ),
        encoding="utf-8",
    )
    key_path = tmp_path / "diagnostic.key"
    key_path.write_bytes(b"k" * 32)
    key_path.chmod(0o600)
    output_path = tmp_path / "probe.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/wfcllm_v4_candidate_probe.py",
            "--contexts",
            str(contexts_path),
            "--root-cause-matrix",
            str(matrix_path),
            "--margin-rows",
            str(margin_path),
            "--candidate-ledger",
            str(ledger_path),
            "--diagnostic-key-file",
            str(key_path),
            "--encoder-model",
            str(tmp_path / "unused-model"),
            "--whitening",
            str(tmp_path / "unused-whitening.json"),
            "--task-id",
            "HumanEval/0",
            "--retry",
            "2",
            "--skip-neural",
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    assert artifact["root_cause_bound_scope"] == "empirical_only"
    assert artifact["candidate_results"]["A"]["status"] == "skipped_by_cli"
    assert artifact["candidate_results"]["B"]["absolute_dot_bound"] == 30
    assert artifact["candidate_results"]["C"]["candidate_pool_match_rate"] == 1.0
    assert artifact["candidate_results"]["D"]["signed_evidence_identical_to_C"] is True
    assert artifact["secret_metadata_included"] is False


def test_shape_isolated_probe_requires_bit_exact_replay_and_cache_identity() -> None:
    calls: list[str] = []

    def encode_one(serialized: str) -> tuple[int, ...]:
        calls.append(serialized)
        digest = hashlib.sha256(serialized.encode("utf-8")).digest()
        return tuple(digest[:8])

    summary = probe_shape_isolated(
        ("context-a", "context-b", "context-c"),
        encode_one=encode_one,
    )

    assert summary.context_count == 3
    assert summary.schedule_names == ("forward", "reverse", "permutation")
    assert summary.total_replays == 9
    assert summary.exact_replays == 9
    assert summary.exact_replay_rate == 1.0
    assert summary.cache_hit_miss_exact is True
    assert summary.physical_encode_calls == 12
    assert len(calls) == 12
    assert summary.mean_cache_miss_seconds_per_context >= 0
    assert summary.mean_cache_hit_seconds_per_context >= 0
