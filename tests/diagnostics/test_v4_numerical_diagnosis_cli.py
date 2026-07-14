from __future__ import annotations

import hashlib

import torch

from scripts import wfcllm_v4_numerical_diagnosis as diagnosis
from wfcllm.diagnostics.numerical_replay_v4 import ContextCase, tensor_delta


def _case(case_id: str, role: str, tokens: int) -> ContextCase:
    serialized = f"context::{case_id}::{role}::{tokens}"
    return ContextCase(
        case_id=case_id,
        serialized=serialized,
        context_sha256=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        role=role,
        token_count=tokens,
        category="control",
    )


def test_parser_exposes_prepare_run_and_summarize_subcommands() -> None:
    parser = diagnosis.build_parser()

    prepare = parser.parse_args(
        [
            "prepare",
            "--public-config",
            "public.json",
            "--whitening",
            "whitening.json",
            "--v3-experiment-dir",
            "v3",
            "--diagnostic-key-file",
            "diagnostic.key",
            "--output-dir",
            "out",
        ]
    )
    run = parser.parse_args(
        [
            "run",
            "--public-config",
            "public.json",
            "--whitening",
            "whitening.json",
            "--contexts",
            "contexts.jsonl",
            "--diagnostic-key-file",
            "diagnostic.key",
            "--output-jsonl",
            "raw.jsonl",
            "--restart-id",
            "cold-0",
        ]
    )
    summarize = parser.parse_args(
        [
            "summarize",
            "--contexts",
            "contexts.jsonl",
            "--input-jsonl",
            "raw0.jsonl",
            "--input-jsonl",
            "raw1.jsonl",
            "--output-json",
            "matrix.json",
        ]
    )

    assert prepare.command == "prepare"
    assert run.command == "run"
    assert run.profile == "full"
    assert run.device == "cuda"
    assert summarize.command == "summarize"
    assert summarize.input_jsonl == ["raw0.jsonl", "raw1.jsonl"]


def test_synthetic_specs_cover_at_least_twenty_ast_and_length_cases() -> None:
    specs = diagnosis.synthetic_context_specs()

    assert len(specs) >= 20
    assert len({item.case_id for item in specs}) == len(specs)
    assert len({item.expected_role_prefix for item in specs}) >= 12
    assert len({len(item.code) for item in specs}) >= 12
    assert all("def " in item.code or "async def " in item.code for item in specs)


def test_control_selection_is_deterministic_and_role_length_diverse() -> None:
    candidates = tuple(
        _case(
            f"c-{role_index}-{length}",
            f"Role{role_index}|FunctionDef|body",
            length,
        )
        for role_index in range(6)
        for length in range(2, 42, 2)
    )

    selected = diagnosis.select_varied_controls(candidates, count=20)
    repeated = diagnosis.select_varied_controls(tuple(reversed(candidates)), count=20)

    assert selected == repeated
    assert len(selected) == 20
    assert len({item.role for item in selected}) == 6
    assert min(item.token_count for item in selected) == 2
    assert max(item.token_count for item in selected) == 40


def test_delta_serialization_keeps_mismatch_values_without_full_embeddings() -> None:
    reference = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    candidate = torch.tensor([1.0, 2.5, 3.0], dtype=torch.float32)

    payload = diagnosis.serialize_delta(
        tensor_delta(reference, candidate),
        reference=reference,
        candidate=candidate,
        mismatch_limit=8,
    )

    assert payload["mismatch_count"] == 1
    assert payload["mismatch_coordinates"] == [
        {"index": 1, "reference": 2.0, "candidate": 2.5}
    ]
    assert "reference_sha256" in payload
    assert "candidate_sha256" in payload
    assert "full_reference" not in payload
    assert "full_candidate" not in payload


def test_root_cause_classification_ignores_expected_masked_tail_input_change() -> None:
    trials = [
        {
            "condition": {"axis": "masked_tail"},
            "first_divergent_layer": "input_ids",
        },
        {
            "condition": {"axis": "batch_size"},
            "first_divergent_layer": "t5_block_00_layer_00_SelfAttention",
        },
    ]

    classification = diagnosis.classify_root_cause_trials(trials)

    assert classification == "A1-batch-dependent-encoder-supported"


def test_root_cause_classification_reports_encoder_and_downstream_dependence() -> None:
    trials = [
        {
            "condition": {"axis": "batch_size"},
            "first_divergent_layer": "t5_block_00_layer_00_SelfAttention",
            "downstream_isolated_first_divergent_layer": "whitening_pre_norm",
        }
    ]

    classification = diagnosis.classify_root_cause_trials(trials)

    assert classification == "A2-encoder-and-downstream-batch-dependence"


def test_condition_value_label_keeps_the_controlled_value_explicit() -> None:
    assert diagnosis.condition_value_label(
        {"axis": "batch_size", "batch_size": 8}
    ) == "batch_size=8"
    assert diagnosis.condition_value_label(
        {"axis": "composition", "batch_size": 8, "composition": "short_mix"}
    ) == "batch_size=8|composition=short_mix"
    assert diagnosis.condition_value_label(
        {"axis": "deterministic", "deterministic_algorithms": True}
    ) == "deterministic_algorithms=true"


def test_strict_candidate_fingerprint_covers_discrete_and_downstream_fields() -> None:
    row = {
        "stage_deltas": {"quantized": {"candidate_sha256": "a" * 64}},
        "downstream_isolated_stage_deltas": {
            "whitening_pre_norm": {"candidate_sha256": "b" * 64}
        },
        "discrete": {
            "candidate_quantized": [1, 2],
            "candidate_projection_dots": [3],
            "candidate_signature_bits": [1],
        },
    }

    fingerprint = diagnosis.strict_candidate_fingerprint(row)
    changed = {
        **row,
        "discrete": {**row["discrete"], "candidate_signature_bits": [0]},
    }

    assert fingerprint == diagnosis.strict_candidate_fingerprint(dict(row))
    assert fingerprint != diagnosis.strict_candidate_fingerprint(changed)
