"""Tests for extract-side block contract alignment."""

from __future__ import annotations

from dataclasses import asdict
import json

from wfcllm.common.block_contract import build_block_contracts
from wfcllm.extract.alignment import compare_block_contracts, rebuild_block_contracts
from wfcllm.watermark.adaptive_gamma.profile import EntropyProfile
from wfcllm.watermark.adaptive_gamma.schedule import PiecewiseQuantileSchedule


def _contract(
    *,
    ordinal: int,
    block_id: str,
    entropy_units: int = 100,
    k: int = 0,
) -> dict:
    return {
        "ordinal": ordinal,
        "block_id": block_id,
        "node_type": "expression_statement",
        "parent_node_type": "module",
        "block_text_hash": f"hash-{block_id}",
        "start_line": ordinal + 1,
        "end_line": ordinal + 1,
        "entropy_units": entropy_units,
        "gamma_target": 0.0,
        "k": k,
        "gamma_effective": 0.0,
    }


class TestCompareBlockContracts:
    def test_reports_block_count_mismatch(self):
        embedded = [_contract(ordinal=0, block_id="0"), _contract(ordinal=1, block_id="1")]
        rebuilt = [_contract(ordinal=0, block_id="0")]

        report = compare_block_contracts(embedded, rebuilt)

        assert report.block_count_mismatch is True
        assert report.structure_mismatch is True
        assert report.numeric_mismatch is False
        assert report.contract_valid is False
        assert report.status == "structure_mismatch"
        assert report.embedded_block_count == 2
        assert report.rebuilt_block_count == 1

    def test_reports_numeric_mismatch_without_structure_mismatch(self):
        embedded_contract = _contract(ordinal=0, block_id="0", entropy_units=101)
        rebuilt_contract = _contract(ordinal=0, block_id="0", entropy_units=100)

        report = compare_block_contracts([embedded_contract], [rebuilt_contract])

        assert report.block_count_mismatch is False
        assert report.structure_mismatch is False
        assert report.numeric_mismatch is True
        assert report.contract_valid is True
        assert report.status == "numeric_mismatch"
        assert len(report.numeric_mismatches) == 1
        mismatch = report.numeric_mismatches[0]
        assert mismatch["ordinal"] == 0
        assert mismatch["field"] == "entropy_units"
        assert mismatch["embedded"] == embedded_contract["entropy_units"]
        assert mismatch["rebuilt"] == rebuilt_contract["entropy_units"]


def test_alignment_ignores_parent_node_type_context_shift():
    """parent_node_type changes when generated_code is parsed without the prompt wrapper.

    HumanEval prompts end with an open function signature (def f(...):).
    During watermarking the parser sees prompt+generated_code, so the generated
    statements are children of function_definition.  During extraction only
    generated_code is parsed; the indented statements become top-level module
    children.  Alignment must not treat this context-dependent field as a
    structural mismatch.
    """
    # Prompt ends with open function body — generated code is indented inside it
    prompt_code = "def f(x):\n    pass\n"
    gen_code = "    result = x + 1\n    return result\n"

    full_contracts = [asdict(c) for c in build_block_contracts(prompt_code + gen_code)]
    gen_code_block_count = len(build_block_contracts(gen_code))
    embedded = full_contracts[-gen_code_block_count:]

    rebuilt = rebuild_block_contracts(gen_code)

    # embedded has parent_node_type=function_definition; rebuilt has parent_node_type=module
    assert any(b["parent_node_type"] == "function_definition" for b in embedded)
    assert any(b["parent_node_type"] == "module" for b in rebuilt)

    report = compare_block_contracts(embedded, rebuilt)

    assert not report.structure_mismatch, (
        f"structure_mismatch should be False but got: {report.structure_mismatches}"
    )
    assert report.contract_valid


def test_alignment_ignores_context_dependent_fields():
    """Blocks embedded in prompt+code context must align with blocks rebuilt from code-only.

    During watermarking the parser runs on lm_prompt+generated_code, so block_id,
    start_line, end_line, and ordinal are all offset by the prompt's blocks.
    During extraction the parser runs on generated_code only, so those fields
    restart from 0.  Alignment must not treat these position-dependent fields as
    structural mismatches.
    """
    prompt_code = "x = 1\ny = 2\n"   # 2 simple blocks → IDs "0","1" in full context
    gen_code = "z = 3\nw = 4\n"      # 2 simple blocks → IDs "2","3" in full context

    # Simulate what the watermark pipeline stores: contracts built from full context,
    # then only the generated-code portion is kept.
    full_contracts = [asdict(c) for c in build_block_contracts(prompt_code + gen_code)]
    # The last len(gen_code_blocks) contracts belong to generated_code
    gen_code_block_count = len([c for c in build_block_contracts(gen_code)])
    embedded = full_contracts[-gen_code_block_count:]

    # Simulate extraction: rebuild from generated_code only
    rebuilt = rebuild_block_contracts(gen_code)

    report = compare_block_contracts(embedded, rebuilt)

    assert not report.structure_mismatch, (
        f"structure_mismatch should be False but got mismatches: {report.structure_mismatches}"
    )
    assert report.contract_valid


def test_rebuild_block_contracts_matches_canonical_builder():
    code = (
        "def f(x):\n"
        "    total = x + 1\n"
        "    return total\n"
    )

    rebuilt = rebuild_block_contracts(code)
    canonical = [asdict(contract) for contract in build_block_contracts(code)]

    assert rebuilt == canonical


def test_rebuild_block_contracts_uses_adaptive_gamma_metadata(tmp_path):
    code = "x = 1\n"
    entropy_units = build_block_contracts(code)[0].entropy_units
    profile_payload = {
        "language": "python",
        "model_family": "demo-model",
        "quantiles_units": {
            "p10": max(0, entropy_units - 2),
            "p50": max(0, entropy_units - 1),
            "p75": entropy_units,
            "p90": entropy_units + 1,
            "p95": entropy_units + 2,
        },
        "strategy": "piecewise_quantile",
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile_payload), encoding="utf-8")

    anchors = {
        "p10": 0.95,
        "p50": 0.75,
        "p75": 0.55,
        "p90": 0.35,
        "p95": 0.25,
    }
    schedule = PiecewiseQuantileSchedule(
        profile=EntropyProfile.load(profile_path),
        anchor_quantiles=tuple(anchors.keys()),
        anchor_gammas=tuple(anchors.values()),
    )
    embedded = [
        asdict(contract)
        for contract in build_block_contracts(
            code,
            gamma_resolver=lambda units: schedule.resolve(units, 4),
        )
    ]

    rebuilt = rebuild_block_contracts(
        code,
        watermark_metadata={
            "adaptive_mode": "piecewise_quantile",
            "watermark_params": {
                "lsh_d": 4,
                "adaptive_gamma": {
                    "strategy": "piecewise_quantile",
                    "profile_id": "entropy-profile-v1",
                    "anchors": anchors,
                    "profile": profile_payload,
                },
            },
        },
    )

    assert rebuilt == embedded
