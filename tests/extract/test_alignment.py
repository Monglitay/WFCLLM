"""Tests for extract-side block contract alignment."""

from __future__ import annotations

from dataclasses import asdict
import json

from wfcllm.common.block_contract import build_block_contracts
from wfcllm.extract.alignment import compare_block_contracts, rebuild_block_contracts
from wfcllm.watermark.entropy_profile import EntropyProfile
from wfcllm.watermark.gamma_schedule import PiecewiseQuantileSchedule


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
