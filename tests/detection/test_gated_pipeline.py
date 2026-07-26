from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from wfcllm.detection.gated_pipeline import (
    GatedDetectionBindings,
    GatedDetectionPipeline,
)
from wfcllm.detection.scoring import GatedWindowScorer


@dataclass(frozen=True)
class _Window:
    suitable: bool
    window_text: str
    parent_descriptor: str = "python-statement-window/v1||parent=module|ordinal=0|role=body"
    start_byte: int = 0
    end_byte: int = 1
    close_probability: float | None = 0.8
    suitable_probability: float | None = 0.9


class _Semantic:
    def score(self, *, window_text: str, parent_descriptor: str):
        del parent_descriptor
        if window_text == "unstable":
            return SimpleNamespace(hit=False, stable=False, margin=0.0)
        return SimpleNamespace(hit=window_text == "hit", stable=True, margin=0.5)


class _Extractor:
    window_contract_version = "python-statement-window/v1"

    def extract(self, final_code: str):
        windows = tuple(
            _Window(True, token, start_byte=index, end_byte=index + 1)
            for index, token in enumerate(final_code.split())
        )
        return SimpleNamespace(windows=windows)


def _row(sample_id: str, code: str) -> dict[str, str]:
    return {"id": sample_id, "dataset": "humaneval", "prompt": "", "final_code": code}


def _bindings() -> GatedDetectionBindings:
    return GatedDetectionBindings(
        gate_bundle_sha256="a" * 64,
        semantic_encoder_sha256="b" * 64,
        lsh_config_sha256="c" * 64,
        key_identifier_sha256="d" * 64,
        negative_corpus_manifest_sha256="e" * 64,
    )


def test_gated_scorer_uses_only_stable_suitable_windows() -> None:
    scorer = GatedWindowScorer(semantic_scorer=_Semantic(), minimum_reliable_windows=2)
    score = scorer.score(
        [
            _Window(True, "hit"),
            _Window(True, "miss"),
            _Window(True, "unstable"),
            _Window(False, "hit"),
        ]
    )
    assert (score.hit_count, score.miss_count, score.abstain_count) == (1, 1, 1)
    assert score.reliable_window_count == 2
    assert score.hit_rate == 0.5


class _ChannelSemantic:
    def score(self, *, window_text: str, parent_descriptor: str):
        raise AssertionError("multi-channel scoring must use score_channels")

    def score_channels(
        self, *, window_text: str, parent_descriptor: str, channel_count: int
    ):
        assert window_text == "hit"
        assert parent_descriptor.endswith("role=body")
        assert channel_count == 2
        return (
            SimpleNamespace(hit=True, stable=True, margin=0.5),
            SimpleNamespace(hit=False, stable=True, margin=0.4),
        )


def test_gated_scorer_counts_each_configured_evidence_channel() -> None:
    scorer = GatedWindowScorer(
        semantic_scorer=_ChannelSemantic(),
        minimum_reliable_windows=2,
        evidence_channels=2,
    )

    score = scorer.score([_Window(True, "hit")])

    assert (score.hit_count, score.miss_count, score.abstain_count) == (1, 1, 0)
    assert score.reliable_window_count == 2
    assert len(score.evidence) == 2
    assert score.evidence[0].parent_descriptor.endswith("role=body")
    assert score.evidence[1].parent_descriptor.endswith(
        "role=body|wfcllm-evidence-channel=1"
    )


def test_gated_scorer_can_count_unsuitable_windows_for_experimental_detection() -> None:
    scorer = GatedWindowScorer(
        semantic_scorer=_Semantic(),
        minimum_reliable_windows=1,
        allow_unsuitable_windows=True,
    )

    score = scorer.score(
        [
            _Window(False, "hit"),
            _Window(False, "miss"),
        ]
    )

    assert (score.hit_count, score.miss_count) == (1, 1)
    assert score.reliable_window_count == 2


def test_insufficient_reliable_windows_abstains() -> None:
    scorer = GatedWindowScorer(semantic_scorer=_Semantic(), minimum_reliable_windows=2)
    assert scorer.detect([_Window(True, "hit")]).decision == "insufficient_evidence"


def test_pipeline_is_final_code_only_and_outputs_derived_details(tmp_path: Path) -> None:
    scorer = GatedWindowScorer(semantic_scorer=_Semantic(), minimum_reliable_windows=2)
    pipeline = GatedDetectionPipeline(
        extractor=_Extractor(), scorer=scorer, bindings=_bindings(), target_fpr=0.05
    )
    artifact = pipeline.calibrate([_row("n1", "miss miss"), _row("n2", "hit miss")])
    output = tmp_path / "details.jsonl"
    results = pipeline.detect([_row("p", "hit hit")], artifact=artifact, output_path=output)

    assert results[0].method_name == "gated_semantic_window_v1"
    assert results[0].detector_mode == "wfcllm-gated-semantic-window/v1"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert set(payload["windows"][0]) <= {
        "start_byte", "end_byte", "parent_descriptor", "close_probability",
        "suitable_probability", "status", "margin",
    }
    assert "final_code" not in payload


def test_pooled_sample_level_calibration_keeps_insufficient_negatives_in_fpr_denominator() -> None:
    pipeline = GatedDetectionPipeline(
        extractor=_Extractor(),
        scorer=GatedWindowScorer(semantic_scorer=_Semantic(), minimum_reliable_windows=2),
        bindings=_bindings(),
        target_fpr=0.05,
        calibration_group_by="pooled_reliable_hit_rate",
    )

    artifact = pipeline.calibrate(
        [
            _row("insufficient", "hit"),
            _row("enough-miss", "miss miss"),
            _row("enough-half", "hit miss"),
        ]
    )

    assert artifact.reliable_window_count_buckets == {"2": (0.0, 0.0, 0.5)}


def test_detector_rejects_generation_audit_as_input(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    path.write_text(json.dumps({"id": "x", "audit_only": True}) + "\n", encoding="utf-8")
    pipeline = GatedDetectionPipeline(
        extractor=_Extractor(),
        scorer=GatedWindowScorer(semantic_scorer=_Semantic(), minimum_reliable_windows=2),
        bindings=_bindings(),
    )
    artifact = pipeline.calibrate([_row("n", "miss miss")])
    with pytest.raises(ValueError, match="final_code|exactly four fields"):
        pipeline.detect_jsonl(path, artifact=artifact)
