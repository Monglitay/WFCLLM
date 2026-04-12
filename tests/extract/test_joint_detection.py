"""Tests for semantic and lexical joint detection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wfcllm.common.offline_code_eval import compute_pass_at_k
from wfcllm.common.offline_code_eval import compute_roc_auc
from wfcllm.common.offline_code_eval import compute_tpr_at_fpr
from wfcllm.extract.hypothesis import JointDetectionResult
from wfcllm.extract.hypothesis import LexicalDetectionResult
from wfcllm.extract.hypothesis import fuse_joint_detection
from wfcllm.watermark.token_channel.config import TokenChannelConfig

from scripts.evaluate_dual_channel import CommandRunResult
from scripts.evaluate_dual_channel import run_evaluation


def test_joint_detection_uses_weighted_lexical_support_factor() -> None:
    lexical = LexicalDetectionResult(
        num_positions_scored=8,
        num_green_hits=6,
        green_fraction=0.75,
        lexical_z_score=2.0,
        lexical_p_value=0.1,
    )
    config = TokenChannelConfig(enabled=True)
    config.joint.semantic_weight = 1.0
    config.joint.lexical_weight = 0.5
    config.joint.lexical_full_weight_min_positions = 16
    config.joint.threshold = 4.0

    result = fuse_joint_detection(semantic_z_score=3.0, lexical_result=lexical, config=config)

    assert result.joint_score == pytest.approx(3.5)
    assert result.p_joint == pytest.approx(1.0 - 0.9997673709209645)
    assert result.prediction is False
    assert result.confidence == pytest.approx(1.0 - result.p_joint)
    assert result.rationale == "semantic borderline, lexical supportive"


def test_lexical_result_can_be_promoted_for_lexical_only_mode() -> None:
    lexical = LexicalDetectionResult(
        num_positions_scored=12,
        num_green_hits=10,
        green_fraction=10 / 12,
        lexical_z_score=4.2,
        lexical_p_value=0.000013,
    )

    result = lexical.to_joint_equivalent(threshold=4.0)

    assert isinstance(result, JointDetectionResult)
    assert result.joint_score == pytest.approx(4.2)
    assert result.p_joint == pytest.approx(0.000013)
    assert result.prediction is True
    assert result.rationale == "lexical-only evidence"


def test_compute_pass_at_k_matches_humaneval_style_estimator() -> None:
    records = [
        {"task_id": "task-1", "passed": True},
        {"task_id": "task-1", "passed": False},
        {"task_id": "task-2", "passed": False},
        {"task_id": "task-2", "passed": False},
    ]

    assert compute_pass_at_k(records, k=1) == pytest.approx(0.25)
    assert compute_pass_at_k(records, k=2) == pytest.approx(0.5)
    assert compute_pass_at_k(records, k=10) == pytest.approx(0.5)


def test_distribution_metrics_cover_roc_auc_and_tpr_at_one_percent_fpr() -> None:
    positive_scores = [0.81, 0.77, 0.75]
    negative_scores = [0.21, 0.18, 0.10]

    assert compute_roc_auc(positive_scores, negative_scores) == pytest.approx(1.0)
    assert compute_tpr_at_fpr(positive_scores, negative_scores, target_fpr=0.01) == pytest.approx(1.0)


def test_evaluate_dual_channel_builds_all_three_modes(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "watermark": {"secret_key": "secret", "lm_model_path": "offline-model"},
                "extract": {"secret_key": "secret"},
                "generate_negative": {"output_path": str(tmp_path / "negative.jsonl")},
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(command: list[str], env: dict[str, str] | None = None) -> CommandRunResult:
        assert env is not None
        assert env["HF_HUB_OFFLINE"] == "1"
        phase = _arg_value(command, "--phase")
        mode = _optional_arg_value(command, "--token-channel-mode") or "semantic-only"

        if phase == "watermark":
            output_dir = Path(_arg_value(command, "--output-dir"))
            output_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = output_dir / f"humaneval_{mode}.jsonl"
            artifact_path.write_text(_watermarked_rows_for_mode(mode), encoding="utf-8")
            return CommandRunResult(exit_code=0, elapsed_seconds=_latency_for_mode(mode))

        if phase == "generate-negative":
            output_path = Path(_arg_value(command, "--negative-output"))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(_negative_rows(), encoding="utf-8")
            return CommandRunResult(exit_code=0, elapsed_seconds=0.2)

        if phase == "extract":
            input_path = Path(_arg_value(command, "--input-file"))
            output_dir = Path(_arg_value(command, "--extract-output-dir"))
            output_dir.mkdir(parents=True, exist_ok=True)
            details_path = output_dir / f"{input_path.stem}_details.jsonl"
            summary_path = output_dir / f"{input_path.stem}_summary.json"
            payload = _details_rows_for_input(mode=mode, input_path=input_path)
            details_path.write_text(payload, encoding="utf-8")
            summary_path.write_text(
                json.dumps({"meta": {"input_file": str(input_path)}, "summary": {}}),
                encoding="utf-8",
            )
            return CommandRunResult(exit_code=0, elapsed_seconds=0.1)

        raise AssertionError(f"unexpected phase: {phase}")

    result = run_evaluation(
        dataset="humaneval",
        config_path=str(config_path),
        output_dir=str(tmp_path / "eval"),
        command_runner=fake_runner,
    )

    assert set(result["modes"]) == {"semantic-only", "lexical-only", "dual-channel"}
    assert (tmp_path / "eval" / "evaluation_summary.json").exists()
    assert (tmp_path / "eval" / "semantic-only" / "metrics.json").exists()
    assert (tmp_path / "eval" / "lexical-only" / "metrics.json").exists()
    assert (tmp_path / "eval" / "dual-channel" / "metrics.json").exists()


def _arg_value(command: list[str], flag: str) -> str:
    index = command.index(flag)
    return command[index + 1]


def _optional_arg_value(command: list[str], flag: str) -> str | None:
    if flag not in command:
        return None
    return _arg_value(command, flag)


def _watermarked_rows_for_mode(mode: str) -> str:
    by_mode = {
        "semantic-only": [
            {
                "id": "task-1-sample-0",
                "task_id": "task-1",
                "generated_code": "def solve(x):\n    return x + 1\n",
                "passed": True,
                "retry_summary": {"attempts_total": 2},
                "total_blocks": 2,
            },
            {
                "id": "task-1-sample-1",
                "task_id": "task-1",
                "generated_code": "def solve(x):\n    return x\n",
                "passed": False,
                "retry_summary": {"attempts_total": 0},
                "total_blocks": 2,
            },
            {
                "id": "task-2-sample-0",
                "task_id": "task-2",
                "generated_code": "def solve(x):\n    return x * 2\n",
                "passed": False,
                "retry_summary": {"attempts_total": 1},
                "total_blocks": 2,
            },
            {
                "id": "task-2-sample-1",
                "task_id": "task-2",
                "generated_code": "def solve(x):\n    return x - 1\n",
                "passed": False,
                "retry_summary": {"attempts_total": 1},
                "total_blocks": 2,
            },
        ],
        "lexical-only": [
            {
                "id": "task-1-sample-0",
                "task_id": "task-1",
                "generated_code": "def solve(x):\n    return x + 1\n",
                "passed": True,
                "retry_summary": {"attempts_total": 3},
                "total_blocks": 2,
            },
            {
                "id": "task-1-sample-1",
                "task_id": "task-1",
                "generated_code": "def solve(x):\n    return x\n",
                "passed": False,
                "retry_summary": {"attempts_total": 1},
                "total_blocks": 2,
            },
            {
                "id": "task-2-sample-0",
                "task_id": "task-2",
                "generated_code": "def solve(x):\n    return x * 2\n",
                "passed": False,
                "retry_summary": {"attempts_total": 2},
                "total_blocks": 2,
            },
            {
                "id": "task-2-sample-1",
                "task_id": "task-2",
                "generated_code": "def solve(x):\n    return x - 1\n",
                "passed": False,
                "retry_summary": {"attempts_total": 2},
                "total_blocks": 2,
            },
        ],
        "dual-channel": [
            {
                "id": "task-1-sample-0",
                "task_id": "task-1",
                "generated_code": "def solve(x):\n    return x + 1\n",
                "passed": True,
                "retry_summary": {"attempts_total": 2},
                "total_blocks": 2,
            },
            {
                "id": "task-1-sample-1",
                "task_id": "task-1",
                "generated_code": "def solve(x):\n    return x\n",
                "passed": False,
                "retry_summary": {"attempts_total": 1},
                "total_blocks": 2,
            },
            {
                "id": "task-2-sample-0",
                "task_id": "task-2",
                "generated_code": "def solve(x):\n    return x * 2\n",
                "passed": True,
                "retry_summary": {"attempts_total": 1},
                "total_blocks": 2,
            },
            {
                "id": "task-2-sample-1",
                "task_id": "task-2",
                "generated_code": "def solve(x):\n    return x - 1\n",
                "passed": False,
                "retry_summary": {"attempts_total": 1},
                "total_blocks": 2,
            },
        ],
    }
    return "\n".join(json.dumps(row) for row in by_mode[mode]) + "\n"


def _negative_rows() -> str:
    rows = [
        {"id": "neg-1", "task_id": "neg-1", "generated_code": "def solve(x):\n    return x\n"},
        {"id": "neg-2", "task_id": "neg-2", "generated_code": "def solve(x):\n    return x\n"},
    ]
    return "\n".join(json.dumps(row) for row in rows) + "\n"


def _details_rows_for_input(mode: str, input_path: Path) -> str:
    positive_rows = {
        "semantic-only": [
            {"id": "task-1-sample-0", "is_watermarked": True, "z_score": 3.1, "p_value": 0.01, "independent_blocks": 4, "hits": 3, "lexical_z_score": 0.2, "lexical_p_value": 0.4, "joint_score": 3.1, "p_joint": 0.01, "joint_prediction": False},
            {"id": "task-1-sample-1", "is_watermarked": False, "z_score": 2.9, "p_value": 0.02, "independent_blocks": 4, "hits": 2, "lexical_z_score": 0.1, "lexical_p_value": 0.5, "joint_score": 2.9, "p_joint": 0.02, "joint_prediction": False},
        ],
        "lexical-only": [
            {"id": "task-1-sample-0", "is_watermarked": True, "z_score": 0.3, "p_value": 0.4, "independent_blocks": 4, "hits": 2, "lexical_z_score": 1.9, "lexical_p_value": 0.03, "joint_score": 1.9, "p_joint": 0.03, "joint_prediction": False},
            {"id": "task-1-sample-1", "is_watermarked": False, "z_score": 0.1, "p_value": 0.45, "independent_blocks": 4, "hits": 2, "lexical_z_score": 1.4, "lexical_p_value": 0.08, "joint_score": 1.4, "p_joint": 0.08, "joint_prediction": False},
        ],
        "dual-channel": [
            {"id": "task-1-sample-0", "is_watermarked": True, "z_score": 3.0, "p_value": 0.01, "independent_blocks": 4, "hits": 3, "lexical_z_score": 1.8, "lexical_p_value": 0.03, "joint_score": 4.1, "p_joint": 0.002, "joint_prediction": True},
            {"id": "task-1-sample-1", "is_watermarked": False, "z_score": 2.7, "p_value": 0.03, "independent_blocks": 4, "hits": 2, "lexical_z_score": 1.2, "lexical_p_value": 0.09, "joint_score": 3.5, "p_joint": 0.01, "joint_prediction": False},
        ],
    }
    negative_rows = {
        "semantic-only": [
            {"id": "neg-1", "is_watermarked": False, "z_score": 0.2, "p_value": 0.4, "independent_blocks": 4, "hits": 1, "lexical_z_score": 0.1, "lexical_p_value": 0.5, "joint_score": 0.2, "p_joint": 0.4, "joint_prediction": False},
            {"id": "neg-2", "is_watermarked": False, "z_score": 0.4, "p_value": 0.35, "independent_blocks": 4, "hits": 1, "lexical_z_score": 0.2, "lexical_p_value": 0.45, "joint_score": 0.4, "p_joint": 0.35, "joint_prediction": False},
        ],
        "lexical-only": [
            {"id": "neg-1", "is_watermarked": False, "z_score": 0.1, "p_value": 0.45, "independent_blocks": 4, "hits": 1, "lexical_z_score": 0.2, "lexical_p_value": 0.4, "joint_score": 0.2, "p_joint": 0.4, "joint_prediction": False},
            {"id": "neg-2", "is_watermarked": False, "z_score": 0.2, "p_value": 0.4, "independent_blocks": 4, "hits": 1, "lexical_z_score": 0.3, "lexical_p_value": 0.35, "joint_score": 0.3, "p_joint": 0.35, "joint_prediction": False},
        ],
        "dual-channel": [
            {"id": "neg-1", "is_watermarked": False, "z_score": 0.2, "p_value": 0.4, "independent_blocks": 4, "hits": 1, "lexical_z_score": 0.2, "lexical_p_value": 0.4, "joint_score": 0.4, "p_joint": 0.3, "joint_prediction": False},
            {"id": "neg-2", "is_watermarked": False, "z_score": 0.3, "p_value": 0.35, "independent_blocks": 4, "hits": 1, "lexical_z_score": 0.4, "lexical_p_value": 0.3, "joint_score": 0.6, "p_joint": 0.25, "joint_prediction": False},
        ],
    }
    rows = negative_rows[mode] if "negative" in input_path.name else positive_rows[mode]
    return "\n".join(json.dumps(row) for row in rows) + "\n"


def _latency_for_mode(mode: str) -> float:
    return {
        "semantic-only": 1.0,
        "lexical-only": 1.3,
        "dual-channel": 1.2,
    }[mode]
