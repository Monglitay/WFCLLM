"""Token-channel-train integration tests (extracted from tests/test_run.py TestCLI)."""

import argparse
import json
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUN_PY = PROJECT_ROOT / "run.py"
README_MD = PROJECT_ROOT / "README.md"

import sys
sys.path.insert(0, str(PROJECT_ROOT))

from wfcllm.orchestration.state import RunStateManager as RunState


def test_main_does_not_run_token_channel_train_by_default(monkeypatch, tmp_path):
    """Top-level main() should run only the 3 main phases when --phase is not specified."""
    # Redirect state file so we start clean
    state_path = tmp_path / "rs.json"
    monkeypatch.setattr("wfcllm.cli.entry.DEFAULT_STATE_FILE", state_path)

    # Fake each runner; capture which were called
    run_calls = []
    def make_fake(name):
        def fake(args, state):
            run_calls.append(name)
            return 0
        return fake

    monkeypatch.setattr("wfcllm.cli.entry.run_encoder", make_fake("encoder"))
    monkeypatch.setattr("wfcllm.cli.entry.run_watermark", make_fake("watermark"))
    monkeypatch.setattr("wfcllm.cli.entry.run_extract", make_fake("extract"))
    monkeypatch.setattr("wfcllm.cli.entry.run_generate_negative", make_fake("generate-negative"))
    monkeypatch.setattr("wfcllm.cli.entry.run_token_channel_train", make_fake("token-channel-train"))

    from wfcllm.cli.entry import main
    rc = main([])  # no flags = default 3-phase flow
    assert rc == 0
    assert run_calls == ["encoder", "watermark", "extract"]

def test_run_phase_dispatches_token_channel_train(tmp_path, monkeypatch):
    state = RunState(tmp_path / "run_state.json")
    args = argparse.Namespace()
    seen = []

    def fake_runner(passed_args, passed_state):
        seen.append((passed_args, passed_state))
        return 0

    monkeypatch.setattr("wfcllm.cli.runners.run_token_channel_train", fake_runner)

    from wfcllm.cli.runners import run_phase
    assert run_phase("token-channel-train", args, state) == 0
    assert seen == [(args, state)]

def test_base_config_includes_token_channel_train_defaults():
    config = json.loads((PROJECT_ROOT / "configs/base_config.json").read_text(encoding="utf-8"))

    assert config["token_channel_train"] == {
        "dataset": "humaneval",
        "dataset_path": "data/datasets",
        "lm_model_path": "data/models/deepseek-coder-7b-base",
        "model_path": "data/models/token-channel",
        "cache_path": "data/token_channel/train_corpus.json",
        "context_width": 128,
        "hidden_size": 64,
        "batch_size": 128,
        "epochs": 3,
        "lr": 0.001,
        "entropy_threshold": 1.0,
        "diversity_threshold": 2,
        "split_ratio": 0.9,
        "seed": 0,
    }

def test_run_token_channel_train_loads_defaults_from_config(tmp_path, capsys):
    from wfcllm.cli.runners import run_token_channel_train
    from wfcllm.watermark.token_channel.train import TokenChannelEpochMetrics
    from wfcllm.watermark.token_channel.train_workflow import TokenChannelTrainWorkflowSummary
    from unittest.mock import patch

    lm_model_path = tmp_path / "teacher-model"
    lm_model_path.mkdir()

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "token_channel_train": {
                    "dataset": "humaneval",
                    "dataset_path": "data/datasets",
                    "lm_model_path": str(lm_model_path),
                    "model_path": "data/models/token-channel",
                    "cache_path": "data/token_channel/train_corpus.json",
                    "context_width": 128,
                    "hidden_size": 64,
                    "batch_size": 128,
                    "epochs": 3,
                    "lr": 0.001,
                    "entropy_threshold": 1.0,
                    "diversity_threshold": 2,
                    "split_ratio": 0.9,
                    "seed": 0,
                }
            }
        ),
        encoding="utf-8",
    )
    state = RunState(tmp_path / "state.json")
    args = argparse.Namespace(
        config=config_path,
        dataset=None,
        dataset_path=None,
        lm_model_path=None,
        token_channel_cache_path=None,
        token_channel_model_path=None,
        token_channel_context_width=None,
        token_channel_hidden_size=None,
        token_channel_batch_size=None,
        token_channel_epochs=None,
        token_channel_lr=None,
        token_channel_entropy_threshold=None,
        token_channel_diversity_threshold=None,
        token_channel_split_ratio=None,
        token_channel_seed=None,
    )

    summary = TokenChannelTrainWorkflowSummary(
        dataset="humaneval",
        training_rows=12,
        train_rows=10,
        validation_rows=2,
        artifact_dir=tmp_path / "artifacts",
        cache_path=tmp_path / "train_cache.json",
        compatibility_ok=True,
        epochs=(
            TokenChannelEpochMetrics(
                epoch=1,
                train_loss=0.5,
                validation_loss=0.4,
                switch_loss=0.3,
            ),
        ),
        switch_target_positive_count=7,
        switch_target_negative_count=5,
    )

    with patch(
        "wfcllm.watermark.token_channel.train_workflow.run_token_channel_train_workflow",
        return_value=summary,
    ):
        assert run_token_channel_train(args, state) == 0

    captured = capsys.readouterr()
    assert "dataset: humaneval" in captured.out
    assert f"cache_path: {summary.cache_path}" in captured.out
    assert f"artifact_dir: {summary.artifact_dir}" in captured.out
    assert state.is_done("token-channel-train") is True
    assert state.get("token-channel-train", "dataset") == "humaneval"
    assert state.get("token-channel-train", "cache_path") == str(summary.cache_path)
    assert state.get("token-channel-train", "artifact_dir") == str(summary.artifact_dir)

def test_run_token_channel_train_cli_overrides_dataset_inputs(tmp_path, capsys):
    from wfcllm.cli.runners import run_token_channel_train
    from wfcllm.watermark.token_channel.train import TokenChannelEpochMetrics
    from wfcllm.watermark.token_channel.train_workflow import TokenChannelTrainWorkflowSummary
    from unittest.mock import patch

    config_lm_model_path = tmp_path / "config-teacher-model"
    config_lm_model_path.mkdir()
    cli_lm_model_path = tmp_path / "cli-teacher-model"
    cli_lm_model_path.mkdir()

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "token_channel_train": {
                    "dataset": "humaneval",
                    "dataset_path": "data/datasets",
                    "lm_model_path": str(config_lm_model_path),
                    "model_path": "data/models/token-channel",
                    "cache_path": "data/token_channel/train_corpus.json",
                    "context_width": 128,
                    "hidden_size": 64,
                    "batch_size": 128,
                    "epochs": 3,
                    "lr": 0.001,
                    "entropy_threshold": 1.0,
                    "diversity_threshold": 2,
                    "split_ratio": 0.9,
                    "seed": 0,
                }
            }
        ),
        encoding="utf-8",
    )
    state = RunState(tmp_path / "state.json")
    args = argparse.Namespace(
        config=config_path,
        dataset="mbpp",
        dataset_path="custom/datasets",
        lm_model_path=str(cli_lm_model_path),
        token_channel_cache_path=None,
        token_channel_model_path=None,
        token_channel_context_width=None,
        token_channel_hidden_size=None,
        token_channel_batch_size=None,
        token_channel_epochs=None,
        token_channel_lr=None,
        token_channel_entropy_threshold=None,
        token_channel_diversity_threshold=None,
        token_channel_split_ratio=None,
        token_channel_seed=None,
    )

    summary = TokenChannelTrainWorkflowSummary(
        dataset="mbpp",
        training_rows=8,
        train_rows=6,
        validation_rows=2,
        artifact_dir=tmp_path / "artifacts",
        cache_path=tmp_path / "train_cache.json",
        compatibility_ok=True,
        epochs=(
            TokenChannelEpochMetrics(
                epoch=1,
                train_loss=0.5,
                validation_loss=0.4,
                switch_loss=0.3,
            ),
        ),
        switch_target_positive_count=4,
        switch_target_negative_count=4,
    )

    seen_config = None

    def fake_workflow(config):
        nonlocal seen_config
        seen_config = config
        return summary

    with patch(
        "wfcllm.watermark.token_channel.train_workflow.run_token_channel_train_workflow",
        side_effect=fake_workflow,
    ):
        assert run_token_channel_train(args, state) == 0

    captured = capsys.readouterr()
    assert seen_config is not None
    assert seen_config.dataset == "mbpp"
    assert seen_config.dataset_path == Path("custom/datasets")
    assert seen_config.lm_model_path == cli_lm_model_path
    assert "dataset: mbpp" in captured.out

def test_run_token_channel_train_cli_overrides_model_path(tmp_path):
    from wfcllm.cli.runners import run_token_channel_train
    from wfcllm.watermark.token_channel.train import TokenChannelEpochMetrics
    from wfcllm.watermark.token_channel.train_workflow import TokenChannelTrainWorkflowSummary
    from unittest.mock import patch

    lm_model_path = tmp_path / "teacher-model"
    lm_model_path.mkdir()

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "token_channel_train": {
                    "dataset": "humaneval",
                    "dataset_path": "data/datasets",
                    "lm_model_path": str(lm_model_path),
                    "model_path": "data/models/token-channel-from-config",
                    "cache_path": "data/token_channel/train_corpus.json",
                    "context_width": 128,
                    "hidden_size": 64,
                    "batch_size": 128,
                    "epochs": 3,
                    "lr": 0.001,
                    "entropy_threshold": 1.0,
                    "diversity_threshold": 2,
                    "split_ratio": 0.9,
                    "seed": 0,
                }
            }
        ),
        encoding="utf-8",
    )
    state = RunState(tmp_path / "state.json")
    args = argparse.Namespace(
        config=config_path,
        dataset=None,
        dataset_path=None,
        lm_model_path=None,
        token_channel_cache_path=None,
        token_channel_model_path=str(tmp_path / "override-model"),
        token_channel_context_width=None,
        token_channel_hidden_size=None,
        token_channel_batch_size=None,
        token_channel_epochs=None,
        token_channel_lr=None,
        token_channel_entropy_threshold=None,
        token_channel_diversity_threshold=None,
        token_channel_split_ratio=None,
        token_channel_seed=None,
    )

    summary = TokenChannelTrainWorkflowSummary(
        dataset="humaneval",
        training_rows=12,
        train_rows=10,
        validation_rows=2,
        artifact_dir=tmp_path / "override-model",
        cache_path=tmp_path / "train_cache.json",
        compatibility_ok=True,
        epochs=(
            TokenChannelEpochMetrics(
                epoch=1,
                train_loss=0.5,
                validation_loss=0.4,
                switch_loss=0.3,
            ),
        ),
        switch_target_positive_count=7,
        switch_target_negative_count=5,
    )

    seen_config = None

    def fake_workflow(config):
        nonlocal seen_config
        seen_config = config
        return summary

    with patch(
        "wfcllm.watermark.token_channel.train_workflow.run_token_channel_train_workflow",
        side_effect=fake_workflow,
    ):
        assert run_token_channel_train(args, state) == 0

    assert seen_config is not None
    assert seen_config.model_path == tmp_path / "override-model"

def test_run_token_channel_train_applies_defaults_for_partial_custom_config(tmp_path):
    from wfcllm.cli.runners import run_token_channel_train
    from wfcllm.watermark.token_channel.train import TokenChannelEpochMetrics
    from wfcllm.watermark.token_channel.train_workflow import TokenChannelTrainWorkflowSummary
    from unittest.mock import patch

    lm_model_path = tmp_path / "teacher-model"
    lm_model_path.mkdir()

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "token_channel_train": {
                    "dataset": "mbpp",
                    "lm_model_path": str(lm_model_path),
                }
            }
        ),
        encoding="utf-8",
    )
    state = RunState(tmp_path / "state.json")
    args = argparse.Namespace(
        config=config_path,
        dataset=None,
        dataset_path=None,
        lm_model_path=None,
        token_channel_cache_path=None,
        token_channel_model_path=None,
        token_channel_context_width=None,
        token_channel_hidden_size=None,
        token_channel_batch_size=None,
        token_channel_epochs=None,
        token_channel_lr=None,
        token_channel_entropy_threshold=None,
        token_channel_diversity_threshold=None,
        token_channel_split_ratio=None,
        token_channel_seed=None,
    )

    summary = TokenChannelTrainWorkflowSummary(
        dataset="mbpp",
        training_rows=8,
        train_rows=6,
        validation_rows=2,
        artifact_dir=tmp_path / "artifacts",
        cache_path=tmp_path / "train_cache.json",
        compatibility_ok=True,
        epochs=(
            TokenChannelEpochMetrics(
                epoch=1,
                train_loss=0.5,
                validation_loss=0.4,
                switch_loss=0.3,
            ),
        ),
        switch_target_positive_count=4,
        switch_target_negative_count=4,
    )

    seen_config = None

    def fake_workflow(config):
        nonlocal seen_config
        seen_config = config
        return summary

    with patch(
        "wfcllm.watermark.token_channel.train_workflow.run_token_channel_train_workflow",
        side_effect=fake_workflow,
    ):
        assert run_token_channel_train(args, state) == 0

    assert seen_config is not None
    assert seen_config.dataset == "mbpp"
    assert seen_config.dataset_path == Path("data/datasets")
    assert seen_config.model_path == Path("data/models/token-channel")
    assert seen_config.cache_path == Path("data/token_channel/train_corpus.json")
    assert seen_config.context_width == 128
    assert seen_config.hidden_size == 64
    assert seen_config.batch_size == 128
    assert seen_config.epochs == 3
    assert seen_config.lr == pytest.approx(0.001)
    assert seen_config.entropy_threshold == pytest.approx(1.0)
    assert seen_config.diversity_threshold == 2
    assert seen_config.split_ratio == pytest.approx(0.9)
    assert seen_config.seed == 0

def test_run_token_channel_train_prints_summary_output(tmp_path, capsys):
    from wfcllm.cli.runners import run_token_channel_train
    from wfcllm.watermark.token_channel.train import TokenChannelEpochMetrics
    from wfcllm.watermark.token_channel.train_workflow import TokenChannelTrainWorkflowSummary
    from unittest.mock import patch

    lm_model_path = tmp_path / "teacher-model"
    lm_model_path.mkdir()

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "token_channel_train": {
                    "dataset": "humaneval",
                    "dataset_path": "data/datasets",
                    "lm_model_path": str(lm_model_path),
                    "model_path": "data/models/token-channel",
                    "cache_path": "data/token_channel/train_corpus.json",
                    "context_width": 128,
                    "hidden_size": 64,
                    "batch_size": 128,
                    "epochs": 3,
                    "lr": 0.001,
                    "entropy_threshold": 1.0,
                    "diversity_threshold": 2,
                    "split_ratio": 0.9,
                    "seed": 0,
                }
            }
        ),
        encoding="utf-8",
    )
    state = RunState(tmp_path / "state.json")
    args = argparse.Namespace(
        config=config_path,
        dataset=None,
        dataset_path=None,
        lm_model_path=None,
        token_channel_cache_path=None,
        token_channel_model_path=None,
        token_channel_context_width=None,
        token_channel_hidden_size=None,
        token_channel_batch_size=None,
        token_channel_epochs=None,
        token_channel_lr=None,
        token_channel_entropy_threshold=None,
        token_channel_diversity_threshold=None,
        token_channel_split_ratio=None,
        token_channel_seed=None,
    )

    summary = TokenChannelTrainWorkflowSummary(
        dataset="humaneval",
        training_rows=12,
        train_rows=10,
        validation_rows=2,
        artifact_dir=tmp_path / "artifact-dir",
        cache_path=tmp_path / "cache.json",
        compatibility_ok=True,
        epochs=(
            TokenChannelEpochMetrics(
                epoch=1,
                train_loss=0.5,
                validation_loss=0.4,
                switch_loss=0.3,
            ),
        ),
        switch_target_positive_count=7,
        switch_target_negative_count=5,
    )

    with patch(
        "wfcllm.watermark.token_channel.train_workflow.run_token_channel_train_workflow",
        return_value=summary,
    ), patch(
        "wfcllm.watermark.token_channel.train_workflow.format_token_channel_train_workflow_summary",
        return_value=["summary line 1", "summary line 2"],
    ):
        assert run_token_channel_train(args, state) == 0

    captured = capsys.readouterr()
    assert "summary line 1" in captured.out
    assert "summary line 2" in captured.out

def test_run_token_channel_train_prints_overwrite_notices_for_existing_paths(tmp_path,
    capsys,
):
    from wfcllm.cli.runners import run_token_channel_train
    from wfcllm.watermark.token_channel.train import TokenChannelEpochMetrics
    from wfcllm.watermark.token_channel.train_workflow import TokenChannelTrainWorkflowSummary
    from unittest.mock import patch

    lm_model_path = tmp_path / "teacher-model"
    lm_model_path.mkdir()
    existing_cache_path = tmp_path / "existing-cache.json"
    existing_cache_path.write_text("[]", encoding="utf-8")
    existing_model_path = tmp_path / "existing-model"
    existing_model_path.mkdir()

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "token_channel_train": {
                    "dataset": "humaneval",
                    "dataset_path": "data/datasets",
                    "lm_model_path": str(lm_model_path),
                    "model_path": str(existing_model_path),
                    "cache_path": str(existing_cache_path),
                    "context_width": 128,
                    "hidden_size": 64,
                    "batch_size": 128,
                    "epochs": 3,
                    "lr": 0.001,
                    "entropy_threshold": 1.0,
                    "diversity_threshold": 2,
                    "split_ratio": 0.9,
                    "seed": 0,
                }
            }
        ),
        encoding="utf-8",
    )
    state = RunState(tmp_path / "state.json")
    args = argparse.Namespace(
        config=config_path,
        dataset=None,
        dataset_path=None,
        lm_model_path=None,
        token_channel_cache_path=None,
        token_channel_model_path=None,
        token_channel_context_width=None,
        token_channel_hidden_size=None,
        token_channel_batch_size=None,
        token_channel_epochs=None,
        token_channel_lr=None,
        token_channel_entropy_threshold=None,
        token_channel_diversity_threshold=None,
        token_channel_split_ratio=None,
        token_channel_seed=None,
    )

    summary = TokenChannelTrainWorkflowSummary(
        dataset="humaneval",
        training_rows=12,
        train_rows=10,
        validation_rows=2,
        artifact_dir=existing_model_path,
        cache_path=existing_cache_path,
        compatibility_ok=True,
        epochs=(
            TokenChannelEpochMetrics(
                epoch=1,
                train_loss=0.5,
                validation_loss=0.4,
                switch_loss=0.3,
            ),
        ),
        switch_target_positive_count=7,
        switch_target_negative_count=5,
    )

    with patch(
        "wfcllm.watermark.token_channel.train_workflow.run_token_channel_train_workflow",
        return_value=summary,
    ):
        assert run_token_channel_train(args, state) == 0

    captured = capsys.readouterr()
    assert f"overwrite existing cache: {existing_cache_path}" in captured.out
    assert f"overwrite existing model artifacts: {existing_model_path}" in captured.out

def test_resolve_token_channel_train_config_applies_defaults_for_partial_custom_config(tmp_path,
):
    from wfcllm.cli.runners import resolve_token_channel_train_config

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "token_channel_train": {
                    "lm_model_path": "data/models/deepseek-coder-7b-base",
                }
            }
        ),
        encoding="utf-8",
    )
    state = RunState(tmp_path / "state.json")
    args = argparse.Namespace(
        config=config_path,
        dataset=None,
        dataset_path=None,
        lm_model_path=None,
        token_channel_cache_path=None,
        token_channel_model_path=None,
        token_channel_context_width=None,
        token_channel_hidden_size=None,
        token_channel_batch_size=None,
        token_channel_epochs=None,
        token_channel_lr=None,
        token_channel_entropy_threshold=None,
        token_channel_diversity_threshold=None,
        token_channel_split_ratio=None,
        token_channel_seed=None,
    )

    train_cfg = resolve_token_channel_train_config(args)

    assert train_cfg["dataset"] == "humaneval"
    assert train_cfg["dataset_path"] == "data/datasets"
    assert train_cfg["model_path"] == "data/models/token-channel"
    assert train_cfg["cache_path"] == "data/token_channel/train_corpus.json"
    assert train_cfg["context_width"] == 128
    assert train_cfg["hidden_size"] == 64
    assert train_cfg["batch_size"] == 128
    assert train_cfg["epochs"] == 3
    assert train_cfg["lr"] == pytest.approx(0.001)
    assert train_cfg["entropy_threshold"] == pytest.approx(1.0)
    assert train_cfg["diversity_threshold"] == 2
    assert train_cfg["split_ratio"] == pytest.approx(0.9)
    assert train_cfg["seed"] == 0

def test_run_token_channel_train_requires_merged_lm_model_path(tmp_path, capsys):
    from wfcllm.cli.runners import run_token_channel_train

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "token_channel_train": {
                    "dataset": "humaneval",
                }
            }
        ),
        encoding="utf-8",
    )
    state = RunState(tmp_path / "state.json")
    args = argparse.Namespace(
        config=config_path,
        dataset=None,
        dataset_path=None,
        lm_model_path=None,
        token_channel_cache_path=None,
        token_channel_model_path=None,
        token_channel_context_width=None,
        token_channel_hidden_size=None,
        token_channel_batch_size=None,
        token_channel_epochs=None,
        token_channel_lr=None,
        token_channel_entropy_threshold=None,
        token_channel_diversity_threshold=None,
        token_channel_split_ratio=None,
        token_channel_seed=None,
    )

    assert run_token_channel_train(args, state) != 0

    captured = capsys.readouterr()
    assert "lm_model_path" in captured.err
    assert state.is_done("token-channel-train") is False

def test_resolve_token_channel_train_config_rejects_non_object_section(tmp_path):
    from wfcllm.cli.runners import resolve_token_channel_train_config

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"token_channel_train": 1}),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        config=config_path,
        dataset=None,
        dataset_path=None,
        lm_model_path=None,
        token_channel_cache_path=None,
        token_channel_model_path=None,
        token_channel_context_width=None,
        token_channel_hidden_size=None,
        token_channel_batch_size=None,
        token_channel_epochs=None,
        token_channel_lr=None,
        token_channel_entropy_threshold=None,
        token_channel_diversity_threshold=None,
        token_channel_split_ratio=None,
        token_channel_seed=None,
    )

    with pytest.raises(ValueError, match="token_channel_train must be a JSON object"):
        resolve_token_channel_train_config(args)

def test_resolve_token_channel_train_config_rejects_invalid_split_ratio(tmp_path):
    from wfcllm.cli.runners import resolve_token_channel_train_config

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"token_channel_train": {"split_ratio": 1.0}}),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        config=config_path,
        dataset=None,
        dataset_path=None,
        lm_model_path=None,
        token_channel_cache_path=None,
        token_channel_model_path=None,
        token_channel_context_width=None,
        token_channel_hidden_size=None,
        token_channel_batch_size=None,
        token_channel_epochs=None,
        token_channel_lr=None,
        token_channel_entropy_threshold=None,
        token_channel_diversity_threshold=None,
        token_channel_split_ratio=None,
        token_channel_seed=None,
    )

    with pytest.raises(ValueError, match=r"split_ratio must be within \(0, 1\)"):
        resolve_token_channel_train_config(args)

def test_resolve_token_channel_train_config_rejects_invalid_diversity_threshold(tmp_path):
    from wfcllm.cli.runners import resolve_token_channel_train_config

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"token_channel_train": {"diversity_threshold": 0}}),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        config=config_path,
        dataset=None,
        dataset_path=None,
        lm_model_path=None,
        token_channel_cache_path=None,
        token_channel_model_path=None,
        token_channel_context_width=None,
        token_channel_hidden_size=None,
        token_channel_batch_size=None,
        token_channel_epochs=None,
        token_channel_lr=None,
        token_channel_entropy_threshold=None,
        token_channel_diversity_threshold=None,
        token_channel_split_ratio=None,
        token_channel_seed=None,
    )

    with pytest.raises(ValueError, match="diversity_threshold must be >= 1"):
        resolve_token_channel_train_config(args)

def test_resolve_token_channel_train_config_rejects_invalid_entropy_threshold(tmp_path):
    from wfcllm.cli.runners import resolve_token_channel_train_config

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"token_channel_train": {"entropy_threshold": -0.1}}),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        config=config_path,
        dataset=None,
        dataset_path=None,
        lm_model_path=None,
        token_channel_cache_path=None,
        token_channel_model_path=None,
        token_channel_context_width=None,
        token_channel_hidden_size=None,
        token_channel_batch_size=None,
        token_channel_epochs=None,
        token_channel_lr=None,
        token_channel_entropy_threshold=None,
        token_channel_diversity_threshold=None,
        token_channel_split_ratio=None,
        token_channel_seed=None,
    )

    with pytest.raises(ValueError, match="entropy_threshold must be >= 0"):
        resolve_token_channel_train_config(args)

def test_resolve_token_channel_train_config_rejects_invalid_dataset(tmp_path):
    from wfcllm.cli.runners import resolve_token_channel_train_config

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"token_channel_train": {"dataset": "custom"}}),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        config=config_path,
        dataset=None,
        dataset_path=None,
        lm_model_path=None,
        token_channel_cache_path=None,
        token_channel_model_path=None,
        token_channel_context_width=None,
        token_channel_hidden_size=None,
        token_channel_batch_size=None,
        token_channel_epochs=None,
        token_channel_lr=None,
        token_channel_entropy_threshold=None,
        token_channel_diversity_threshold=None,
        token_channel_split_ratio=None,
        token_channel_seed=None,
    )

    with pytest.raises(ValueError, match="dataset must be one of: humaneval, mbpp"):
        resolve_token_channel_train_config(args)

@pytest.mark.parametrize(
    ("field_name", "field_value", "expected_message"),
    [
        ("context_width", 0, "context_width must be > 0"),
        ("hidden_size", 0, "hidden_size must be > 0"),
        ("batch_size", 0, "batch_size must be > 0"),
        ("epochs", 0, "epochs must be > 0"),
        ("lr", 0, "lr must be > 0"),
    ],
)
def test_resolve_token_channel_train_config_rejects_non_positive_numeric_knobs(tmp_path,
    field_name,
    field_value,
    expected_message,
):
    from wfcllm.cli.runners import resolve_token_channel_train_config

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"token_channel_train": {field_name: field_value}}),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        config=config_path,
        dataset=None,
        dataset_path=None,
        lm_model_path=None,
        token_channel_cache_path=None,
        token_channel_model_path=None,
        token_channel_context_width=None,
        token_channel_hidden_size=None,
        token_channel_batch_size=None,
        token_channel_epochs=None,
        token_channel_lr=None,
        token_channel_entropy_threshold=None,
        token_channel_diversity_threshold=None,
        token_channel_split_ratio=None,
        token_channel_seed=None,
    )

    with pytest.raises(ValueError, match=expected_message):
        resolve_token_channel_train_config(args)

def test_resolve_token_channel_config_applies_cli_overrides():
    from wfcllm.cli.config_resolver import resolve_token_channel_config

    args = argparse.Namespace(
        token_channel_enabled=True,
        token_channel_mode="lexical-only",
        token_channel_model_path="data/models/token-channel-demo",
        token_channel_context_width=64,
        token_channel_switch_threshold=0.25,
        token_channel_delta=1.5,
        token_channel_ignore_repeated_ngrams=True,
        token_channel_ignore_repeated_prefixes=True,
        token_channel_debug_mode=True,
        token_channel_lexical_min_block_tokens=12,
        token_channel_lexical_retry_decay_start=3,
        token_channel_lexical_retry_disable_after=5,
        token_channel_lexical_gate_probe_tokens=20,
        token_channel_lexical_gate_min_fraction=0.2,
        token_channel_joint_semantic_weight=1.25,
        token_channel_joint_lexical_weight=0.6,
        token_channel_lexical_full_weight_min_positions=48,
        token_channel_joint_threshold=5.0,
    )

    resolved = resolve_token_channel_config(
        {
            "enabled": False,
            "channel_mode": "semantic-only",
            "delta": 2.0,
        },
        args,
    )

    assert resolved.enabled is True
    assert resolved.mode == "lexical-only"
    assert resolved.model_path == "data/models/token-channel-demo"
    assert resolved.context_width == 64
    assert resolved.switch_threshold == pytest.approx(0.25)
    assert resolved.delta == pytest.approx(1.5)
    assert resolved.ignore_repeated_ngrams is True
    assert resolved.ignore_repeated_prefixes is True
    assert resolved.debug_mode is True
    assert resolved.lexical_min_block_tokens == 12
    assert resolved.lexical_retry_decay_start == 3
    assert resolved.lexical_retry_disable_after == 5
    assert resolved.lexical_gate_probe_tokens == 20
    assert resolved.lexical_gate_min_fraction == pytest.approx(0.2)
    assert resolved.joint_semantic_weight == pytest.approx(1.25)
    assert resolved.joint_lexical_weight == pytest.approx(0.6)
    assert resolved.lexical_full_weight_min_positions == 48
    assert resolved.joint_threshold == pytest.approx(5.0)

def test_resolve_token_channel_config_preserves_value_error_for_invalid_joint_config():
    from wfcllm.cli.config_resolver import resolve_token_channel_config

    args = argparse.Namespace(
        token_channel_enabled=True,
        token_channel_mode=None,
        token_channel_model_path=None,
        token_channel_context_width=None,
        token_channel_switch_threshold=None,
        token_channel_delta=None,
        token_channel_ignore_repeated_ngrams=None,
        token_channel_ignore_repeated_prefixes=None,
        token_channel_debug_mode=None,
        token_channel_lexical_min_block_tokens=None,
        token_channel_lexical_retry_decay_start=None,
        token_channel_lexical_retry_disable_after=None,
        token_channel_lexical_gate_probe_tokens=None,
        token_channel_lexical_gate_min_fraction=None,
        token_channel_joint_semantic_weight=None,
        token_channel_joint_lexical_weight=None,
        token_channel_lexical_full_weight_min_positions=None,
        token_channel_joint_threshold=None,
    )

    with pytest.raises(ValueError, match="joint must be a JSON object"):
        resolve_token_channel_config({"joint": 1}, args)

def test_help_lists_token_channel_flags():
    result = subprocess.run(
        ["conda", "run", "-n", "WFCLLM", "python", str(RUN_PY), "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--token-channel-enabled" in result.stdout
    assert "--token-channel-cache-path" in result.stdout
    assert "--token-channel-model-path" in result.stdout
    assert "--token-channel-joint-threshold" in result.stdout
    assert "token-channel-train" in result.stdout

def test_readme_documents_official_token_channel_train_workflow():
    readme_text = README_MD.read_text(encoding="utf-8")

    assert "python run.py --phase token-channel-train" in readme_text
    assert "--dataset-path data/datasets" in readme_text
    assert "重建训练 cache" in readme_text
    assert "data/models/token-channel/model.pt" in readme_text
    assert "data/models/token-channel/metadata.json" in readme_text
    assert "data/models/token-channel/training_evidence.json" in readme_text
    assert "校验 metadata / tokenizer / context_width 等兼容性" in readme_text

def test_readme_limits_token_channel_docs_to_training_workflow():
    readme_text = README_MD.read_text(encoding="utf-8")
    section_start = readme_text.index("### Token-Channel Commands")
    section_end = readme_text.index("---", section_start)
    token_channel_section = readme_text[section_start:section_end]

    assert "python run.py --phase watermark" not in token_channel_section
    assert "python run.py --phase extract" not in token_channel_section
    assert "semantic_prediction" not in token_channel_section
    assert "lexical_z_score" not in token_channel_section
    assert "joint_score" not in token_channel_section
    assert "joint_prediction" not in token_channel_section

