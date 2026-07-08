"""Tests for the pretrain phase: stage selection + fail-fast on missing encoder."""
from __future__ import annotations

import argparse

import pytest


def test_pretrain_config_default_stages_both():
    from wfcllm.pretrain.config import PretrainConfig
    cfg = PretrainConfig()
    assert cfg.stages == ["encoder", "lexical"]


def test_pretrain_config_normalizes_stage_order():
    """Even if user passes ['lexical', 'encoder'], pipeline runs in canonical order."""
    from wfcllm.pretrain.config import PretrainConfig
    cfg = PretrainConfig(stages=["lexical", "encoder"])
    assert cfg.stages == ["encoder", "lexical"]


def test_pretrain_config_rejects_unknown_stage():
    from wfcllm.pretrain.config import PretrainConfig
    with pytest.raises(ValueError, match="unknown stage"):
        PretrainConfig(stages=["encoder", "bogus"])


def test_pipeline_runs_only_encoder_when_stages_encoder(monkeypatch):
    """--stages encoder dispatches only the encoder runner."""
    from wfcllm.pretrain.pipeline import PretrainPipeline
    from wfcllm.pretrain.config import PretrainConfig

    calls: list[str] = []

    def fake_encoder(args, state):
        calls.append("encoder")
        return 0

    def fake_lexical(args, state):
        calls.append("lexical")
        return 0

    monkeypatch.setattr("wfcllm.pretrain.pipeline.run_encoder", fake_encoder)
    monkeypatch.setattr("wfcllm.pretrain.pipeline.run_token_channel_train", fake_lexical)

    pipeline = PretrainPipeline(PretrainConfig(stages=["encoder"]))
    rc = pipeline.run(args=argparse.Namespace(), state=_DummyState(encoder_done=False))
    assert rc == 0
    assert calls == ["encoder"]


def test_pipeline_lexical_failfast_when_encoder_not_done(monkeypatch, capsys):
    """--stages lexical with no encoder ckpt prints error and returns 1."""
    from wfcllm.pretrain.pipeline import PretrainPipeline
    from wfcllm.pretrain.config import PretrainConfig

    monkeypatch.setattr(
        "wfcllm.pretrain.pipeline.run_encoder",
        lambda *a, **k: pytest.fail("encoder runner must not be called when stages=['lexical']"),
    )
    monkeypatch.setattr(
        "wfcllm.pretrain.pipeline.run_token_channel_train",
        lambda *a, **k: pytest.fail("lexical runner must not be reached when encoder not done"),
    )

    pipeline = PretrainPipeline(PretrainConfig(stages=["lexical"]))
    rc = pipeline.run(args=argparse.Namespace(), state=_DummyState(encoder_done=False))
    assert rc == 1
    captured = capsys.readouterr()
    assert "encoder" in captured.err.lower() and "未完成" in captured.err


def test_pipeline_runs_both_in_order(monkeypatch):
    """--stages encoder lexical (or default) dispatches encoder then lexical."""
    from wfcllm.pretrain.pipeline import PretrainPipeline
    from wfcllm.pretrain.config import PretrainConfig

    calls: list[str] = []
    monkeypatch.setattr(
        "wfcllm.pretrain.pipeline.run_encoder",
        lambda args, state: (calls.append("encoder") or 0),
    )
    monkeypatch.setattr(
        "wfcllm.pretrain.pipeline.run_token_channel_train",
        lambda args, state: (calls.append("lexical") or 0),
    )
    pipeline = PretrainPipeline(PretrainConfig(stages=["encoder", "lexical"]))
    rc = pipeline.run(args=argparse.Namespace(), state=_DummyState(encoder_done=True))
    assert rc == 0
    assert calls == ["encoder", "lexical"]


class _DummyState:
    def __init__(self, *, encoder_done: bool):
        self._encoder_done = encoder_done

    def is_done(self, phase: str) -> bool:
        return self._encoder_done if phase == "encoder" else False


def test_cli_phase_pretrain_dispatches_pipeline(monkeypatch, tmp_path):
    """`run.py --phase legacy-pretrain --legacy --stages encoder` calls run_encoder."""
    from wfcllm.cli.entry import main

    state_path = tmp_path / "run_state.json"
    monkeypatch.setattr("wfcllm.cli.entry.DEFAULT_STATE_FILE", state_path)

    calls: list[str] = []
    monkeypatch.setattr(
        "wfcllm.pretrain.pipeline.run_encoder",
        lambda args, state: (calls.append("encoder") or 0),
    )
    monkeypatch.setattr(
        "wfcllm.pretrain.pipeline.run_token_channel_train",
        lambda args, state: (calls.append("lexical") or 0),
    )

    rc = main(["--phase", "legacy-pretrain", "--legacy", "--stages", "encoder"])
    assert rc == 0
    assert calls == ["encoder"]


def test_cli_phase_pretrain_default_runs_both_in_order(monkeypatch, tmp_path):
    """`run.py --phase legacy-pretrain --legacy` runs encoder then lexical."""
    from wfcllm.cli.entry import main

    state_path = tmp_path / "run_state.json"
    monkeypatch.setattr("wfcllm.cli.entry.DEFAULT_STATE_FILE", state_path)

    calls: list[str] = []

    def fake_encoder(args, state):
        # Simulate encoder completing successfully so the lexical stage proceeds.
        state.mark_done("encoder", checkpoint="/tmp/fake")
        calls.append("encoder")
        return 0

    monkeypatch.setattr("wfcllm.pretrain.pipeline.run_encoder", fake_encoder)
    monkeypatch.setattr(
        "wfcllm.pretrain.pipeline.run_token_channel_train",
        lambda args, state: (calls.append("lexical") or 0),
    )

    rc = main(["--phase", "legacy-pretrain", "--legacy"])
    assert rc == 0
    assert calls == ["encoder", "lexical"]
