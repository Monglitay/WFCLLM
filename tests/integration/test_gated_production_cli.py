from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from wfcllm.cli.arguments import build_parser
from wfcllm.cli import entry
from wfcllm.cli import runners


def test_parser_exposes_single_gpu_gated_runtime_paths(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "--state-file", str(tmp_path / "state.json"),
            "--gate-source-catalog", str(tmp_path / "sources.jsonl"),
            "--generation-model-path", str(tmp_path / "generator"),
            "--rewrite-model-path", str(tmp_path / "rewriter"),
            "--semantic-encoder-model-path", str(tmp_path / "semantic"),
            "--semantic-encoder-checkpoint-path", str(tmp_path / "semantic.pt"),
            "--semantic-whitening-path", str(tmp_path / "whitening.pt"),
            "--gate-base-model-path", str(tmp_path / "gate-base"),
            "--model-device", "cuda",
            "--gate-device", "cuda",
            "--detector-device", "cpu",
            "--gate-cache-dir", str(tmp_path / "cache"),
            "--gate-batch-size", "12",
            "--gate-epochs", "10",
            "--gate-early-stopping-patience", "10",
            "--gate-resume-checkpoint", str(tmp_path / "checkpoint.pt"),
        ]
    )

    assert args.state_file == str(tmp_path / "state.json")
    assert args.gate_source_catalog == str(tmp_path / "sources.jsonl")
    assert args.generation_model_path == str(tmp_path / "generator")
    assert args.rewrite_model_path == str(tmp_path / "rewriter")
    assert args.semantic_encoder_model_path == str(tmp_path / "semantic")
    assert args.gate_base_model_path == str(tmp_path / "gate-base")
    assert args.gate_batch_size == 12
    assert args.gate_epochs == 10
    assert args.gate_early_stopping_patience == 10


def test_local_hf_runtime_options_prefer_explicit_training_overrides(
    tmp_path: Path,
) -> None:
    source_catalog = tmp_path / "sources.jsonl"
    source_catalog.write_text("", encoding="utf-8")
    generation_model = tmp_path / "generator"
    semantic_model = tmp_path / "semantic"
    gate_model = tmp_path / "gate-base"
    cache_dir = tmp_path / "cache"
    for path in (generation_model, semantic_model, gate_model, cache_dir):
        path.mkdir()
    args = build_parser().parse_args(
        [
            "--gate-source-catalog", str(source_catalog),
            "--generation-model-path", str(generation_model),
            "--semantic-encoder-model-path", str(semantic_model),
            "--gate-base-model-path", str(gate_model),
            "--gate-cache-dir", str(cache_dir),
            "--gate-epochs", "10",
            "--gate-early-stopping-patience", "10",
        ]
    )

    options = runners._local_hf_runtime_options(
        args,
        {
            "method": {
                "rewrite": {
                    "temperature": 0.2,
                    "top_p": 0.95,
                    "max_new_tokens": 16,
                    "generation_attempts": 3,
                }
            },
            "semantic_lsh": {
                "lsh_d": 1,
                "rule_name": "keyed_text_region",
            },
            "gate_train": {
                "max_epochs": 4,
                "early_stopping_patience": 1,
            }
        },
        "gate-train",
    )

    assert options.gate_epochs == 10
    assert options.gate_early_stopping_patience == 10
    assert options.rewrite_temperature == 0.2
    assert options.rewrite_top_p == 0.95
    assert options.rewrite_max_new_tokens == 16
    assert options.rewrite_generation_attempts == 3
    assert options.lsh_dimension == 1
    assert options.semantic_evidence_rule == "keyed_text_region"


def test_formal_runtime_rejects_keyed_text_region_implicit_carrier(
    tmp_path: Path,
) -> None:
    source_catalog = tmp_path / "sources.jsonl"
    source_catalog.write_text("", encoding="utf-8")
    paths = [tmp_path / name for name in ("generator", "semantic", "gate-base", "cache")]
    for path in paths:
        path.mkdir()
    args = build_parser().parse_args(
        [
            "--gate-source-catalog", str(source_catalog),
            "--generation-model-path", str(paths[0]),
            "--semantic-encoder-model-path", str(paths[1]),
            "--gate-base-model-path", str(paths[2]),
            "--gate-cache-dir", str(paths[3]),
        ]
    )

    with pytest.raises(ValueError, match="formal runtime requires semantic_lsh"):
        runners._local_hf_runtime_options(
            args,
            {
                "method": {
                    "gate": {"require_validated": True},
                    "rewrite": {},
                },
                "semantic_lsh": {
                    "lsh_d": 1,
                    "rule_name": "keyed_text_region",
                },
                "gate_train": {},
            },
            "gate-data",
        )


def test_formal_runtime_reads_d12_and_reference_cosine_threshold(
    tmp_path: Path,
) -> None:
    source_catalog = tmp_path / "sources.jsonl"
    source_catalog.write_text("", encoding="utf-8")
    paths = [tmp_path / name for name in ("generator", "semantic", "gate-base", "cache")]
    for path in paths:
        path.mkdir()
    args = build_parser().parse_args(
        [
            "--gate-source-catalog", str(source_catalog),
            "--generation-model-path", str(paths[0]),
            "--semantic-encoder-model-path", str(paths[1]),
            "--gate-base-model-path", str(paths[2]),
            "--gate-cache-dir", str(paths[3]),
        ]
    )

    options = runners._local_hf_runtime_options(
        args,
        {
            "method": {
                "gate": {"require_validated": True},
                "rewrite": {},
                "semantic": {
                    "lsh": {"d": 12},
                    "preservation": {
                        "rule": "codet5-cosine-to-original/v1",
                        "threshold": 0.9,
                    },
                },
            },
            "semantic_lsh": {"lsh_d": 12, "rule_name": "semantic_lsh"},
            "gate_train": {},
        },
        "gate-data",
    )

    assert options.lsh_dimension == 12
    assert options.semantic_preservation_threshold == pytest.approx(0.9)


def test_entry_uses_explicit_state_file(monkeypatch, tmp_path: Path) -> None:
    captured = []

    class _State:
        def __init__(self, path):
            captured.append(path)

        def status(self):
            return {phase: {"done": False} for phase in entry.ALL_PHASES}

    monkeypatch.setattr(entry, "RunStateManager", _State)

    assert entry.main(["--status", "--state-file", str(tmp_path / "state.json")]) == 0
    assert captured == [tmp_path / "state.json"]


def test_formal_dependencies_select_local_hf_runtime(monkeypatch, tmp_path: Path) -> None:
    source_manifest = tmp_path / "manifest.json"
    source_catalog = tmp_path / "catalog.jsonl"
    training = tmp_path / "training.json"
    holdout = tmp_path / "holdout.json"
    for path in (source_manifest, source_catalog, training, holdout):
        path.write_text("{}", encoding="utf-8")
    for name in ("generator", "semantic", "gate-base"):
        (tmp_path / name).mkdir()
    captured = {}

    def _build(**kwargs):
        captured.update(kwargs)
        return Namespace(diagnostic_test_backend=False)

    monkeypatch.setattr("wfcllm.gate.dependencies.build_local_gate_dependencies", _build)
    args = Namespace(
        _config_cache={
            "method": {"name": "gated_semantic_window_v1"},
            "gate_train": {"base_encoder_id": "unused"},
        },
        gate_source_manifest=str(source_manifest),
        gate_source_catalog=str(source_catalog),
        training_key_bank_file=str(training),
        training_key_bank_env=None,
        holdout_key_bank_file=str(holdout),
        holdout_key_bank_env=None,
        generation_model_path=str(tmp_path / "generator"),
        rewrite_model_path=None,
        semantic_encoder_model_path=str(tmp_path / "semantic"),
        semantic_encoder_checkpoint_path=None,
        semantic_whitening_path=None,
        gate_base_model_path=str(tmp_path / "gate-base"),
        model_device="cuda",
        gate_device="cuda",
        gate_cache_dir=str(tmp_path / "cache"),
    )

    dependencies = runners._formal_gate_dependencies(args, "gate-data")

    assert dependencies.diagnostic_test_backend is False
    assert captured["adapter_name"] == "local-hf-v1"
    assert captured["adapter_options"].source_catalog == source_catalog
    assert captured["base_model_path"] == tmp_path / "gate-base"


def test_real_gated_generate_never_succeeds_without_runtime_model() -> None:
    with pytest.raises(ValueError, match="generation-model-path"):
        runners._optional_gated_generation_pipeline(
            Namespace(generation_model_path=None)
        )
