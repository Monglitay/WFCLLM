"""Runner-level wiring for gated generation/detection pipeline construction.

These tests stub the model boundaries (HF program generator, gate runtime
bundle, semantic scorer, window extractor) and assert that
``wfcllm.cli.runners`` wires the correct language rewriters and calibration
options into the real pipeline objects.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from wfcllm.cli.config_resolver import resolve_method_config
from wfcllm.cli.runners import (
    _build_local_gated_detection_pipeline,
    _build_local_gated_generation_pipeline,
    _safe_tree_hash,
)
from wfcllm.generation.window_rewriter import (
    KeyBlindAstEquivalentWindowRewriter,
    KeyBlindCppEquivalentWindowRewriter,
    KeyBlindJavaEquivalentWindowRewriter,
)
from wfcllm.method.presets import GATED_SEMANTIC_WINDOW_V1_NAME, load_method_preset
from wfcllm.windowing.partitioner import GateThresholds, WindowPartitioner


ROOT = Path(__file__).resolve().parents[2]


class _FakeRuntimeGateBundle:
    def __init__(
        self,
        *,
        root: Path,
        base_model_path: Path,
        bundle_sha256: str,
        max_tokens: int,
        window_contract_version: str,
    ) -> None:
        self.manifest = SimpleNamespace(
            window_contract_version=window_contract_version,
            gate_input_contract_version="wfcllm-gate-input/v1",
            tokenizer_sha256="b" * 64,
        )


class _FakeWindowExtractor:
    """Stands in for GatedWindowExtractor at the model boundary."""

    def __init__(self, bundle: object, **_kwargs: object) -> None:
        contract = bundle.manifest.window_contract_version
        self.window_contract_version = contract
        self.partitioner = WindowPartitioner(
            predictor=SimpleNamespace(),
            thresholds=GateThresholds(
                close_low=0.2, close_high=0.8, suitable_accept=0.5
            ),
            tokenizer_counter=len,
            window_contract_version=contract,
        )
        self.unit_extractor = SimpleNamespace(kind="fake-unit-extractor")

    def extract(self, final_code: str):
        return SimpleNamespace(windows=())


class _FakeProgramGenerator:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def generate_program(self, *, prompt: str, sample_id: str) -> str:
        return prompt


class _FakeDatasetAdapter:
    def supports(self, language: str) -> bool:
        return True

    def iter_samples(self, language: str | None = None):
        for index in range(2):
            yield SimpleNamespace(task_id=f"ds/{index}", prompt=f"prompt {index}\n")


def _patch_model_boundaries(monkeypatch) -> None:
    monkeypatch.setattr(
        "wfcllm.gate.production.LocalCandidateRuntimeGateBundle",
        _FakeRuntimeGateBundle,
    )
    monkeypatch.setattr(
        "wfcllm.gate.production.LocalHFProgramGenerator", _FakeProgramGenerator
    )
    monkeypatch.setattr(
        "wfcllm.gate.production.build_local_semantic_window_scorer",
        lambda options, key: SimpleNamespace(score=lambda **kwargs: None),
    )
    monkeypatch.setattr(
        "wfcllm.gate.production.local_semantic_runtime_hash",
        lambda options: "c" * 64,
    )
    monkeypatch.setattr(
        "wfcllm.detection.gated_windows.GatedWindowExtractor",
        _FakeWindowExtractor,
    )
    monkeypatch.setattr(
        "wfcllm.datasets.get", lambda name: _FakeDatasetAdapter()
    )


def _write_candidate_bundle(run_dir: Path, manifest_extra: dict) -> None:
    candidate = run_dir / "gate-train" / "candidate_bundle"
    candidate.mkdir(parents=True)
    (candidate / "gate_float.pt").write_bytes(b"weights")
    manifest = {
        "schema_version": "wfcllm-gate-train-candidate/v1",
        "candidate_bundle_sha256": _safe_tree_hash(candidate),
        "config_hash": "0" * 64,
        **manifest_extra,
    }
    (run_dir / "gate-train" / "candidate_bundle_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def _experiment_config(name: str) -> dict:
    config_path = ROOT / "configs" / "wfcllm" / "experiments" / name
    return resolve_method_config(
        json.loads(config_path.read_text(encoding="utf-8"))
    )


def _runner_args(tmp_path: Path, config: dict) -> argparse.Namespace:
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    deployment = tmp_path / "deployment.key"
    deployment.write_bytes(b"deployment-secret")
    for name in ("generation-model", "encoder-model", "gate-base-model"):
        (tmp_path / name).mkdir(exist_ok=True)
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text("", encoding="utf-8")
    return argparse.Namespace(
        _config_cache=config,
        run_dir=str(run_dir),
        run_id=None,
        secret_key_file=str(deployment),
        secret_key_env=None,
        gate_source_catalog=str(catalog),
        generation_model_path=str(tmp_path / "generation-model"),
        semantic_encoder_model_path=str(tmp_path / "encoder-model"),
        gate_base_model_path=str(tmp_path / "gate-base-model"),
        model_device="cpu",
        gate_device="cpu",
        state_file=str(tmp_path / "state.json"),
        dataset=None,
        language=None,
        dataset_path=None,
        sample_offset=None,
        sample_limit=2,
    )


_EXPERIMENTAL_MANIFEST_MARKERS = {
    "experimental_only": True,
    "diagnostic_only": True,
    "not_official_method": True,
    "formal_eligible": False,
}
_FORMAL_MANIFEST_MARKERS = {
    "diagnostic_test_backend": False,
    "formal_eligible": True,
}


def _build_fast_experimental_generation(tmp_path, monkeypatch, config_name: str):
    config = _experiment_config(config_name)
    args = _runner_args(tmp_path, config)
    _write_candidate_bundle(
        Path(args.run_dir), dict(_EXPERIMENTAL_MANIFEST_MARKERS)
    )
    _patch_model_boundaries(monkeypatch)
    return _build_local_gated_generation_pipeline(args)


def test_cpp_fast_experimental_generation_uses_cpp_rewriter(
    tmp_path, monkeypatch
) -> None:
    pipeline = _build_fast_experimental_generation(
        tmp_path, monkeypatch, "cpp_humanevalpack_fast.json"
    )

    generator = pipeline._generator
    assert isinstance(generator._rewriter, KeyBlindCppEquivalentWindowRewriter)
    assert generator._accept_certified_rewrite_without_hit is True


def test_java_fast_experimental_generation_uses_java_rewriter(
    tmp_path, monkeypatch
) -> None:
    pipeline = _build_fast_experimental_generation(
        tmp_path, monkeypatch, "java_humanevalpack_fast.json"
    )

    generator = pipeline._generator
    assert isinstance(generator._rewriter, KeyBlindJavaEquivalentWindowRewriter)
    assert generator._accept_certified_rewrite_without_hit is True


def test_python_non_fast_generation_uses_ast_rewriter_without_certified_accept(
    tmp_path, monkeypatch
) -> None:
    config = _experiment_config("python_humaneval_fast.json")
    assert config["method"]["gate"].get("fast_experimental") is not True
    args = _runner_args(tmp_path, config)
    _write_candidate_bundle(Path(args.run_dir), dict(_FORMAL_MANIFEST_MARKERS))
    _patch_model_boundaries(monkeypatch)

    pipeline = _build_local_gated_generation_pipeline(args)

    generator = pipeline._generator
    assert isinstance(generator._rewriter, KeyBlindAstEquivalentWindowRewriter)
    assert generator._accept_certified_rewrite_without_hit is False


def _detection_config_with_external_bundle(tmp_path: Path) -> dict:
    bundle = tmp_path / "bundle"
    (bundle / "tokenizer").mkdir(parents=True)
    (bundle / "gate_float.pt").write_bytes(b"formal")
    (bundle / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    config = load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME).to_dict()
    config["method"]["gate"]["bundle_path"] = str(bundle)
    config["method"]["gate"]["bundle_sha256"] = _safe_tree_hash(bundle)
    return config


def _build_detection_pipeline(tmp_path, monkeypatch, config: dict):
    args = _runner_args(tmp_path, config)
    setattr(args, "_gated_negative_corpus_hash", "e" * 64)
    _patch_model_boundaries(monkeypatch)
    return _build_local_gated_detection_pipeline(args)


def test_detection_pipeline_calibration_pool_flag_defaults_to_false(
    tmp_path, monkeypatch
) -> None:
    config = _detection_config_with_external_bundle(tmp_path)
    assert "pool_excludes_insufficient" not in config["calibration"]

    pipeline = _build_detection_pipeline(tmp_path, monkeypatch, config)

    assert pipeline._calibration_pool_excludes_insufficient is False


def test_detection_pipeline_calibration_pool_flag_wired_from_config(
    tmp_path, monkeypatch
) -> None:
    config = _detection_config_with_external_bundle(tmp_path)
    config["calibration"]["pool_excludes_insufficient"] = True

    pipeline = _build_detection_pipeline(tmp_path, monkeypatch, config)

    assert pipeline._calibration_pool_excludes_insufficient is True
