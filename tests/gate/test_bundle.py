from __future__ import annotations

from dataclasses import dataclass, FrozenInstanceError
import json
from pathlib import Path
import types

import pytest
import torch
from torch import nn

from wfcllm.gate.bundle import (
    GateBundle,
    GateBundleManifest,
    StableGatePredictor,
    TokenizerSnapshot,
    quantize_gate_model_dynamic,
)
from wfcllm.gate.bundle import GateValidationThresholds


class TinyGate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2, 2)

    def forward(self, *, input_features: torch.Tensor) -> object:
        logits = self.projection(input_features)
        return types.SimpleNamespace(
            close_logits=logits[:, 0], suitable_logits=logits[:, 1]
        )


def _sha(char: str) -> str:
    return char * 64


def _manifest(**overrides: object) -> GateBundleManifest:
    values: dict[str, object] = {
        "contract_version": "wfcllm-gate-bundle/v1",
        "model_format_version": "wfcllm-gate-model-state/v1",
        "window_contract_version": "python-statement-window/v1",
        "gate_input_contract_version": "wfcllm-gate-input/v1",
        "model_architecture": "tiny-dual-head",
        "base_model_id": "data/models/tiny-local",
        "torch_version": torch.__version__,
        "tokenizer_sha256": _sha("a"),
        "float_model_sha256": _sha("b"),
        "quantized_model_sha256": _sha("c"),
        "validation_summary_sha256": _sha("d"),
        "close_low_threshold": 0.4,
        "close_high_threshold": 0.6,
        "suitable_accept_threshold": 0.8,
        "max_probability_delta": 0.02,
        "max_tokens": 512,
        "training_data_manifest_sha256": _sha("e"),
        "training_key_bank_id": "training-key-bank/v1:sha256:" + _sha("f"),
        "holdout_key_bank_id": "holdout-key-bank/v1:sha256:" + _sha("1"),
        "quantization": "torch-dynamic-qint8-linear",
    }
    values.update(overrides)
    return GateBundleManifest(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "contract",
    ["python-statement-window/v1", "cpp-statement-window/v1", "java-statement-window/v1"],
)
def test_bundle_manifest_accepts_supported_language_contracts(contract: str) -> None:
    assert _manifest(window_contract_version=contract).window_contract_version == contract


def _write_bundle(root: Path) -> Path:
    root.mkdir()
    float_model = TinyGate().eval()
    with torch.no_grad():
        float_model.projection.weight.copy_(torch.eye(2))
        float_model.projection.bias.zero_()
    int8_model = quantize_gate_model_dynamic(float_model)
    torch.save(float_model.state_dict(), root / "gate_float.pt")
    torch.save(int8_model.state_dict(), root / "gate_int8.pt")
    tokenizer = root / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text('{"version":1}\n', encoding="utf-8")
    from wfcllm.gate.bundle import sha256_directory, sha256_file

    float_sha256 = sha256_file(root / "gate_float.pt")
    int8_sha256 = sha256_file(root / "gate_int8.pt")
    summary = {
        "contract_version": "wfcllm-gate-validation/v1",
        "validated": True,
        "decision_agreement": 1.0,
        "float_quantized_accepted_set_agreement": 1.0,
        "formal_accepted_span_consensus": 1.0,
        "suitable_false_positive_rate": 0.0,
        "validation_scope": "cpu",
        "gpu_status": "not_available",
        "mode_names": [
            f"cpu-{precision}-batch-{batch}-{order}"
            for precision in ("float", "int8")
            for batch in (1,)
            for order in ("original",)
        ],
        "threshold_fit_group_digest": _sha("2"),
        "agreement_group_digest": _sha("3"),
        "thresholds": {
            "close_low": 0.4,
            "close_high": 0.6,
            "suitable_accept": 0.8,
            "max_probability_delta": 0.02,
            "max_tokens": 512,
            "threshold_fit_group_digest": _sha("2"),
        },
        "forced_uncertain_count": 0,
        "formal_accepted_spans": [["agreement-group", "example-1", 0, 2]],
        "formal_confusion": {"tp": 1, "tn": 1, "fp": 0, "fn": 0},
        "process_attestations": [
            {
                "mode_name": f"cpu-{precision}-batch-{batch}-{order}",
                "pids": [1000 + index * 2, 1001 + index * 2],
                "device": "cpu",
                "precision": precision,
                "artifact_name": (
                    "gate_float.pt" if precision == "float" else "gate_int8.pt"
                ),
                "artifact_sha256": (
                    float_sha256 if precision == "float" else int8_sha256
                ),
            }
            for index, (precision, batch, order) in enumerate(
                (precision, batch, order)
                for precision in ("float", "int8")
                for batch in (1,)
                for order in ("original",)
            )
        ],
    }
    (root / "validation_summary.json").write_text(
        json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    manifest = _manifest(
        tokenizer_sha256=sha256_directory(tokenizer),
        float_model_sha256=float_sha256,
        quantized_model_sha256=int8_sha256,
        validation_summary_sha256=sha256_file(root / "validation_summary.json"),
    )
    (root / "manifest.json").write_text(manifest.to_json(), encoding="utf-8")
    return root


def test_bundle_rejects_tampered_model_file(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path / "bundle")
    bundle = GateBundle.load(root)
    assert bundle.manifest.contract_version == "wfcllm-gate-bundle/v1"
    (root / "gate_float.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        GateBundle.load(root)


def test_bundle_manifest_binds_all_formal_inputs(tmp_path: Path) -> None:
    manifest = GateBundle.load(_write_bundle(tmp_path / "bundle")).manifest
    assert manifest.window_contract_version == "python-statement-window/v1"
    assert manifest.gate_input_contract_version == "wfcllm-gate-input/v1"
    assert manifest.tokenizer_sha256 and manifest.float_model_sha256
    assert manifest.quantized_model_sha256 and manifest.validation_summary_sha256
    assert manifest.close_low_threshold < manifest.close_high_threshold
    assert manifest.suitable_accept_threshold > 0
    assert manifest.max_tokens == 512
    with pytest.raises(FrozenInstanceError):
        manifest.max_tokens = 1  # type: ignore[misc]


@pytest.mark.parametrize("pollution", ["extra.txt", "missing", "root-symlink", "child-symlink"])
def test_bundle_layout_is_exact_and_symlink_free(tmp_path: Path, pollution: str) -> None:
    root = _write_bundle(tmp_path / "bundle")
    if pollution == "extra.txt":
        (root / "extra.txt").write_text("x", encoding="utf-8")
    elif pollution == "missing":
        (root / "gate_int8.pt").unlink()
    elif pollution == "root-symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(root, target_is_directory=True)
        root = alias
    else:
        target = root / "actual.json"
        target.write_text("{}", encoding="utf-8")
        (root / "tokenizer" / "tokenizer.json").unlink()
        (root / "tokenizer" / "tokenizer.json").symlink_to(target)
    with pytest.raises(ValueError, match="allowlist|symlink|missing"):
        GateBundle.load(root)


def test_manifest_rejects_unknown_sensitive_and_invalid_fields() -> None:
    payload = _manifest().to_dict()
    payload["raw_key"] = "secret"
    with pytest.raises(ValueError, match="schema"):
        GateBundleManifest.from_dict(payload)
    with pytest.raises(ValueError, match="uncertain"):
        _manifest(close_low_threshold=0.5, close_high_threshold=0.5)
    with pytest.raises(ValueError, match="bank"):
        _manifest(training_key_bank_id="train-key-001")
    with pytest.raises(ValueError, match="local"):
        _manifest(base_model_id="https://example.invalid/model")


def test_validation_summary_is_exact_validated_and_bound_to_manifest(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path / "bundle")
    summary_path = root / "validation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["raw_key"] = "secret"
    summary_path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    from wfcllm.gate.bundle import sha256_file

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["validation_summary_sha256"] = sha256_file(summary_path)
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schema"):
        GateBundle.load(root)

    root = _write_bundle(tmp_path / "bundle2")
    summary_path = root / "validation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["validated"] = False
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["validation_summary_sha256"] = sha256_file(summary_path)
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="validated"):
        GateBundle.load(root)

    root = _write_bundle(tmp_path / "bundle3")
    summary_path = root / "validation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["thresholds"]["close_low"] = 0.3
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["validation_summary_sha256"] = sha256_file(summary_path)
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="threshold"):
        GateBundle.load(root)

    root = _write_bundle(tmp_path / "bundle4")
    summary_path = root / "validation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["process_attestations"][0]["artifact_sha256"] = "0" * 64
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["validation_summary_sha256"] = sha256_file(summary_path)
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="process artifact hash"):
        GateBundle.load(root)


def test_bundle_rejects_validated_summary_without_formal_coverage(
    tmp_path: Path,
) -> None:
    root = _write_bundle(tmp_path / "bundle")
    summary_path = root / "validation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["formal_accepted_spans"] = []
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    from wfcllm.gate.bundle import sha256_file

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["validation_summary_sha256"] = sha256_file(summary_path)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="coverage"):
        GateBundle.load(root)


@pytest.mark.parametrize(
    "confusion",
    (
        {"tp": 1, "tn": 0, "fp": 1, "fn": 0},
        {"tp": 0, "tn": 1, "fp": 0, "fn": 1},
    ),
)
def test_bundle_rejects_accept_all_and_reject_all_confusion(
    tmp_path: Path,
    confusion: dict[str, int],
) -> None:
    root = _write_bundle(tmp_path / "bundle")
    summary_path = root / "validation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["formal_confusion"] = confusion
    summary["formal_accepted_spans"] = (
        [
            ["agreement-group", "example-1", 0, 2],
            ["agreement-group", "example-2", 2, 4],
        ]
        if confusion["tp"] + confusion["fp"] == 2
        else []
    )
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    from wfcllm.gate.bundle import sha256_file

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["validation_summary_sha256"] = sha256_file(summary_path)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="nondegenerate"):
        GateBundle.load(root)


def test_safe_model_state_load_never_falls_back_to_pickle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _write_bundle(tmp_path / "bundle")
    calls: list[dict[str, object]] = []

    def fail(*_: object, **kwargs: object) -> object:
        calls.append(dict(kwargs))
        raise TypeError("weights_only unsupported")

    monkeypatch.setattr(torch, "load", fail)
    with pytest.raises(ValueError, match="safe model load"):
        GateBundle.load(root)
    assert calls == [{"map_location": "cpu", "weights_only": True}]


def test_dynamic_quantization_is_real_qint8_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = TinyGate()
    source.train()
    before = {name: value.detach().clone() for name, value in source.state_dict().items()}
    quantized = quantize_gate_model_dynamic(source)
    modules = tuple(quantized.modules())
    assert any("quantized.dynamic" in type(item).__module__ for item in modules)
    assert source.training is True
    assert all(torch.equal(before[name], value) for name, value in source.state_dict().items())

    monkeypatch.setattr(torch.ao.quantization, "quantize_dynamic", lambda *a, **k: TinyGate())
    with pytest.raises(RuntimeError, match="formal bundle"):
        quantize_gate_model_dynamic(TinyGate())


def test_create_is_atomic_safe_and_loadable(tmp_path: Path) -> None:
    seed = _write_bundle(tmp_path / "seed")
    model = TinyGate()
    quantized = quantize_gate_model_dynamic(model)
    float_artifact = tmp_path / "validated_float.pt"
    int8_artifact = tmp_path / "validated_int8.pt"
    torch.save(model.state_dict(), float_artifact)
    torch.save(quantized.state_dict(), int8_artifact)
    summary = _summary_for_artifacts(seed, float_artifact, int8_artifact)
    thresholds = GateValidationThresholds(**summary["thresholds"])
    output = tmp_path / "formal"
    bundle = GateBundle.create(
        root=output,
        validated_float_artifact=float_artifact,
        validated_int8_artifact=int8_artifact,
        tokenizer_source=seed / "tokenizer",
        validation_summary=summary,
        model_architecture="tiny-dual-head",
        base_model_id="data/models/tiny-local",
        thresholds=thresholds,
        training_data_manifest_sha256=_sha("e"),
        training_key_bank_id="training-key-bank/v1:sha256:" + _sha("f"),
        holdout_key_bank_id="holdout-key-bank/v1:sha256:" + _sha("1"),
    )
    assert bundle.root == output
    assert {item.name for item in output.iterdir()} == {
        "manifest.json",
        "gate_float.pt",
        "gate_int8.pt",
        "tokenizer",
        "validation_summary.json",
    }


@pytest.mark.parametrize("kind", ["extra-state", "nonfinite", "complex-nonfinite"])
def test_create_rejects_unsafe_model_state_before_writing(
    tmp_path: Path, kind: str
) -> None:
    seed = _write_bundle(tmp_path / "seed")
    float_artifact = tmp_path / "unsafe_float.pt"
    if kind == "extra-state":
        torch.save({"projection._extra_state": "deployment-secret"}, float_artifact)
    elif kind == "nonfinite":
        torch.save({"projection.weight": torch.tensor([float("nan")])}, float_artifact)
    else:
        torch.save(
            {"projection.weight": torch.tensor([complex(1.0, float("inf"))])},
            float_artifact,
        )
    int8_artifact = seed / "gate_int8.pt"
    summary = _summary_for_artifacts(seed, float_artifact, int8_artifact)
    thresholds = GateValidationThresholds(**summary["thresholds"])

    output = tmp_path / "formal"
    with pytest.raises(ValueError, match="extra state|finite"):
        GateBundle.create(
            root=output,
            validated_float_artifact=float_artifact,
            validated_int8_artifact=int8_artifact,
            tokenizer_source=seed / "tokenizer",
            validation_summary=summary,
            model_architecture="tiny-dual-head",
            base_model_id="data/models/tiny-local",
            thresholds=thresholds,
            training_data_manifest_sha256=_sha("e"),
            training_key_bank_id="training-key-bank/v1:sha256:" + _sha("f"),
            holdout_key_bank_id="holdout-key-bank/v1:sha256:" + _sha("1"),
        )
    assert not output.exists()


def test_create_cleans_staging_when_serialization_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = _write_bundle(tmp_path / "seed")
    float_artifact = seed / "gate_float.pt"
    int8_artifact = seed / "gate_int8.pt"
    summary = _summary_for_artifacts(seed, float_artifact, int8_artifact)
    thresholds = GateValidationThresholds(**summary["thresholds"])
    output = tmp_path / "formal"

    def fail_copy(*_: object, **__: object) -> None:
        raise OSError("disk failure")

    import wfcllm.gate.bundle as bundle_module

    monkeypatch.setattr(bundle_module.shutil, "copy2", fail_copy)
    with pytest.raises(OSError, match="disk failure"):
        GateBundle.create(
            root=output,
            validated_float_artifact=float_artifact,
            validated_int8_artifact=int8_artifact,
            tokenizer_source=seed / "tokenizer",
            validation_summary=summary,
            model_architecture="tiny-dual-head",
            base_model_id="data/models/tiny-local",
            thresholds=thresholds,
            training_data_manifest_sha256=_sha("e"),
            training_key_bank_id="training-key-bank/v1:sha256:" + _sha("f"),
            holdout_key_bank_id="holdout-key-bank/v1:sha256:" + _sha("1"),
        )
    assert not output.exists()
    assert not tuple(tmp_path.glob(".formal.staging-*"))


def _summary_for_artifacts(
    seed: Path, float_artifact: Path, int8_artifact: Path
) -> dict[str, object]:
    from wfcllm.gate.bundle import sha256_file

    summary = json.loads(
        (seed / "validation_summary.json").read_text(encoding="utf-8")
    )
    hashes = {
        "float": sha256_file(float_artifact),
        "int8": sha256_file(int8_artifact),
    }
    for row in summary["process_attestations"]:
        row["artifact_sha256"] = hashes[row["precision"]]
    return summary


@pytest.mark.parametrize(
    "masquerading_weight",
    [torch.tensor([1.0]), torch.tensor([1], dtype=torch.int8)],
)
def test_loader_rejects_non_qint8_artifact_masquerading_as_dynamic_qint8(
    tmp_path: Path, masquerading_weight: torch.Tensor
) -> None:
    root = _write_bundle(tmp_path / "bundle")
    torch.save({"weight": masquerading_weight}, root / "gate_int8.pt")
    from wfcllm.gate.bundle import sha256_file

    int8_hash = sha256_file(root / "gate_int8.pt")
    summary_path = root / "validation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for row in summary["process_attestations"]:
        if row["precision"] == "int8":
            row["artifact_sha256"] = int8_hash
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["quantized_model_sha256"] = int8_hash
    manifest["validation_summary_sha256"] = sha256_file(summary_path)
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not qint8"):
        GateBundle.load(root)


def test_tokenizer_hash_binds_empty_subdirectories(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path / "bundle")
    (root / "tokenizer" / "unexpected-empty").mkdir()
    with pytest.raises(ValueError, match="tokenizer"):
        GateBundle.load(root)


class ControlledAdapter:
    def __init__(
        self,
        *,
        model: nn.Module,
        tokenizer_sha256: str,
        inputs: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        self.model = model
        self.tokenizer_sha256 = tokenizer_sha256
        self.inputs = inputs

    def encode_input(self, serialized_input: str) -> dict[str, torch.Tensor]:
        assert serialized_input
        return {"input_features": torch.tensor([self.inputs])}


@dataclass(slots=True)
class SlotsStateHolder:
    retained_state: object


@dataclass(slots=True)
class SlotsControlledAdapter:
    model: nn.Module
    tokenizer_sha256: str
    inputs: tuple[float, float] = (-1.0, 1.0)
    retained_state: object | None = None
    nested: object | None = None

    def encode_input(self, serialized_input: str) -> dict[str, torch.Tensor]:
        assert serialized_input
        return {"input_features": torch.tensor([self.inputs])}


def _controlled_loader(
    *,
    precision: str,
    state: object,
    tokenizer_snapshot: TokenizerSnapshot,
    manifest: object,
    inputs: tuple[float, float] = (-1.0, 1.0),
) -> ControlledAdapter:
    assert manifest
    assert tokenizer_snapshot.file_bytes("tokenizer.json") == b'{"version":1}\n'
    model: nn.Module = TinyGate().eval()
    if precision == "int8":
        model = quantize_gate_model_dynamic(model)
    model.load_state_dict(state, strict=True)
    return ControlledAdapter(
        model=model,
        tokenizer_sha256=tokenizer_snapshot.sha256,
        inputs=inputs,
    )


def test_bundle_exposes_one_predictor_that_loads_int8_then_float(tmp_path: Path) -> None:
    bundle = GateBundle.load(_write_bundle(tmp_path / "bundle"))
    loads: list[str] = []

    def loader(
        *,
        precision: str,
        state: object,
        tokenizer_snapshot: TokenizerSnapshot,
        manifest: object,
    ) -> ControlledAdapter:
        assert state and manifest is bundle.manifest
        loads.append(precision)
        return _controlled_loader(
            precision=precision,
            state=state,
            tokenizer_snapshot=tokenizer_snapshot,
            manifest=manifest,
        )

    predictor = bundle.stable_predictor(loader)
    assert isinstance(predictor, StableGatePredictor)
    assert loads == ["int8", "float"]
    assert predictor.predict("formal input").stable is True


def test_stable_predictor_runs_int8_then_float_and_closes_on_disagreement(
    tmp_path: Path,
) -> None:
    bundle = GateBundle.load(_write_bundle(tmp_path / "bundle"))
    order: list[str] = []

    def loader(**kwargs: object) -> ControlledAdapter:
        precision = str(kwargs["precision"])
        order.append(precision)
        probability_inputs = {
            "int8": (0.45, 2.2),
            "float": (0.36, 2.2),
        }
        return _controlled_loader(
            **kwargs, inputs=probability_inputs[precision]  # type: ignore[arg-type]
        )

    predictor = bundle.stable_predictor(loader)
    scores = predictor.predict("formal input")
    assert order == ["int8", "float"]
    assert scores.stable is False
    assert scores.decision_agreement is False
    assert scores.close_probability == pytest.approx(0.61, abs=0.002)


def test_stable_predictor_rejects_probability_delta_and_bad_outputs(
    tmp_path: Path,
) -> None:
    bundle = GateBundle.load(_write_bundle(tmp_path / "bundle"))

    def drifting_loader(**kwargs: object) -> ControlledAdapter:
        precision = str(kwargs["precision"])
        inputs = (-1.4, 1.4 if precision == "int8" else 1.75)
        return _controlled_loader(**kwargs, inputs=inputs)  # type: ignore[arg-type]

    predictor = bundle.stable_predictor(drifting_loader)
    assert predictor.predict("x").stable is False

    def bad_loader(**kwargs: object) -> ControlledAdapter:
        return _controlled_loader(
            **kwargs, inputs=(float("nan"), 1.4)  # type: ignore[arg-type]
        )

    bad = bundle.stable_predictor(bad_loader)
    with pytest.raises(ValueError, match="finite"):
        bad.predict("x")


def test_loaded_bundle_uses_verified_snapshots_after_paths_are_replaced(
    tmp_path: Path,
) -> None:
    root = _write_bundle(tmp_path / "bundle")
    bundle = GateBundle.load(root)
    (root / "gate_float.pt").write_bytes(b"replaced")
    (root / "gate_int8.pt").write_bytes(b"replaced")
    (root / "tokenizer" / "tokenizer.json").write_text(
        '{"tampered":true}\n', encoding="utf-8"
    )
    predictor = bundle.stable_predictor(_controlled_loader)
    assert predictor.predict("formal input").stable is True


def test_bundle_rejects_constant_or_state_ignoring_loaders(tmp_path: Path) -> None:
    bundle = GateBundle.load(_write_bundle(tmp_path / "bundle"))

    class ConstantLoaderResult:
        def predict_probabilities(self, serialized_input: str) -> tuple[float, float]:
            return 0.2, 0.9

    with pytest.raises(ValueError, match="model"):
        bundle.stable_predictor(lambda **_: ConstantLoaderResult())

    def ignoring_loader(**kwargs: object) -> ControlledAdapter:
        model: nn.Module = TinyGate().eval()
        if kwargs["precision"] == "int8":
            model = quantize_gate_model_dynamic(model)
        snapshot = kwargs["tokenizer_snapshot"]
        assert isinstance(snapshot, TokenizerSnapshot)
        return ControlledAdapter(model=model, tokenizer_sha256=snapshot.sha256)

    with pytest.raises(ValueError, match="state"):
        bundle.stable_predictor(ignoring_loader)


def test_predictor_deserializes_each_artifact_once_and_keeps_only_runtime_seals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import wfcllm.gate.bundle as bundle_module

    original_load = bundle_module.torch.load
    load_count = 0

    def counting_load(*args: object, **kwargs: object) -> object:
        nonlocal load_count
        load_count += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(bundle_module.torch, "load", counting_load)
    bundle = GateBundle.load(_write_bundle(tmp_path / "bundle"))
    predictor = bundle.stable_predictor(_controlled_loader)
    assert load_count == 2
    assert not hasattr(predictor._quantized, "expected_state")
    assert not hasattr(predictor._float, "expected_state")

    assert predictor.predict("formal input").stable is True
    second = bundle.stable_predictor(_controlled_loader)
    assert load_count == 2
    assert second.predict("formal input").stable is True


def test_runtime_seal_rejects_in_place_parameter_mutation(tmp_path: Path) -> None:
    bundle = GateBundle.load(_write_bundle(tmp_path / "bundle"))
    predictor = bundle.stable_predictor(_controlled_loader)
    parameter = next(predictor._float.adapter.model.parameters())
    with torch.no_grad():
        parameter.add_(1.0)
    with pytest.raises(ValueError, match="metadata or version"):
        predictor.predict("formal input")


def test_runtime_seal_rejects_module_replacement(tmp_path: Path) -> None:
    bundle = GateBundle.load(_write_bundle(tmp_path / "bundle"))
    predictor = bundle.stable_predictor(_controlled_loader)
    predictor._float.adapter.model.projection = nn.Linear(2, 2)
    with pytest.raises(ValueError, match="module structure"):
        predictor.predict("formal input")


def test_adapter_cannot_retain_transient_artifact_state(tmp_path: Path) -> None:
    bundle = GateBundle.load(_write_bundle(tmp_path / "bundle"))

    def retaining_loader(**kwargs: object) -> ControlledAdapter:
        adapter = _controlled_loader(**kwargs)  # type: ignore[arg-type]
        adapter.retained_state = kwargs["state"]  # type: ignore[attr-defined]
        return adapter

    with pytest.raises(ValueError, match="retain mutable artifact state"):
        bundle.stable_predictor(retaining_loader)


def test_adapter_model_cannot_retain_transient_artifact_state(tmp_path: Path) -> None:
    bundle = GateBundle.load(_write_bundle(tmp_path / "bundle"))

    def retaining_loader(**kwargs: object) -> ControlledAdapter:
        adapter = _controlled_loader(**kwargs)  # type: ignore[arg-type]
        adapter.model.retained_state = kwargs["state"]
        return adapter

    with pytest.raises(ValueError, match="retain mutable artifact state"):
        bundle.stable_predictor(retaining_loader)


def test_slots_adapter_cannot_retain_transient_artifact_state(tmp_path: Path) -> None:
    bundle = GateBundle.load(_write_bundle(tmp_path / "bundle"))

    def retaining_loader(**kwargs: object) -> SlotsControlledAdapter:
        base = _controlled_loader(**kwargs)  # type: ignore[arg-type]
        return SlotsControlledAdapter(
            model=base.model,
            tokenizer_sha256=base.tokenizer_sha256,
            retained_state=kwargs["state"],
        )

    with pytest.raises(ValueError, match="retain mutable artifact state"):
        bundle.stable_predictor(retaining_loader)


def test_nested_slots_holder_cannot_retain_transient_artifact_state(
    tmp_path: Path,
) -> None:
    bundle = GateBundle.load(_write_bundle(tmp_path / "bundle"))

    def retaining_loader(**kwargs: object) -> SlotsControlledAdapter:
        base = _controlled_loader(**kwargs)  # type: ignore[arg-type]
        return SlotsControlledAdapter(
            model=base.model,
            tokenizer_sha256=base.tokenizer_sha256,
            nested=SlotsStateHolder(kwargs["state"]),
        )

    with pytest.raises(ValueError, match="retain mutable artifact state"):
        bundle.stable_predictor(retaining_loader)


def test_normal_slots_adapter_is_a_valid_controlled_path(tmp_path: Path) -> None:
    bundle = GateBundle.load(_write_bundle(tmp_path / "bundle"))

    def loader(**kwargs: object) -> SlotsControlledAdapter:
        base = _controlled_loader(**kwargs)  # type: ignore[arg-type]
        return SlotsControlledAdapter(
            model=base.model,
            tokenizer_sha256=base.tokenizer_sha256,
        )

    assert bundle.stable_predictor(loader).predict("formal input").stable is True


def test_runtime_seal_rejects_quantized_packed_object_replacement(
    tmp_path: Path,
) -> None:
    bundle = GateBundle.load(_write_bundle(tmp_path / "bundle"))
    predictor = bundle.stable_predictor(_controlled_loader)
    packed_module = predictor._quantized.adapter.model.projection._packed_params
    packed_module._packed_params = object()
    with pytest.raises(ValueError, match="packed structure"):
        predictor.predict("formal input")
