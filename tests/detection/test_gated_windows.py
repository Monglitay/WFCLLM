from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from wfcllm.detection.config import GATED_DETECTOR_MODE, GatedDetectionConfig
from wfcllm.detection.gated_windows import GatedWindowExtractor
from wfcllm.windowing import GateScores


class _StablePredictor:
    def predict(self, _serialized_input: str) -> GateScores:
        return GateScores(
            close_probability=0.9,
            suitable_probability=0.9,
            stable=True,
            precision_delta=0.0,
            decision_agreement=True,
        )


@dataclass
class _ValidatedFakeBundle:
    root: Path
    bundle_sha256: str = "a" * 64
    tokenizer_sha256: str = "b" * 64

    def __post_init__(self) -> None:
        self.validation_summary = {"validated": True}
        self.manifest = SimpleNamespace(
            window_contract_version="python-statement-window/v1",
            gate_input_contract_version="wfcllm-gate-input/v1",
            tokenizer_sha256=self.tokenizer_sha256,
            close_low_threshold=0.4,
            close_high_threshold=0.6,
            suitable_accept_threshold=0.8,
            max_tokens=512,
        )
        self.stable_gate_predictor = _StablePredictor()
        self.tokenizer_counter = lambda text: len(text.encode("utf-8"))


def _config(tmp_path: Path, **overrides: object) -> GatedDetectionConfig:
    values: dict[str, object] = {
        "gate_bundle_path": tmp_path / "bundle",
        "gate_bundle_sha256": "a" * 64,
        "window_contract_version": "python-statement-window/v1",
        "gate_input_contract_version": "wfcllm-gate-input/v1",
        "semantic_encoder_hash": "c" * 64,
        "lsh_config_hash": "d" * 64,
        "minimum_reliable_windows": 2,
        "calibration_artifact_path": tmp_path / "calibration.json",
    }
    values.update(overrides)
    return GatedDetectionConfig(**values)  # type: ignore[arg-type]


def _extract(tmp_path: Path, source: str, **bundle_overrides: object):
    bundle = _ValidatedFakeBundle(
        root=tmp_path / "bundle", **bundle_overrides  # type: ignore[arg-type]
    )
    return GatedWindowExtractor(bundle, _config(tmp_path)).extract(source)


def test_gated_detection_config_is_separate_and_strict(tmp_path: Path) -> None:
    config = _config(tmp_path)

    assert config.detector_mode == GATED_DETECTOR_MODE
    assert config.minimum_reliable_windows == 2
    with pytest.raises(ValueError, match="gate_bundle_sha256"):
        _config(tmp_path, gate_bundle_sha256="not-a-digest")
    with pytest.raises(ValueError, match="window_contract_version"):
        _config(tmp_path, window_contract_version="other")
    with pytest.raises(ValueError, match="minimum_reliable_windows"):
        _config(tmp_path, minimum_reliable_windows=0)


@pytest.mark.parametrize(
    "source",
    [
        "def f(c):\n    x = 1; y = 2\n    if c: a = 3; b = 4\n",
        "def f(xs):\n    if xs:\n        for x in xs:\n            use(x)\n",
        (
            "def f():\n    try:\n        work()\n    except ValueError:\n"
            "        recover()\n    finally:\n        close()\n"
        ),
        "def f(value):\n    match value:\n        case 0:\n            return '零'\n",
        "async def f(items):\n    async for item in items:\n        await use(item)\n",
        "def f(a, b):\n    value = (a +\n             b)\n    return value\n",
        "def f():\n    import os\n    assert True\n    pass\n    return 'λ'\n",
    ],
)
def test_extractor_matches_the_shared_generation_partition_contract(
    tmp_path: Path, source: str
) -> None:
    bundle = _ValidatedFakeBundle(root=tmp_path / "bundle")
    extractor = GatedWindowExtractor(bundle, _config(tmp_path))

    detected = extractor.extract(source)
    generated = extractor.partitioner.partition(extractor.unit_extractor.extract(source))

    projection = lambda windows: [
        (
            window.byte_span,
            tuple(unit.text for unit in window.units),
            window.parent_descriptor,
            window.suitable,
            window.uncertain,
            window.overflow,
        )
        for window in windows
    ]
    assert projection(detected.windows) == projection(
        extractor.project_windows(generated.windows)
    )


def test_extractor_is_final_code_only_and_deterministic(tmp_path: Path) -> None:
    bundle = _ValidatedFakeBundle(root=tmp_path / "bundle")
    extractor = GatedWindowExtractor(bundle, _config(tmp_path))
    code = "def f():\n    x = 1\n    return x\n"

    assert extractor.extract(code) == extractor.extract(code)
    with pytest.raises(TypeError):
        extractor.extract(code, prompt="different")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        GatedWindowExtractor(bundle, _config(tmp_path), sidecar={})  # type: ignore[call-arg]


def test_unvalidated_or_contract_and_hash_mismatched_bundle_is_rejected(
    tmp_path: Path,
) -> None:
    bundle = _ValidatedFakeBundle(root=tmp_path / "bundle")
    bundle.validation_summary["validated"] = False
    with pytest.raises(ValueError, match="validated"):
        GatedWindowExtractor(bundle, _config(tmp_path))

    with pytest.raises(ValueError, match="bundle hash"):
        GatedWindowExtractor(
            _ValidatedFakeBundle(root=tmp_path / "bundle", bundle_sha256="e" * 64),
            _config(tmp_path),
        )

    bundle = _ValidatedFakeBundle(root=tmp_path / "bundle")
    bundle.manifest.window_contract_version = "other"
    with pytest.raises(ValueError, match="window contract"):
        GatedWindowExtractor(bundle, _config(tmp_path))

    bundle = _ValidatedFakeBundle(root=tmp_path / "bundle")
    bundle.manifest.tokenizer_sha256 = "f" * 64
    with pytest.raises(ValueError, match="tokenizer hash"):
        GatedWindowExtractor(bundle, _config(tmp_path))


def test_explicit_experimental_mode_accepts_marked_unvalidated_bundle(
    tmp_path: Path,
) -> None:
    bundle = _ValidatedFakeBundle(root=tmp_path / "bundle")
    bundle.validation_summary = {
        "validated": False,
        "experimental_only": True,
        "diagnostic_only": True,
        "not_official_method": True,
    }
    bundle.experimental_only = True

    extractor = GatedWindowExtractor(
        bundle,
        _config(tmp_path),
        allow_experimental=True,
    )

    assert extractor.extract("x = 1\n").windows


def test_explicit_unvalidated_mode_accepts_non_diagnostic_candidate(
    tmp_path: Path,
) -> None:
    bundle = _ValidatedFakeBundle(root=tmp_path / "bundle")
    bundle.validation_summary = {
        "validated": False,
        "validation_skipped_by_protocol": True,
        "unvalidated_candidate": True,
        "diagnostic_only": False,
        "not_official_method": False,
    }
    bundle.unvalidated_candidate = True

    extractor = GatedWindowExtractor(
        bundle,
        _config(tmp_path),
        allow_unvalidated=True,
    )

    assert extractor.extract("x = 1\n").windows


def test_token_overflow_is_closed_and_skipped(tmp_path: Path) -> None:
    result = _extract(tmp_path, "value = '" + ("λ" * 300) + "'\n")

    assert len(result.windows) == 1
    assert result.windows[0].overflow is True
    assert result.windows[0].suitable is False
    assert result.windows[0].skip_reason == "input_overflow"


def test_byte_spans_slice_original_utf8_source(tmp_path: Path) -> None:
    source = "def f():\n    名称 = 'λ'\n    return 名称\n"
    result = _extract(tmp_path, source)
    raw = source.encode("utf-8")

    for window in result.windows:
        assert raw[window.byte_span[0] : window.byte_span[1]].decode("utf-8") == (
            raw[window.start_byte : window.end_byte].decode("utf-8")
        )
