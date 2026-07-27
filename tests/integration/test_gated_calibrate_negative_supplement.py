"""Seam three: calibrate supplements the negative corpus autonomously (ADR 0008)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from wfcllm.cli.runners import (
    _generation_model_identifier,
    _safe_file_hash,
    _safe_tree_hash,
    _validate_negative_corpus_manifest,
    run_calibrate,
)
from wfcllm.datasets.adapter import CodeSample
from wfcllm.method.presets import GATED_SEMANTIC_WINDOW_V1_NAME, load_method_preset
from wfcllm.orchestration.state import RunStateManager


class _FakeDatasetAdapter:
    def __init__(self, size: int) -> None:
        self._size = size

    def supports(self, language: str) -> bool:
        return language == "python"

    def iter_samples(self, language: str | None = None):
        for index in range(self._size):
            yield CodeSample(
                task_id=f"ds/{index}",
                prompt=f"prompt ds/{index}\n",
                language="python",
            )


def test_model_identity_distinguishes_same_basename_trees(tmp_path: Path) -> None:
    first = tmp_path / "first" / "model"
    second = tmp_path / "second" / "model"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "weights.bin").write_bytes(b"first")
    (second / "weights.bin").write_bytes(b"second")

    first_id = _generation_model_identifier(first)
    second_id = _generation_model_identifier(second)

    assert first_id.startswith("model:sha256:")
    assert second_id.startswith("model:sha256:")
    assert first_id != second_id


def _make_fake_generator_class():
    class _FakeProgramGenerator:
        instances: list = []

        def __init__(self, **kwargs) -> None:
            type(self).instances.append(self)
            self.kwargs = kwargs

        def generate_program(self, *, prompt: str, sample_id: str) -> str:
            return prompt + "    return 'unwatermarked'\n"

    return _FakeProgramGenerator


def _forbidden_generator_class():
    class _ForbiddenProgramGenerator:
        def __init__(self, **kwargs) -> None:
            raise AssertionError(
                "generation model must not be loaded without supplementation"
            )

    return _ForbiddenProgramGenerator


def _external_row(index: int) -> dict[str, str]:
    return {
        "id": f"external/{index}",
        "dataset": "humaneval",
        "prompt": f"external prompt {index}\n",
        "final_code": f"def external_{index}():\n    return {index}\n",
    }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _supplement_setup(
    tmp_path: Path,
    monkeypatch,
    *,
    target: int,
    external_rows: list[dict[str, str]] | None,
    dataset_size: int = 8,
    positive_ids: tuple[str, ...] = ("ds/0", "ds/1"),
    supplement: dict[str, object] | None = None,
    generator_class=None,
):
    run_dir = tmp_path / "run"
    bundle = run_dir / "gate-train" / "candidate_bundle"
    (bundle / "tokenizer").mkdir(parents=True)
    (bundle / "gate_float.pt").write_bytes(b"formal")
    (bundle / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    config = load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME).to_dict()
    config["calibration"]["target_negative_count"] = target
    if supplement is not None:
        config["calibration"]["supplement"] = supplement

    from wfcllm.gate.production import experiment_contract_hash

    (run_dir / "gate-train" / "candidate_bundle_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "wfcllm-gate-train-candidate/v1",
                "config_hash": experiment_contract_hash(config),
                "candidate_bundle_sha256": _safe_tree_hash(bundle),
                "diagnostic_test_backend": False,
                "formal_eligible": True,
                "diagnostic_only": False,
                "not_official_method": False,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "inputs").mkdir(parents=True)
    _write_jsonl(
        run_dir / "inputs" / "final_code.jsonl",
        [
            {
                "id": positive_id,
                "dataset": "humaneval",
                "prompt": f"prompt {positive_id}\n",
                "final_code": "def f():\n    return 0\n",
            }
            for positive_id in positive_ids
        ],
    )
    final_code_path = run_dir / "inputs" / "final_code.jsonl"

    negative = tmp_path / "negative.jsonl"
    if external_rows is not None:
        _write_jsonl(negative, external_rows)
    deployment = tmp_path / "deployment.key"
    deployment.write_bytes(b"deployment-secret")
    generation_model = tmp_path / "generation-model"
    generation_model.mkdir()
    generation_model_identifier = _generation_model_identifier(generation_model)
    generation_manifest = run_dir / "generation" / "manifest.json"
    generation_manifest.parent.mkdir(parents=True)
    generation_manifest.write_text(
        json.dumps(
            {
                "final_code_sha256": _safe_file_hash(final_code_path),
                "final_code_row_count": len(positive_ids),
                "generation_model_identifier": generation_model_identifier,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rewrite_model = tmp_path / "rewrite-model"
    rewrite_model.mkdir()
    source_catalog = tmp_path / "catalog.jsonl"
    source_catalog.write_text("", encoding="utf-8")
    encoder_model = tmp_path / "encoder-model"
    encoder_model.mkdir()
    gate_base_model = tmp_path / "gate-base-model"
    gate_base_model.mkdir()
    calibrate_jsonl = MagicMock()

    def write_calibration(_input_path, *, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"schema_version": "test-calibration"}) + "\n",
            encoding="utf-8",
        )

    calibrate_jsonl.side_effect = write_calibration
    pipeline = SimpleNamespace(
        calibrate_jsonl=calibrate_jsonl,
        detect_jsonl=MagicMock(),
    )
    args = argparse.Namespace(
        _config_cache=config,
        run_dir=str(run_dir),
        run_id=None,
        negative_input=str(negative),
        secret_key_file=str(deployment),
        secret_key_env=None,
        _gated_detection_pipeline=pipeline,
        gate_source_catalog=str(source_catalog),
        generation_model_path=str(generation_model),
        rewrite_model_path=str(rewrite_model),
        semantic_encoder_model_path=str(encoder_model),
        gate_base_model_path=str(gate_base_model),
        model_device="cpu",
        state_file=str(tmp_path / "state.json"),
        dataset=None,
        language=None,
        dataset_path=None,
        sample_offset=None,
        sample_limit=None,
    )

    monkeypatch.setattr(
        "wfcllm.datasets.get", lambda name: _FakeDatasetAdapter(dataset_size)
    )
    fake_generator = generator_class or _make_fake_generator_class()
    monkeypatch.setattr(
        "wfcllm.gate.production.LocalHFProgramGenerator", fake_generator
    )

    state = RunStateManager(tmp_path / "state.json")
    encoder_checkpoint = run_dir / "encoder" / "best_model.pt"
    encoder_checkpoint.parent.mkdir(parents=True)
    encoder_checkpoint.write_bytes(b"semantic-projection")
    from wfcllm.gate.production import experiment_contract_hash

    state.mark_done(
        "encoder",
        best_model_path=str(encoder_checkpoint),
        best_model_sha256=_safe_file_hash(encoder_checkpoint),
        source_catalog_sha256=_safe_file_hash(source_catalog),
        config_sha256=experiment_contract_hash(config),
    )
    state.mark_done(
        "generate",
        gate_bundle_sha256=_safe_tree_hash(bundle),
        output_path=str(final_code_path),
        final_code_sha256=_safe_file_hash(final_code_path),
        final_code_row_count=len(positive_ids),
        generation_manifest_sha256=_safe_file_hash(generation_manifest),
        generation_model_identifier=generation_model_identifier,
    )
    return args, pipeline, config, state, run_dir, fake_generator


def test_supplement_triggered_when_external_below_target(tmp_path, monkeypatch) -> None:
    external_rows = [_external_row(0), _external_row(1)]
    args, pipeline, config, state, run_dir, fake_generator = _supplement_setup(
        tmp_path, monkeypatch, target=5, external_rows=external_rows
    )

    assert run_calibrate(args, state) == 0

    corpus_path = run_dir / "calibration" / "negative_corpus.jsonl"
    corpus_rows = _read_jsonl(corpus_path)
    assert len(corpus_rows) == 5
    assert [row["id"] for row in corpus_rows[:2]] == ["external/0", "external/1"]
    # Deterministic dataset-order sampling, skipping positive evaluation ids.
    assert [row["id"] for row in corpus_rows[2:]] == ["ds/2", "ds/3", "ds/4"]
    for row in corpus_rows:
        assert set(row) == {"id", "dataset", "prompt", "final_code"}

    # Non-overlap with the positive evaluation set (ids and prompts).
    positive_rows = _read_jsonl(run_dir / "inputs" / "final_code.jsonl")
    positive_ids = {row["id"] for row in positive_rows}
    positive_prompts = {row["prompt"] for row in positive_rows}
    for row in corpus_rows:
        assert row["id"] not in positive_ids
        assert row["prompt"] not in positive_prompts

    # The pipeline calibrates on the consolidated corpus file.
    (called_path,), called_kwargs = pipeline.calibrate_jsonl.call_args
    assert called_path == str(corpus_path)
    assert called_kwargs == {
        "output_path": run_dir / "calibration" / "reference_calibration.json"
    }

    # The detector binding covers the provenance manifest, which in turn
    # binds the consolidated corpus.
    corpus_hash = hashlib.sha256(corpus_path.read_bytes()).hexdigest()

    manifest = json.loads(
        (run_dir / "calibration" / "negative_corpus_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["external_count"] == 2
    assert manifest["generated_count"] == 3
    assert manifest["target_negative_count"] == 5
    assert manifest["generation_model_identifier"].startswith(
        "generation-model:sha256:"
    )
    assert manifest["corpus_sha256"] == corpus_hash
    assert args._gated_negative_manifest_hash == hashlib.sha256(
        (run_dir / "calibration" / "negative_corpus_manifest.json").read_bytes()
    ).hexdigest()
    decoding = manifest["decoding_config"]
    assert decoding["max_new_tokens"] == config["generation"]["max_new_tokens"]
    assert decoding["temperature"] == config["generation"]["temperature"]
    assert decoding["top_p"] == config["generation"]["top_p"]
    assert decoding["seed"] == config["generation"]["seed"]
    assert str(args.generation_model_path) not in json.dumps(manifest)

    assert state.get("calibrate", "gate_bundle_sha256") == _safe_tree_hash(
        run_dir / "gate-train" / "candidate_bundle"
    )
    assert len(fake_generator.instances) == 1


def test_supplement_not_triggered_when_external_meets_target(
    tmp_path, monkeypatch
) -> None:
    external_rows = [_external_row(0), _external_row(1), _external_row(2)]
    args, pipeline, config, state, run_dir, _generator = _supplement_setup(
        tmp_path,
        monkeypatch,
        target=3,
        external_rows=external_rows,
        generator_class=_forbidden_generator_class(),
    )

    assert run_calibrate(args, state) == 0

    corpus_path = run_dir / "calibration" / "negative_corpus.jsonl"
    corpus_rows = _read_jsonl(corpus_path)
    assert [row["id"] for row in corpus_rows] == [
        "external/0",
        "external/1",
        "external/2",
    ]
    manifest = json.loads(
        (run_dir / "calibration" / "negative_corpus_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["external_count"] == 3
    assert manifest["generated_count"] == 0
    assert manifest["generation_model_identifier"] is None
    assert manifest["decoding_config"] is None
    pipeline.calibrate_jsonl.assert_called_once()


@pytest.mark.parametrize("overlap_field", ["id", "prompt"])
@pytest.mark.parametrize("target", [1, 3])
def test_external_negative_corpus_must_be_disjoint_even_when_target_is_met(
    tmp_path,
    monkeypatch,
    overlap_field: str,
    target: int,
) -> None:
    overlap = _external_row(0)
    overlap[overlap_field] = (
        " DS/0 " if overlap_field == "id" else "prompt ds/0\r\n"
    )
    external_rows = [overlap, *[_external_row(index) for index in range(1, target)]]
    args, pipeline, _config, state, run_dir, _generator = _supplement_setup(
        tmp_path,
        monkeypatch,
        target=target,
        external_rows=external_rows,
        generator_class=_forbidden_generator_class(),
    )

    with pytest.raises(ValueError, match=f"positive population by {overlap_field}"):
        run_calibrate(args, state)

    pipeline.calibrate_jsonl.assert_not_called()
    assert not (run_dir / "calibration" / "negative_corpus.jsonl").exists()


def test_supplement_requires_external_negative_corpus(
    tmp_path, monkeypatch
) -> None:
    args, pipeline, config, state, run_dir, _generator = _supplement_setup(
        tmp_path, monkeypatch, target=2, external_rows=None
    )
    args.negative_input = None

    with pytest.raises(ValueError, match="--negative-input"):
        run_calibrate(args, state)
    pipeline.calibrate_jsonl.assert_not_called()
    assert not (run_dir / "calibration" / "negative_corpus.jsonl").exists()


def test_full_dataset_can_supplement_from_external_held_out_prompts(
    tmp_path, monkeypatch
) -> None:
    args, pipeline, config, state, _run_dir, _generator = _supplement_setup(
        tmp_path,
        monkeypatch,
        target=5,
        external_rows=[_external_row(0)],
        dataset_size=3,
        positive_ids=("ds/0", "ds/1", "ds/2"),
    )

    assert run_calibrate(args, state) == 0
    rows = _read_jsonl(
        Path(args.run_dir) / "calibration" / "negative_corpus.jsonl"
    )
    assert len(rows) == 5
    assert all(
        row["prompt"] == "external prompt 0\n" for row in rows[1:]
    )
    assert len({row["id"] for row in rows}) == 5
    pipeline.calibrate_jsonl.assert_called_once()


def test_supplement_rejects_external_rows_with_quality_signals(
    tmp_path, monkeypatch
) -> None:
    tainted = {**_external_row(0), "pass": True}
    args, pipeline, config, state, _run_dir, _generator = _supplement_setup(
        tmp_path, monkeypatch, target=3, external_rows=[tainted]
    )

    with pytest.raises(ValueError, match="four fields"):
        run_calibrate(args, state)
    pipeline.calibrate_jsonl.assert_not_called()


def test_calibrate_rejects_final_code_modified_after_generate(
    tmp_path, monkeypatch
) -> None:
    args, pipeline, config, state, run_dir, _generator = _supplement_setup(
        tmp_path,
        monkeypatch,
        target=3,
        external_rows=[_external_row(0)],
    )
    _write_jsonl(
        run_dir / "inputs" / "final_code.jsonl",
        [
            {
                "id": "ds/0",
                "dataset": "humaneval",
                "prompt": "prompt ds/0\n",
                "final_code": "def correctness_selected():\n    return 1\n",
            }
        ],
    )

    with pytest.raises(ValueError, match="changed after generate"):
        run_calibrate(args, state)
    pipeline.calibrate_jsonl.assert_not_called()


def test_negative_manifest_tampering_breaks_current_binding(
    tmp_path, monkeypatch
) -> None:
    args, _pipeline, _config, state, run_dir, _generator = _supplement_setup(
        tmp_path,
        monkeypatch,
        target=3,
        external_rows=[_external_row(0), _external_row(1), _external_row(2)],
    )
    assert run_calibrate(args, state) == 0
    manifest_path = run_dir / "calibration" / "negative_corpus_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["external_count"] = 999
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="row count"):
        _validate_negative_corpus_manifest(run_dir)


def test_supplement_decoding_overrides_come_from_calibration_config(
    tmp_path, monkeypatch
) -> None:
    args, pipeline, config, state, run_dir, fake_generator = _supplement_setup(
        tmp_path,
        monkeypatch,
        target=3,
        external_rows=[_external_row(0)],
        supplement={"temperature": 0.8, "max_new_tokens": 64},
    )

    assert run_calibrate(args, state) == 0

    kwargs = fake_generator.instances[0].kwargs
    assert kwargs["temperature"] == 0.8
    assert kwargs["max_new_tokens"] == 64
    assert kwargs["top_p"] == config["generation"]["top_p"]
    assert kwargs["seed"] == config["generation"]["seed"]
    # The remaining decoding surface matches the generate-side construction:
    # rewrite_* comes from method.rewrite via the shared runtime options,
    # quantization/dtype/prompt-mode come from the generation section.
    rewrite = config["method"]["rewrite"]
    assert kwargs["rewrite_max_new_tokens"] == rewrite["max_new_tokens"]
    assert kwargs["rewrite_generation_attempts"] == rewrite["generation_attempts"]
    assert kwargs["rewrite_temperature"] == rewrite["temperature"]
    assert kwargs["rewrite_top_p"] == rewrite["top_p"]
    assert kwargs["load_in_4bit"] == config["generation"]["load_in_4bit"]
    assert kwargs["torch_dtype"] == config["generation"]["torch_dtype"]
    assert kwargs["program_prompt_mode"] == config["generation"]["prompt_mode"]
    manifest = json.loads(
        (run_dir / "calibration" / "negative_corpus_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["decoding_config"]["temperature"] == 0.8
    assert manifest["decoding_config"]["max_new_tokens"] == 64
