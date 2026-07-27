from __future__ import annotations

import json
from pathlib import Path

import pytest

from wfcllm.cli.config_resolver import load_config, resolve_method_config
from wfcllm.cli.experiment_preflight import (
    validate_experiment_config,
    validate_runtime_capabilities,
    validate_runtime_resources,
)

ROOT = Path(__file__).resolve().parents[2]
MATRIX = (
    ("python", "humaneval", "python_ast_equivalent"),
    ("python", "mbpp", "python_ast_equivalent"),
    ("cpp", "humanevalpack", "model_semantic_window"),
    ("java", "humanevalpack", "model_semantic_window"),
    ("js", "humanevalpack", "model_semantic_window"),
)


def _write_model(path: Path) -> Path:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps({"model_type": "t5"}), encoding="utf-8"
    )
    (path / "model.safetensors").write_bytes(b"weights")
    (path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "T5Tokenizer"}), encoding="utf-8"
    )
    return path


def _catalog_rows(suffix: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(3):
        rows.extend(
            [
                {
                    "source_family": "main_generation",
                    "source_id": f"{suffix}-main-{index}",
                    "code": f"def main_{index}():\n    return {index}\n",
                    "repository_id": None,
                    "task_id": f"{suffix}-task-{index}",
                    "function_id": None,
                    "source_model_id": f"model-{index}",
                    "license_id": None,
                    "contract_or_hard_set": False,
                    "prompt": "",
                },
                {
                    "source_family": "oss_python",
                    "source_id": f"{suffix}-oss-{index}",
                    "code": f"def oss_{index}():\n    return {index}\n",
                    "repository_id": f"{suffix}-repo-{index}",
                    "task_id": None,
                    "function_id": f"oss-{index}",
                    "source_model_id": None,
                    "license_id": "MIT",
                    "contract_or_hard_set": False,
                    "prompt": "",
                },
            ]
        )
    rows.append(
        {
            "source_family": "parser_boundary",
            "source_id": f"{suffix}-parser",
            "code": "if True:\n    x = 1\n",
            "repository_id": None,
            "task_id": f"{suffix}-parser-task",
            "function_id": None,
            "source_model_id": None,
            "license_id": None,
            "contract_or_hard_set": True,
            "prompt": "",
        }
    )
    return rows


def _write_catalog(path: Path, suffix: str) -> Path:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in _catalog_rows(suffix)),
        encoding="utf-8",
    )
    return path


class _FakeHumanEvalAdapter:
    supported_languages = ("python",)

    def __init__(self, dataset_path: str = "data/datasets") -> None:
        self.dataset_path = dataset_path

    def supports(self, language: str) -> bool:
        return language == "python"

    def iter_samples(self, language: str | None = None):
        from wfcllm.datasets.adapter import CodeSample

        for index in range(164):
            yield CodeSample(
                task_id=f"HumanEval/{index}",
                prompt=f"def task_{index}():\n",
                language="python",
            )


@pytest.mark.parametrize(("language", "dataset", "strategy"), MATRIX)
def test_preflight_accepts_exact_committed_full_profiles(
    language: str, dataset: str, strategy: str
) -> None:
    path = (
        ROOT
        / "configs"
        / "wfcllm"
        / "experiments"
        / f"{language}_{dataset}_full.json"
    )
    config = resolve_method_config(load_config(path))
    assert config["method"]["rewrite"]["strategy"] == strategy
    validate_experiment_config(config, language, dataset, "full")
    validate_runtime_capabilities(config)


def test_preflight_rejects_removed_compatibility_keys() -> None:
    config = resolve_method_config(
        load_config(
            ROOT
            / "configs"
            / "wfcllm"
            / "experiments"
            / "python_humaneval_full.json"
        )
    )
    config["method"]["gate"]["bundle_path"] = "old/bundle"
    with pytest.raises(ValueError, match="method.gate"):
        validate_experiment_config(config, "python", "humaneval", "full")


def test_full_profile_rejects_relaxed_parameter_override() -> None:
    path = (
        ROOT
        / "configs"
        / "wfcllm"
        / "experiments"
        / "python_humaneval_full.json"
    )
    raw = load_config(path)
    raw["generation"]["max_new_tokens"] = 8

    with pytest.raises(ValueError, match="canonical overlay"):
        resolve_method_config(raw)


def test_runtime_resources_fail_closed_without_local_inputs(tmp_path) -> None:
    missing = tmp_path / "missing"
    config = resolve_method_config(
        load_config(
            ROOT
            / "configs"
            / "wfcllm"
            / "experiments"
            / "python_humaneval_full.json"
        )
    )
    with pytest.raises(ValueError, match="generation model"):
        validate_runtime_resources(
            config,
            generation_model_path=missing,
            rewrite_model_path=missing,
            semantic_encoder_model_path=missing,
            gate_base_model_path=missing,
            dataset_path=missing,
            pilot_source_catalog=missing,
            full_source_catalog=missing,
            negative_input=missing,
        )


def test_python_runtime_does_not_require_a_rewrite_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directories = {
        name: _write_model(tmp_path / name)
        for name in ("generation", "encoder", "gate")
    }
    directories["datasets"] = tmp_path / "datasets"
    directories["datasets"].mkdir()
    files = {
        "pilot.jsonl": _write_catalog(tmp_path / "pilot.jsonl", "pilot"),
        "full.jsonl": _write_catalog(tmp_path / "full.jsonl", "full"),
        "negative.jsonl": tmp_path / "negative.jsonl",
    }
    files["negative.jsonl"].write_text(
        json.dumps(
            {
                "id": "held-out/0",
                "dataset": "humaneval",
                "prompt": "def held_out():\n",
                "final_code": "def held_out():\n    return 0\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("wfcllm.datasets.get", lambda _name: _FakeHumanEvalAdapter())
    config = resolve_method_config(
        load_config(
            ROOT
            / "configs"
            / "wfcllm"
            / "experiments"
            / "python_humaneval_full.json"
        )
    )

    validate_runtime_resources(
        config,
        generation_model_path=directories["generation"],
        rewrite_model_path=None,
        semantic_encoder_model_path=directories["encoder"],
        gate_base_model_path=directories["gate"],
        dataset_path=directories["datasets"],
        pilot_source_catalog=files["pilot.jsonl"],
        full_source_catalog=files["full.jsonl"],
        negative_input=files["negative.jsonl"],
    )


def test_runtime_resources_reject_empty_model_directory(tmp_path: Path) -> None:
    empty_model = tmp_path / "empty-model"
    empty_model.mkdir()
    config = resolve_method_config(
        load_config(
            ROOT
            / "configs"
            / "wfcllm"
            / "experiments"
            / "python_humaneval_full.json"
        )
    )

    with pytest.raises(ValueError, match="generation model config"):
        validate_runtime_resources(
            config,
            generation_model_path=empty_model,
            rewrite_model_path=None,
            semantic_encoder_model_path=empty_model,
            gate_base_model_path=empty_model,
            dataset_path=tmp_path,
            pilot_source_catalog=tmp_path / "pilot.jsonl",
            full_source_catalog=tmp_path / "full.jsonl",
            negative_input=tmp_path / "negative.jsonl",
        )


def test_runtime_resources_reject_identical_catalog_bytes(
    tmp_path: Path,
) -> None:
    model = _write_model(tmp_path / "model")
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    pilot = _write_catalog(tmp_path / "pilot.jsonl", "same")
    full = tmp_path / "full.jsonl"
    full.write_bytes(pilot.read_bytes())
    negative = tmp_path / "negative.jsonl"
    negative.write_text(
        '{"id":"negative/0","dataset":"humaneval",'
        '"prompt":"negative prompt","final_code":"def f():\\n    return 0\\n"}\n',
        encoding="utf-8",
    )
    config = resolve_method_config(
        load_config(
            ROOT
            / "configs"
            / "wfcllm"
            / "experiments"
            / "python_humaneval_full.json"
        )
    )

    with pytest.raises(ValueError, match="identical input"):
        validate_runtime_resources(
            config,
            generation_model_path=model,
            rewrite_model_path=None,
            semantic_encoder_model_path=model,
            gate_base_model_path=model,
            dataset_path=datasets,
            pilot_source_catalog=pilot,
            full_source_catalog=full,
            negative_input=negative,
        )


def test_runtime_resources_reject_unloadable_target_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directories = {
        name: _write_model(tmp_path / name)
        for name in ("generation", "encoder", "gate")
    }
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    pilot = _write_catalog(tmp_path / "pilot.jsonl", "pilot")
    full = _write_catalog(tmp_path / "full.jsonl", "full")
    negative = tmp_path / "negative.jsonl"
    negative.write_text(
        '{"id":"negative/0","dataset":"humaneval",'
        '"prompt":"negative prompt","final_code":"def f():\\n    return 0\\n"}\n',
        encoding="utf-8",
    )

    class _MissingDataset(_FakeHumanEvalAdapter):
        def iter_samples(self, language: str | None = None):
            del language
            raise FileNotFoundError("offline cache missing")

    monkeypatch.setattr("wfcllm.datasets.get", lambda _name: _MissingDataset())
    config = resolve_method_config(
        load_config(
            ROOT
            / "configs"
            / "wfcllm"
            / "experiments"
            / "python_humaneval_full.json"
        )
    )

    with pytest.raises(ValueError, match="not loadable"):
        validate_runtime_resources(
            config,
            generation_model_path=directories["generation"],
            rewrite_model_path=None,
            semantic_encoder_model_path=directories["encoder"],
            gate_base_model_path=directories["gate"],
            dataset_path=dataset_root,
            pilot_source_catalog=pilot,
            full_source_catalog=full,
            negative_input=negative,
        )


def test_model_rewrite_runtime_requires_a_rewrite_model(tmp_path: Path) -> None:
    directories = {
        name: tmp_path / name
        for name in ("generation", "encoder", "gate", "datasets")
    }
    for path in directories.values():
        path.mkdir()
    pilot = tmp_path / "pilot.jsonl"
    full = tmp_path / "full.jsonl"
    negative = tmp_path / "negative.jsonl"
    for path in (pilot, full, negative):
        path.write_text("{}\n", encoding="utf-8")
    config = resolve_method_config(
        load_config(
            ROOT
            / "configs"
            / "wfcllm"
            / "experiments"
            / "js_humanevalpack_full.json"
        )
    )

    with pytest.raises(ValueError, match="rewrite model"):
        validate_runtime_resources(
            config,
            generation_model_path=directories["generation"],
            rewrite_model_path=None,
            semantic_encoder_model_path=directories["encoder"],
            gate_base_model_path=directories["gate"],
            dataset_path=directories["datasets"],
            pilot_source_catalog=pilot,
            full_source_catalog=full,
            negative_input=negative,
        )
