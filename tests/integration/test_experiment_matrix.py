from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest

from wfcllm.cli.config_resolver import resolve_method_config
from wfcllm.orchestration.state import PHASES


ROOT = Path(__file__).resolve().parents[2]
MATRIX = (
    ("python", "humaneval", "python_ast_equivalent"),
    ("python", "mbpp", "python_ast_equivalent"),
    ("cpp", "humanevalpack", "model_semantic_window"),
    ("java", "humanevalpack", "model_semantic_window"),
    ("js", "humanevalpack", "model_semantic_window"),
)


@pytest.mark.parametrize(("language", "dataset", "rewrite_strategy"), MATRIX)
def test_full_reproduction_public_surface(
    language: str,
    dataset: str,
    rewrite_strategy: str,
) -> None:
    stem = f"{language}_{dataset}_full"
    wrapper_path = ROOT / "scripts" / "experiments" / f"run_{stem}.sh"
    config_path = ROOT / "configs" / "wfcllm" / "experiments" / f"{stem}.json"

    wrapper = wrapper_path.read_text(encoding="utf-8")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    resolved = resolve_method_config(config)

    assert f'WFCLLM_LANGUAGE="{language}"' in wrapper
    assert f'WFCLLM_DATASET="{dataset}"' in wrapper
    assert 'WFCLLM_PROFILE="full"' in wrapper
    assert "run_gated_experiment.sh" in wrapper
    assert config["generation"]["language"] == language
    assert config["generation"]["dataset"] == dataset
    assert config["experiment"]["profile"] == "full"
    assert config["method"]["rewrite"]["strategy"] == rewrite_strategy
    assert config["semantic_lsh"]["rule_name"] == "semantic_lsh"
    assert resolved["method"]["name"] == "gated_semantic_window_v1"
    assert resolved["method"]["windowing"]["contract_version"] == (
        f"{language}-statement-window/v1"
    )
    assert resolved["method"]["rewrite"]["strategy"] == rewrite_strategy
    assert resolved["runtime"]["default_phases"] == PHASES


def test_exactly_five_full_configs_and_wrappers_are_public() -> None:
    config_dir = ROOT / "configs" / "wfcllm" / "experiments"
    wrapper_dir = ROOT / "scripts" / "experiments"
    expected_stems = sorted(
        f"{language}_{dataset}_full" for language, dataset, _strategy in MATRIX
    )

    assert sorted(path.stem for path in config_dir.glob("*.json")) == expected_stems
    assert sorted(ROOT.glob("configs/**/*.json")) == sorted(
        [
            ROOT / "configs" / "base_config.json",
            *(config_dir / f"{stem}.json" for stem in expected_stems),
        ]
    )
    assert sorted(
        path.stem.removeprefix("run_")
        for path in wrapper_dir.glob("run_*_*.sh")
        if path.name != "run_gated_experiment.sh"
    ) == expected_stems


def test_shared_runner_is_full_fresh_and_fail_closed() -> None:
    runner = (
        ROOT / "scripts" / "experiments" / "run_gated_experiment.sh"
    ).read_text(encoding="utf-8")

    assert runner.index("experiment_preflight") < runner.index("mkdir -p")
    assert "run_basic_multilanguage_experiment.py" not in runner
    assert "WFCLLM_PROFILE" not in runner or '"full"' in runner
    assert 'PILOT_SOURCE_CATALOG:?set PILOT_SOURCE_CATALOG' in runner
    assert 'FULL_SOURCE_CATALOG:?set FULL_SOURCE_CATALOG' in runner
    assert "SOURCE_CATALOG" not in runner.replace(
        "PILOT_SOURCE_CATALOG", ""
    ).replace("FULL_SOURCE_CATALOG", "")
    assert "head -n" not in runner
    assert "SAMPLE_LIMIT" not in runner
    assert "skipped audit" not in runner
    assert "/root/" not in runner
    assert "wfcllm_metric_contract.py" in runner
    assert runner.index("run_phase audit") < runner.index(
        "wfcllm_metric_contract.py"
    )
    for phase in PHASES:
        assert f"run_phase {phase}" in runner or (
            phase == "encoder" and '--phase encoder "${{PILOT_COMMON[@]}}' in runner
        )


@pytest.mark.parametrize(("language", "dataset", "_strategy"), MATRIX)
def test_full_wrappers_fail_before_artifacts_when_heavy_resources_are_absent(
    tmp_path: Path,
    language: str,
    dataset: str,
    _strategy: str,
) -> None:
    experiment_root = tmp_path / f"{language}-{dataset}-full"
    missing = tmp_path / "missing"
    environment = {
        **os.environ,
        "EXPERIMENT_ROOT": str(experiment_root),
        "GENERATION_MODEL_PATH": str(missing / "generation"),
        "REWRITE_MODEL_PATH": str(missing / "rewrite"),
        "SEMANTIC_ENCODER_MODEL_PATH": str(missing / "encoder"),
        "GATE_BASE_MODEL_PATH": str(missing / "gate"),
        "DATASET_PATH": str(missing / "datasets"),
        "PILOT_SOURCE_CATALOG": str(missing / "pilot.jsonl"),
        "FULL_SOURCE_CATALOG": str(missing / "full.jsonl"),
        "NEGATIVE_INPUT": str(missing / "negative.jsonl"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }

    completed = subprocess.run(
        [
            "bash",
            str(
                ROOT
                / "scripts"
                / "experiments"
                / f"run_{language}_{dataset}_full.sh"
            ),
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "generation model directory is missing" in completed.stderr
    assert not experiment_root.exists()


def test_default_config_is_canonical_python_humaneval_full_profile() -> None:
    config = json.loads(
        (
            ROOT
            / "configs"
            / "wfcllm"
            / "experiments"
            / "python_humaneval_full.json"
        ).read_text(encoding="utf-8")
    )
    resolved = resolve_method_config(config)

    assert resolved["method"]["name"] == "gated_semantic_window_v1"
    assert resolved["generation"]["language"] == "python"
    assert resolved["generation"]["dataset"] == "humaneval"
    assert resolved["experiment"]["profile"] == "full"
    assert resolved["runtime"]["default_phases"] == PHASES
