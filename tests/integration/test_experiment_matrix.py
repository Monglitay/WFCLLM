from __future__ import annotations

import json
from pathlib import Path

import pytest

from wfcllm.cli.config_resolver import resolve_method_config


ROOT = Path(__file__).resolve().parents[2]
MATRIX = (
    ("python", "humaneval", "full"),
    ("python", "humaneval", "fast"),
    ("python", "mbpp", "full"),
    ("python", "mbpp", "fast"),
    ("cpp", "humanevalpack", "full"),
    ("cpp", "humanevalpack", "fast"),
    ("java", "humanevalpack", "full"),
    ("java", "humanevalpack", "fast"),
    ("js", "humanevalpack", "full"),
    ("js", "humanevalpack", "fast"),
)


@pytest.mark.parametrize(("language", "dataset", "profile"), MATRIX)
def test_experiment_entry_and_config_contract(
    language: str,
    dataset: str,
    profile: str,
) -> None:
    stem = f"{language}_{dataset}_{profile}"
    wrapper_path = ROOT / "scripts" / "experiments" / f"run_{stem}.sh"
    config_path = ROOT / "configs" / "wfcllm" / "experiments" / f"{stem}.json"

    wrapper = wrapper_path.read_text(encoding="utf-8")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert f'WFCLLM_LANGUAGE="{language}"' in wrapper
    assert f'WFCLLM_DATASET="{dataset}"' in wrapper
    assert f'WFCLLM_PROFILE="{profile}"' in wrapper
    assert "run_gated_experiment.sh" in wrapper
    assert config["generation"]["language"] == language
    assert config["generation"]["dataset"] == dataset
    assert config["experiment"]["profile"] == profile
    assert config["semantic_lsh"]["rule_name"] == "semantic_lsh"
    assert config["method"]["rewrite"]["strategy"] != "keyed_text_region"

    resolved = resolve_method_config(config)
    assert resolved["generation"]["language"] == language
    assert resolved["generation"]["dataset"] == dataset
    assert resolved["semantic_lsh"]["rule_name"] == "semantic_lsh"
    assert resolved["method"]["windowing"]["contract_version"] == (
        f"{language}-statement-window/v1"
    )
    assert {
        "python": "oss_python",
        "cpp": "oss_cpp",
        "java": "oss_java",
        "js": "oss_js",
    }[language] in resolved["gate_data"]["sources"]
    assert resolved["gate_data"]["scale"] == (
        "full" if profile == "full" else "pilot"
    )


def test_matrix_contains_only_the_eight_supported_entries() -> None:
    wrappers = sorted(
        path.name
        for path in (ROOT / "scripts" / "experiments").glob("run_*_*.sh")
        if path.name != "run_gated_experiment.sh"
    )
    expected = sorted(
        f"run_{language}_{dataset}_{profile}.sh"
        for language, dataset, profile in MATRIX
    )

    assert wrappers == expected


def test_shared_runner_performs_preflight_before_preparing_keys() -> None:
    runner = (
        ROOT / "scripts" / "experiments" / "run_gated_experiment.sh"
    ).read_text(encoding="utf-8")

    assert runner.index("experiment_preflight") < runner.index(
        "wfcllm_prepare_gated_experiment.py"
    )
    assert "PILOT_SOURCE_CATALOG" in runner
    assert "FULL_SOURCE_CATALOG" in runner
    assert "keyed_text_region" in runner
    assert '--config "${WFCLLM_CONFIG}"' in runner
    assert "--gate-data-scale pilot" in runner
    assert "--gate-data-scale full" in runner


def test_shared_runner_trains_encoder_before_each_gate_data_chain() -> None:
    runner = (
        ROOT / "scripts" / "experiments" / "run_gated_experiment.sh"
    ).read_text(encoding="utf-8")

    assert 'run.py --phase encoder "${PILOT_COMMON[@]}"' in runner
    assert runner.index('run.py --phase encoder "${PILOT_COMMON[@]}"') < runner.index(
        "--gate-data-scale pilot"
    )
    assert "run_phase encoder" in runner
    assert runner.index("run_phase encoder") < runner.index("run_phase gate-data")
