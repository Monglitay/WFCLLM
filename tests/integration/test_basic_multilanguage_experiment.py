from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "experiments" / "run_basic_multilanguage_experiment.py"


def _write_local_humanevalpack(root: Path, language: str) -> None:
    data_dir = root / "humanevalpack" / language
    data_dir.mkdir(parents=True)
    prefix = "CPP" if language == "cpp" else "Java"
    rows = [
        {
            "task_id": f"{prefix}/{index}",
            "prompt": f"{language} prompt {index}\n",
            "canonical_solution": f"{language} solution {index}\n",
            "entry_point": f"f{index}",
        }
        for index in range(3)
    ]
    data_dir.joinpath("test.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _run_basic(tmp_path: Path, language: str) -> Path:
    dataset_root = tmp_path / "datasets"
    _write_local_humanevalpack(dataset_root, language)
    run_dir = tmp_path / f"{language}-run"
    config = ROOT / "configs" / "wfcllm" / "experiments" / f"{language}_humanevalpack_fast.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(config),
            "--language",
            language,
            "--dataset",
            "humanevalpack",
            "--profile",
            "fast",
            "--dataset-path",
            str(dataset_root),
            "--run-dir",
            str(run_dir),
            "--sample-limit",
            "2",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return run_dir


def test_basic_cpp_experiment_cli_writes_complete_run(tmp_path: Path) -> None:
    run_dir = _run_basic(tmp_path, "cpp")

    final_rows = [
        json.loads(line)
        for line in (run_dir / "inputs" / "final_code.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["id"] for row in final_rows] == ["CPP/0", "CPP/1"]
    assert set(final_rows[0]) == {"id", "dataset", "prompt", "final_code"}
    report = json.loads((run_dir / "reports" / "reference_report.json").read_text(encoding="utf-8"))
    assert report["language"] == "cpp"
    assert report["dataset"] == "humanevalpack"
    assert report["sample_count"] == 2
    assert (run_dir / "calibration" / "reference_calibration.json").exists()
    assert (run_dir / "detection" / "positive_details.jsonl").exists()
    detector_audit = json.loads(
        (run_dir / "audit" / "detector_input_integrity.json").read_text(
            encoding="utf-8"
        )
    )
    assert detector_audit["ok"] is True


def test_basic_java_experiment_cli_writes_complete_run(tmp_path: Path) -> None:
    run_dir = _run_basic(tmp_path, "java")

    final_rows = [
        json.loads(line)
        for line in (run_dir / "inputs" / "final_code.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["id"] for row in final_rows] == ["Java/0", "Java/1"]
    report = json.loads((run_dir / "reports" / "reference_report.json").read_text(encoding="utf-8"))
    assert report["language"] == "java"
    assert report["profile"] == "fast"


def test_cpp_fast_wrapper_falls_back_to_basic_experiment_without_model_env(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "datasets"
    _write_local_humanevalpack(dataset_root, "cpp")
    run_root = tmp_path / "experiment"
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "GENERATION_MODEL_PATH",
            "SEMANTIC_ENCODER_MODEL_PATH",
            "GATE_BASE_MODEL_PATH",
            "NEGATIVE_INPUT",
            "SOURCE_CATALOG",
            "PILOT_SOURCE_CATALOG",
            "FULL_SOURCE_CATALOG",
        }
    }
    env.update(
        {
            "CONDA_EXE": os.environ.get("CONDA_EXE", "conda"),
            "CONDA_ENV_PREFIX": sys.prefix,
            "DATASET_PATH": str(dataset_root),
            "EXPERIMENT_ROOT": str(run_root),
            "SAMPLE_LIMIT": "2",
        }
    )

    result = subprocess.run(
        [str(ROOT / "scripts" / "experiments" / "run_cpp_humanevalpack_fast.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(
        (run_root / "run" / "reports" / "reference_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["language"] == "cpp"
    assert report["basic_experiment"] is True
