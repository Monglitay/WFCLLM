from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "analyze_sawr_evidence_goal.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_sawr_evidence_goal", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _report(auroc: float, positives: int = 4, negatives: int = 4) -> dict[str, object]:
    return {
        "primary": {
            "auroc": auroc,
            "positive_samples": positives,
            "negative_samples": negatives,
            "positive_insufficient_samples": 1,
            "negative_insufficient_samples": 0,
        }
    }


def test_build_model_negative_robustness_flags_length_only_drop(tmp_path: Path) -> None:
    family = tmp_path / "length_adjusted"
    _write_json(family / "reference_report.json", _report(0.84))
    _write_json(family / "model_negative_report.json", _report(0.61))
    _write_jsonl(
        family / "positive_details.jsonl",
        [
            {"id": "p0", "score": 0.9, "code_chars": 900},
            {"id": "p1", "score": 0.8, "code_chars": 700},
        ],
    )
    _write_jsonl(
        family / "reference_negative_details.jsonl",
        [
            {"id": "r0", "score": 0.1, "code_chars": 120},
            {"id": "r1", "score": 0.2, "code_chars": 180},
        ],
    )
    _write_jsonl(
        family / "model_negative_details.jsonl",
        [
            {"id": "m0", "score": 0.7, "code_chars": 850},
            {"id": "m1", "score": 0.6, "code_chars": 650},
        ],
    )

    module = _load_module()
    summary = module.build_model_negative_robustness(tmp_path)

    row = summary["rows"][0]
    assert row["family"] == "length_adjusted"
    assert row["reference_auroc"] == 0.84
    assert row["model_negative_auroc"] == 0.61
    assert row["model_negative_drop"] == 0.23
    assert row["diagnostic_only"] is True
    assert row["mean_code_chars"]["positive"] == 800
    assert row["mean_code_chars"]["reference_negative"] == 150
    assert row["mean_code_chars"]["model_negative"] == 750


def test_evidence_policy_report_selects_by_train_detection_then_pass(
    tmp_path: Path,
) -> None:
    module = _load_module()
    prompt = "def target(x):\n"
    candidate_rows = {
        "baseline": [
            {"id": "HumanEval/0", "prompt": prompt, "final_code": prompt + "    return 0\n"},
            {"id": "HumanEval/1", "prompt": prompt, "final_code": prompt + "    return 0\n"},
        ],
        "seed": [
            {"id": "HumanEval/0", "prompt": prompt, "final_code": prompt + "    return 1\n"},
            {"id": "HumanEval/1", "prompt": prompt, "final_code": prompt + "    return 1\n"},
        ],
    }
    details = {
        "baseline": [
            {"id": "HumanEval/0", "score": 0.4, "proxy_windows": 2, "scoreable_contexts": 1},
            {"id": "HumanEval/1", "score": 0.3, "proxy_windows": 2, "scoreable_contexts": 1},
        ],
        "seed": [
            {"id": "HumanEval/0", "score": 0.8, "proxy_windows": 3, "scoreable_contexts": 1},
            {"id": "HumanEval/1", "score": 0.2, "proxy_windows": 3, "scoreable_contexts": 1},
        ],
    }
    pass_rows = {
        "baseline": [
            {"id": "HumanEval/0", "passed": False},
            {"id": "HumanEval/1", "passed": True},
        ],
        "seed": [
            {"id": "HumanEval/0", "passed": True},
            {"id": "HumanEval/1", "passed": False},
        ],
    }
    candidates = []
    for name in ("baseline", "seed"):
        input_path = tmp_path / f"{name}.jsonl"
        details_path = tmp_path / f"{name}_details.jsonl"
        pass_path = tmp_path / f"{name}_pass.jsonl"
        _write_jsonl(input_path, candidate_rows[name])
        _write_jsonl(details_path, details[name])
        _write_jsonl(pass_path, pass_rows[name])
        candidates.append(module.CandidateSpec(name, input_path, details_path, pass_path))

    reference_path = tmp_path / "reference_details.jsonl"
    model_negative_path = tmp_path / "model_negative_details.jsonl"
    _write_jsonl(reference_path, [{"id": "r", "score": 0.1}])
    _write_jsonl(model_negative_path, [{"id": "m", "score": 0.15}])
    output_dir = tmp_path / "out"

    summary = module.build_evidence_policy_report(
        candidates,
        reference_details_path=reference_path,
        model_negative_details_path=model_negative_path,
        output_dir=output_dir,
        folds=2,
        final_policy=None,
    )

    assert summary["final"]["pass"] == 2
    assert summary["final"]["selected_counts"] == {"baseline": 1, "seed": 1}
    final_rows = [
        json.loads(line)
        for line in (output_dir / "final_selected_sanitized.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert set(final_rows[0]) == {"id", "dataset", "prompt", "final_code"}
    assert final_rows[0]["final_code"] == prompt + "    return 1\n"
