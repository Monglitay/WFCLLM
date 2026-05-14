"""Tests for the unified scripts/evaluate.py CLI."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import the script's main() via runpy-style attribute access so we don't shell out.
import importlib.util


def _load_evaluate_module():
    spec = importlib.util.spec_from_file_location(
        "_scripts_evaluate", REPO_ROOT / "scripts" / "evaluate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- exec subcommand ---

def test_evaluate_cli_exec_computes_pass_at_k(tmp_path, capsys):
    input_path = tmp_path / "candidates.jsonl"
    input_path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"task_id": "t1", "is_correct": True},
                {"task_id": "t1", "is_correct": False},
                {"task_id": "t2", "is_correct": True},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    evaluate = _load_evaluate_module()
    rc = evaluate.main(["exec", str(input_path), "--metric", "pass_at_1"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["metric"] == "pass_at_1"
    assert out["value"] == pytest.approx(0.75)
    assert out["sample_count"] == 3


def test_evaluate_cli_exec_with_reference_annotates_correctness(tmp_path, capsys):
    candidates_path = tmp_path / "cands.jsonl"
    reference_path = tmp_path / "refs.jsonl"
    candidates_path.write_text(
        json.dumps({"task_id": "t1", "generated_code": "def f(x):\n    return x + 1\n"}) + "\n",
        encoding="utf-8",
    )
    reference_path.write_text(
        json.dumps({"id": "t1", "generated_code": "def f(x):\n    return x + 1\n"}) + "\n",
        encoding="utf-8",
    )
    evaluate = _load_evaluate_module()
    rc = evaluate.main([
        "exec", str(candidates_path),
        "--metric", "pass_at_1",
        "--reference", str(reference_path),
    ])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["value"] == pytest.approx(1.0)


# --- detection subcommand ---

def test_evaluate_cli_detection_writes_report(tmp_path, capsys):
    summary_payload = {
        "dataset": "HumanEval",
        "watermark_params": {"lsh_d": 3},
        "summary": {"watermark_rate": 0.5},
    }
    detail_row = {
        "id": "HumanEval/0",
        "is_watermarked": True,
        "z_score": 4.0,
        "p_value": 0.0001,
        "independent_blocks": 5,
        "hits": 4,
    }
    left_summary = tmp_path / "ls.json"; left_summary.write_text(json.dumps(summary_payload))
    right_summary = tmp_path / "rs.json"; right_summary.write_text(json.dumps(summary_payload))
    left_details = tmp_path / "ld.jsonl"; left_details.write_text(json.dumps(detail_row) + "\n")
    right_details = tmp_path / "rd.jsonl"; right_details.write_text(json.dumps(detail_row) + "\n")
    output = tmp_path / "report.json"

    evaluate = _load_evaluate_module()
    rc = evaluate.main([
        "detection",
        "--left-summary", str(left_summary),
        "--left-details", str(left_details),
        "--right-summary", str(right_summary),
        "--right-details", str(right_details),
        "--output", str(output),
    ])
    assert rc == 0
    assert output.exists()
    report = json.loads(output.read_text())
    assert "compatibility" in report
    assert "detail_delta" in report
    assert report["compatibility"]["same_id_set"] is True


# --- dual subcommand routing ---

def test_evaluate_cli_dual_dispatches_to_dual_channel(tmp_path, monkeypatch):
    """Verify the dual subcommand calls wfcllm.evaluation.dual_channel.run_evaluation."""
    config_path = tmp_path / "cfg.json"
    config_path.write_text(json.dumps({"watermark": {}, "extract": {}}))

    captured: dict = {}

    def fake_run_evaluation(*, dataset, config_path, output_dir, candidate_count, **kw):
        captured["dataset"] = dataset
        captured["config_path"] = config_path
        captured["output_dir"] = output_dir
        captured["candidate_count"] = candidate_count
        return {"dataset": dataset, "modes": {}}

    import wfcllm.evaluation.dual_channel as dc
    monkeypatch.setattr(dc, "run_evaluation", fake_run_evaluation)

    evaluate = _load_evaluate_module()
    rc = evaluate.main([
        "dual",
        "--dataset", "humaneval",
        "--config", str(config_path),
        "--output-dir", str(tmp_path / "out"),
        "--num-candidates", "1",
    ])
    assert rc == 0
    assert captured["dataset"] == "humaneval"
    assert captured["candidate_count"] == 1


def test_evaluate_cli_unknown_subcommand_exits_nonzero(capsys):
    evaluate = _load_evaluate_module()
    with pytest.raises(SystemExit):
        evaluate.main(["frobnicate"])
