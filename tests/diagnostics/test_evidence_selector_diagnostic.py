from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from wfcllm.diagnostics.evidence_selector import mark_evidence_selector_diagnostic
from wfcllm.diagnostics.evidence_selector import select_candidate_rows


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "diagnostics"
    / "select_evidence_only_candidates.py"
)


def test_mark_evidence_selector_diagnostic_sets_required_fields() -> None:
    payload = mark_evidence_selector_diagnostic(
        {"name": "evidence_only_selector_diagnostic"}
    )

    assert payload["diagnostic_only"] is True
    assert payload["not_official_method"] is True
    assert payload["name"] == "evidence_only_selector_diagnostic"


def test_select_candidate_rows_marks_evidence_only_without_quality_proxy(tmp_path) -> None:
    prompt = 'def target(x):\n    """Return a value."""\n'
    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    baseline_detail_path = tmp_path / "baseline_details.jsonl"
    candidate_detail_path = tmp_path / "candidate_details.jsonl"
    baseline_path.write_text(
        json.dumps(
            {
                "id": "HumanEval/0",
                "dataset": "humaneval",
                "prompt": prompt,
                "final_code": prompt + "    return 0\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    candidate_path.write_text(
        json.dumps(
            {
                "id": "HumanEval/0",
                "dataset": "humaneval",
                "prompt": prompt,
                "final_code": "def target(x):\n    return (\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    baseline_detail_path.write_text(
        json.dumps({"id": "HumanEval/0", "score": 0.1, "proxy_windows": 1})
        + "\n",
        encoding="utf-8",
    )
    candidate_detail_path.write_text(
        json.dumps({"id": "HumanEval/0", "score": 2.0, "proxy_windows": 4})
        + "\n",
        encoding="utf-8",
    )

    rows, analysis = select_candidate_rows(
        [baseline_path, candidate_path],
        [baseline_detail_path, candidate_detail_path],
    )

    assert rows[0]["final_code"] == "def target(x):\n    return (\n"
    assert set(rows[0]) == {"id", "dataset", "prompt", "final_code"}
    assert analysis["diagnostic_only"] is True
    assert analysis["not_official_method"] is True
    assert "uses_quality_proxy" not in analysis


def test_evidence_selector_cli_does_not_expose_quality_gate_options() -> None:
    spec = importlib.util.spec_from_file_location(
        "select_evidence_only_candidates",
        SCRIPT_PATH,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    help_text = module._build_parser().format_help()  # noqa: SLF001

    assert "--require-public-doctest-passed" not in help_text
    assert "--reject-suspicious-tail" not in help_text
    assert "quality_first" not in help_text
