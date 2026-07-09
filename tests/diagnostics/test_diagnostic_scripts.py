from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[2] / "scripts" / "diagnostics"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_evidence_selector_upper_bound_script_exists() -> None:
    path = SCRIPT_ROOT / "analyze_evidence_selector_upper_bound.py"

    assert path.exists()


def test_evidence_selector_upper_bound_write_json_marks_list_payload(tmp_path) -> None:
    module = _load_module(
        SCRIPT_ROOT / "analyze_evidence_selector_upper_bound.py",
        "analyze_evidence_selector_upper_bound",
    )
    output_path = tmp_path / "policy_rows.json"

    module._write_json(output_path, [{"id": "HumanEval/0", "score": 1.0}])

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "policy_rows"
    assert payload["diagnostic_only"] is True
    assert payload["not_official_method"] is True
    assert payload["rows"] == [{"id": "HumanEval/0", "score": 1.0}]


def test_sparse_statistics_write_json_wraps_list_payload(tmp_path) -> None:
    module = _load_module(
        SCRIPT_ROOT / "analyze_sparse_statistics.py",
        "analyze_sparse_statistics",
    )
    output_path = tmp_path / "summary.json"

    module.write_json(output_path, [{"id": "HumanEval/0", "score": 1.0}])

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["artifact_type"] == "summary"
    assert payload["diagnostic_only"] is True
    assert payload["not_official_method"] is True
    assert payload["rows"] == [{"id": "HumanEval/0", "score": 1.0}]
