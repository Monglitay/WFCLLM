from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "select_sawr_static_candidates.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("select_sawr_static_candidates", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_static_candidate_selection_uses_syntax_signature_and_evidence() -> None:
    prompt = 'def target(x):\n    """Return a value."""\n'
    baseline_rows = [
        {"id": "HumanEval/0", "prompt": prompt, "final_code": prompt + "    return 0\n"},
        {"id": "HumanEval/1", "prompt": prompt, "final_code": prompt + "    return 0\n"},
        {"id": "HumanEval/2", "prompt": prompt, "final_code": prompt + "    return 0\n"},
    ]
    candidate_rows = [
        {
            "id": "HumanEval/0",
            "prompt": prompt,
            "final_code": prompt + "    return 1\n",
            "retry_trace": ["forbidden"],
        },
        {"id": "HumanEval/1", "prompt": prompt, "final_code": "def target(x, y):\n    return x\n"},
        {"id": "HumanEval/2", "prompt": prompt, "final_code": prompt + "    return 2\n"},
    ]
    baseline_details = [
        {"id": row["id"], "score": 0.0, "insufficient_evidence": False, "proxy_windows": 3}
        for row in baseline_rows
    ]
    candidate_details = [
        {"id": "HumanEval/0", "score": 0.1, "insufficient_evidence": False, "proxy_windows": 3},
        {"id": "HumanEval/1", "score": 0.2, "insufficient_evidence": False, "proxy_windows": 3},
        {"id": "HumanEval/2", "score": 0.3, "insufficient_evidence": True, "proxy_windows": 1},
    ]

    module = _load_module()
    rows, analysis = module.select_static_candidates(
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
        baseline_details=baseline_details,
        candidate_details=candidate_details,
        policy="candidate_if_syntax_signature_sufficient",
    )

    assert [row["final_code"] for row in rows] == [
        prompt + "    return 1\n",
        prompt + "    return 0\n",
        prompt + "    return 0\n",
    ]
    assert set(rows[0]) == {"id", "dataset", "prompt", "final_code"}
    assert analysis["selected_counts"] == {"baseline": 2, "candidate": 1}
