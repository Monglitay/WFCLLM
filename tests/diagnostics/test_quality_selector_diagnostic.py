from __future__ import annotations

import json

from wfcllm.diagnostics.quality_selector import mark_quality_selector_diagnostic
from wfcllm.diagnostics.quality_selector import select_candidate_rows
from wfcllm.diagnostics.static_selector import select_static_candidates


def test_mark_quality_selector_diagnostic_sets_required_fields() -> None:
    payload = mark_quality_selector_diagnostic({"name": "quality_selector"})

    assert payload["diagnostic_only"] is True
    assert payload["not_official_method"] is True
    assert payload["uses_quality_proxy"] is True


def test_select_candidate_rows_marks_quality_proxy_payload(tmp_path) -> None:
    prompt = 'def target(x):\n    """Return one more."""\n'
    candidate_path = tmp_path / "candidate.jsonl"
    detail_path = tmp_path / "details.jsonl"
    candidate_path.write_text(
        json.dumps(
            {
                "id": "HumanEval/0",
                "dataset": "humaneval",
                "prompt": prompt,
                "final_code": prompt + "    return x + 1\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    detail_path.write_text(
        json.dumps(
            {
                "id": "HumanEval/0",
                "score": 0.7,
                "scoreable_contexts": 2,
                "proxy_windows": 3,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows, analysis = select_candidate_rows([candidate_path], [detail_path])

    assert set(rows[0]) == {"id", "dataset", "prompt", "final_code"}
    assert analysis["diagnostic_only"] is True
    assert analysis["not_official_method"] is True
    assert analysis["uses_quality_proxy"] is True


def test_static_selector_marks_quality_proxy_payload() -> None:
    prompt = 'def target(x):\n    """Return a value."""\n'
    rows, analysis = select_static_candidates(
        baseline_rows=[
            {
                "id": "HumanEval/0",
                "prompt": prompt,
                "final_code": prompt + "    return 0\n",
            }
        ],
        candidate_rows=[
            {
                "id": "HumanEval/0",
                "prompt": prompt,
                "final_code": prompt + "    return 1\n",
            }
        ],
        baseline_details=[
            {
                "id": "HumanEval/0",
                "score": 0.1,
                "insufficient_evidence": False,
                "proxy_windows": 1,
            }
        ],
        candidate_details=[
            {
                "id": "HumanEval/0",
                "score": 0.4,
                "insufficient_evidence": False,
                "proxy_windows": 2,
            }
        ],
        policy="candidate_if_syntax_signature_sufficient",
    )

    assert rows[0]["final_code"] == prompt + "    return 1\n"
    assert analysis["diagnostic_only"] is True
    assert analysis["not_official_method"] is True
    assert analysis["uses_quality_proxy"] is True
