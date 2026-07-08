from __future__ import annotations

import builtins
import importlib.util
import json
from pathlib import Path

from wfcllm.sawr.selection import (
    CandidateSelectionFeatures,
    evaluate_candidate_quality,
    select_best_candidate,
)


PROMPT = '''def add_one(x):
    """
    >>> add_one(1)
    2
    >>> add_one(-1)
    0
    """
'''


def test_evaluate_candidate_quality_runs_prompt_doctests_without_hidden_tests() -> None:
    features = evaluate_candidate_quality(
        {
            "id": "HumanEval/0",
            "prompt": PROMPT,
            "final_code": PROMPT + "    return x + 1\n",
        },
        detector_score=0.1,
        candidate_index=0,
    )

    assert features.syntax_valid is True
    assert features.target_function_present is True
    assert features.signature_compatible is True
    assert features.prompt_doctest_count == 2
    assert features.prompt_doctest_passed is True
    assert features.detector_score == 0.1


def test_evaluate_candidate_quality_flags_syntax_and_signature_failures() -> None:
    syntax_features = evaluate_candidate_quality(
        {
            "id": "HumanEval/0",
            "prompt": PROMPT,
            "final_code": "def add_one(x):\n    return (\n",
        },
        detector_score=9.0,
        candidate_index=0,
    )
    signature_features = evaluate_candidate_quality(
        {
            "id": "HumanEval/0",
            "prompt": PROMPT,
            "final_code": "def add_one(x, y):\n    return x + y\n",
        },
        detector_score=8.0,
        candidate_index=1,
    )

    assert syntax_features.syntax_valid is False
    assert syntax_features.prompt_doctest_passed is False
    assert signature_features.syntax_valid is True
    assert signature_features.target_function_present is True
    assert signature_features.signature_compatible is False


def test_evaluate_candidate_quality_times_out_slow_prompt_doctests() -> None:
    slow_prompt = '''def loops_forever():
    """
    >>> loops_forever()
    1
    """
'''

    features = evaluate_candidate_quality(
        {
            "id": "HumanEval/slow",
            "prompt": slow_prompt,
            "final_code": slow_prompt + "    while True:\n        pass\n",
        },
        doctest_timeout_seconds=0.05,
    )

    assert features.syntax_valid is True
    assert features.prompt_doctest_count == 1
    assert features.prompt_doctest_passed is False
    assert features.public_doctest_timeout is True


def test_evaluate_candidate_quality_runs_public_doctests_in_subprocess() -> None:
    leak_name = "_sawr_selection_leak"
    if hasattr(builtins, leak_name):
        delattr(builtins, leak_name)
    leaking_code = (
        PROMPT
        + f"    __import__('builtins').{leak_name} = True\n"
        + "    return x + 1\n"
    )

    features = evaluate_candidate_quality(
        {
            "id": "HumanEval/0",
            "prompt": PROMPT,
            "final_code": leaking_code,
        },
        doctest_timeout_seconds=0.5,
    )

    assert features.public_doctest_passed is True
    assert not hasattr(builtins, leak_name)


def test_evaluate_candidate_quality_records_code_and_evidence_features() -> None:
    features = evaluate_candidate_quality(
        {
            "id": "HumanEval/0",
            "prompt": PROMPT,
            "final_code": PROMPT + "    return x +\n",
        },
        detector_score=0.2,
        scoreable_contexts=3,
        proxy_windows=5,
        insufficient_evidence=False,
        baseline_detector_score=0.4,
        baseline_proxy_windows=7,
    )

    assert features.code_chars > 0
    assert features.suspicious_tail is True
    assert features.truncation_suspected is True
    assert features.scoreable_contexts == 3
    assert features.proxy_windows == 5
    assert features.insufficient_evidence is False
    assert features.score_delta_vs_baseline == -0.2
    assert features.proxy_delta_vs_baseline == -2


def test_evaluate_candidate_quality_handles_malformed_prompt_doctest() -> None:
    malformed_prompt = '''def bad_docstring(x):
    """
    >>> bad_docstring("abc"
    "abcdef")
ghijklm")
    """
'''

    features = evaluate_candidate_quality(
        {
            "id": "HumanEval/malformed",
            "prompt": malformed_prompt,
            "final_code": malformed_prompt + "    return x\n",
        },
        doctest_timeout_seconds=0.2,
    )

    assert features.public_doctest_parse_error is True
    assert features.public_doctest_passed is False
    assert features.public_doctest_count == 0


def test_select_best_candidate_prefers_prompt_doctest_pass_over_score() -> None:
    high_score_wrong = CandidateSelectionFeatures(
        sample_id="HumanEval/0",
        candidate_index=0,
        syntax_valid=True,
        target_function_present=True,
        signature_compatible=True,
        prompt_doctest_passed=False,
        prompt_doctest_count=2,
        detector_score=2.0,
    )
    lower_score_correct = CandidateSelectionFeatures(
        sample_id="HumanEval/0",
        candidate_index=1,
        syntax_valid=True,
        target_function_present=True,
        signature_compatible=True,
        prompt_doctest_passed=True,
        prompt_doctest_count=2,
        detector_score=0.2,
    )

    selected = select_best_candidate([high_score_wrong, lower_score_correct])

    assert selected.candidate_index == 1
    assert selected.selection_reason == "syntax_signature_doctest_score"


def test_select_best_candidate_uses_detector_score_after_quality_gates() -> None:
    lower = CandidateSelectionFeatures(
        sample_id="HumanEval/0",
        candidate_index=0,
        syntax_valid=True,
        target_function_present=True,
        signature_compatible=True,
        prompt_doctest_passed=True,
        prompt_doctest_count=0,
        detector_score=0.2,
    )
    higher = CandidateSelectionFeatures(
        sample_id="HumanEval/0",
        candidate_index=1,
        syntax_valid=True,
        target_function_present=True,
        signature_compatible=True,
        prompt_doctest_passed=True,
        prompt_doctest_count=0,
        detector_score=0.9,
    )

    selected = select_best_candidate([lower, higher])

    assert selected.candidate_index == 1


def test_select_best_candidate_public_then_detector_prefers_score_over_proxy() -> None:
    more_proxy_lower_score = CandidateSelectionFeatures(
        sample_id="HumanEval/0",
        candidate_index=0,
        syntax_valid=True,
        target_function_present=True,
        signature_compatible=True,
        prompt_doctest_passed=True,
        prompt_doctest_count=0,
        detector_score=0.2,
        proxy_windows=10,
        scoreable_contexts=5,
    )
    fewer_proxy_higher_score = CandidateSelectionFeatures(
        sample_id="HumanEval/0",
        candidate_index=1,
        syntax_valid=True,
        target_function_present=True,
        signature_compatible=True,
        prompt_doctest_passed=True,
        prompt_doctest_count=0,
        detector_score=0.9,
        proxy_windows=2,
        scoreable_contexts=1,
    )

    selected = select_best_candidate(
        [more_proxy_lower_score, fewer_proxy_higher_score],
        ranking_mode="public_then_detector",
    )

    assert selected.candidate_index == 1
    assert selected.selection_reason == "public_then_detector"


def test_select_best_candidate_prefers_signature_match_over_score() -> None:
    mismatch = CandidateSelectionFeatures(
        sample_id="HumanEval/0",
        candidate_index=0,
        syntax_valid=True,
        target_function_present=True,
        signature_compatible=False,
        prompt_doctest_passed=True,
        prompt_doctest_count=0,
        detector_score=10.0,
    )
    match = CandidateSelectionFeatures(
        sample_id="HumanEval/0",
        candidate_index=1,
        syntax_valid=True,
        target_function_present=True,
        signature_compatible=True,
        prompt_doctest_passed=True,
        prompt_doctest_count=0,
        detector_score=0.1,
    )

    selected = select_best_candidate([mismatch, match])

    assert selected.candidate_index == 1


def test_select_candidate_rows_sanitizes_generation_private_fields(tmp_path) -> None:
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "select_sawr_candidates.py"
    )
    spec = importlib.util.spec_from_file_location("select_sawr_candidates", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    candidate_path = tmp_path / "candidate.jsonl"
    detail_path = tmp_path / "details.jsonl"
    candidate_path.write_text(
        json.dumps(
            {
                "id": "HumanEval/0",
                "dataset": "humaneval",
                "prompt": PROMPT,
                "final_code": PROMPT + "    return x + 1\n",
                "blocks": [{"private": True}],
                "watermark_params": {"secret_key": "leak"},
                "retry_trace": ["private"],
                "sampling_logits": [1.0],
                "detector_score": 9.0,
                "z_score": 9.0,
                "p_value": 0.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    detail_path.write_text(
        json.dumps(
            {
                "id": "HumanEval/0",
                "score": 0.3,
                "scoreable_contexts": 2,
                "proxy_windows": 4,
                "insufficient_evidence": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    selected, analysis = module.select_candidate_rows([candidate_path], [detail_path])

    assert set(selected[0]) == {"id", "dataset", "prompt", "final_code"}
    selected_features = analysis["per_sample"][0]["candidates"][0]
    assert selected_features["scoreable_contexts"] == 2
    assert selected_features["proxy_windows"] == 4
    assert selected_features["insufficient_evidence"] is False


def test_select_candidate_rows_respects_detector_drop_gate(tmp_path) -> None:
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "select_sawr_candidates.py"
    )
    spec = importlib.util.spec_from_file_location("select_sawr_candidates", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    baseline_detail_path = tmp_path / "baseline_details.jsonl"
    candidate_detail_path = tmp_path / "candidate_details.jsonl"
    baseline_path.write_text(
        json.dumps(
            {
                "id": "HumanEval/0",
                "prompt": PROMPT,
                "final_code": PROMPT + "    return x + 1\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    candidate_path.write_text(
        json.dumps(
            {
                "id": "HumanEval/0",
                "prompt": PROMPT,
                "final_code": PROMPT + "    return x + 1\n# candidate\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    baseline_detail_path.write_text(
        json.dumps({"id": "HumanEval/0", "score": 0.5, "proxy_windows": 3})
        + "\n",
        encoding="utf-8",
    )
    candidate_detail_path.write_text(
        json.dumps({"id": "HumanEval/0", "score": 0.1, "proxy_windows": 4})
        + "\n",
        encoding="utf-8",
    )

    selected, analysis = module.select_candidate_rows(
        [baseline_path, candidate_path],
        [baseline_detail_path, candidate_detail_path],
        max_score_drop_vs_baseline=0.0,
    )

    assert selected[0]["final_code"] == PROMPT + "    return x + 1\n"
    assert analysis["selected_by_candidate_index"] == {"0": 1}


def test_select_candidate_rows_accepts_public_then_detector_ranking(tmp_path) -> None:
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "select_sawr_candidates.py"
    )
    spec = importlib.util.spec_from_file_location("select_sawr_candidates", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    baseline_detail_path = tmp_path / "baseline_details.jsonl"
    candidate_detail_path = tmp_path / "candidate_details.jsonl"
    baseline_path.write_text(
        json.dumps(
            {
                "id": "HumanEval/0",
                "prompt": PROMPT,
                "final_code": PROMPT + "    return x + 1\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    candidate_path.write_text(
        json.dumps(
            {
                "id": "HumanEval/0",
                "prompt": PROMPT,
                "final_code": PROMPT + "    return x + 1\n# candidate\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    baseline_detail_path.write_text(
        json.dumps(
            {
                "id": "HumanEval/0",
                "score": 0.1,
                "proxy_windows": 20,
                "scoreable_contexts": 4,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    candidate_detail_path.write_text(
        json.dumps(
            {
                "id": "HumanEval/0",
                "score": 0.8,
                "proxy_windows": 1,
                "scoreable_contexts": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    selected, analysis = module.select_candidate_rows(
        [baseline_path, candidate_path],
        [baseline_detail_path, candidate_detail_path],
        ranking_mode="public_then_detector",
    )

    assert selected[0]["final_code"] == PROMPT + "    return x + 1\n# candidate\n"
    assert analysis["policy"]["ranking_mode"] == "public_then_detector"
