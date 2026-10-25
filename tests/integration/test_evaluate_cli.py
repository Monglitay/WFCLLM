from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _module():
    spec = importlib.util.spec_from_file_location(
        "_wfcllm_evaluate", ROOT / "scripts" / "evaluate.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evaluate_defaults_to_posthoc_pass_at_1(tmp_path, capsys) -> None:
    source = tmp_path / "correctness.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"task_id": "a", "is_correct": True},
                {"task_id": "a", "is_correct": False},
                {"task_id": "b", "is_correct": True},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "pass_report_posthoc.json"

    assert _module().main([str(source), "--output", str(output)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["metric"] == "pass@1"
    assert payload["k"] == 1
    assert payload["value"] == pytest.approx(0.75)
    assert payload["passed_count"] == 2
    assert payload["total_count"] == 3
    assert payload["posthoc_only"] is True
    assert payload["not_used_for_detection"] is True
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_evaluate_rejects_removed_pass_at_k_option(tmp_path) -> None:
    source = tmp_path / "correctness.jsonl"
    source.write_text(
        json.dumps({"task_id": "a", "is_correct": True}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit):
        _module().main([str(source), "--k", "2"])


def test_evaluate_rejects_non_boolean_correctness(tmp_path) -> None:
    source = tmp_path / "bad.jsonl"
    source.write_text(
        json.dumps({"task_id": "a", "is_correct": 1}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="boolean is_correct"):
        _module().main([str(source)])


def test_evaluate_generated_input_writes_posthoc_correctness(
    tmp_path, monkeypatch, capsys
) -> None:
    module = _module()
    generated = tmp_path / "final_code.jsonl"
    generated.write_text(
        json.dumps(
            {
                "id": "HumanEval/0",
                "dataset": "humaneval",
                "prompt": "def add_one(x):\n",
                "final_code": "def add_one(x):\n    return x + 1\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    stdlib = tmp_path / "stdlib.json"
    stdlib.write_text("{}\n", encoding="utf-8")

    class _Case:
        entry_point = "add_one"
        test_code = "def check(candidate):\n    assert candidate(1) == 2\n"

    monkeypatch.setattr(
        module,
        "load_test_cases",
        lambda dataset, dataset_path: {"HumanEval/0": _Case()},
    )

    correctness = tmp_path / "correctness.jsonl"
    output = tmp_path / "pass_report_posthoc.json"
    assert (
        module.main(
            [
                "--generated-input",
                str(generated),
                "--dataset",
                "humaneval",
                "--dataset-path",
                str(tmp_path / "datasets"),
                "--correctness-output",
                str(correctness),
                "--sandbox-stdlib-path",
                str(stdlib),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    row = json.loads(correctness.read_text(encoding="utf-8"))
    assert row["task_id"] == "HumanEval/0"
    assert row["is_correct"] is True
    assert row["posthoc_only"] is True
    payload = json.loads(capsys.readouterr().out)
    assert payload["value"] == pytest.approx(1.0)
    assert payload["not_used_for_detection"] is True


def test_evaluate_generated_input_requires_strict_code_only(tmp_path) -> None:
    generated = tmp_path / "bad_final_code.jsonl"
    generated.write_text(
        json.dumps(
            {
                "id": "HumanEval/0",
                "dataset": "humaneval",
                "prompt": "def add_one(x):\n",
                "final_code": "def add_one(x):\n    return x + 1\n",
                "correctness": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    stdlib = tmp_path / "stdlib.json"
    stdlib.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly"):
        _module().main(
            [
                "--generated-input",
                str(generated),
                "--dataset",
                "humaneval",
                "--correctness-output",
                str(tmp_path / "correctness.jsonl"),
                "--sandbox-stdlib-path",
                str(stdlib),
            ]
        )


def test_cpp_generated_input_retains_wrong_and_uncompilable_samples(tmp_path, capsys):
    from wfcllm.evaluation.cpp_execution import require_cpp_toolchain

    try:
        require_cpp_toolchain()
    except ValueError as error:
        pytest.skip(str(error))
    (tmp_path / "fixture-manifest.json").write_text(
        json.dumps({"formal_eligible": False, "diagnostic_only": True,
                    "diagnostic_test_backend": True, "not_official_method": True}) + "\n",
        encoding="utf-8",
    )
    dataset = tmp_path / "datasets" / "humanevalpack" / "cpp"
    dataset.mkdir(parents=True)
    original_test = "#undef NDEBUG\n#include<assert.h>\nint main(){assert(plus_one(2)==3);}\n"
    cases = [{"task_id": f"CPP/{i}", "prompt": "int plus_one(int x){", "test": original_test}
             for i in range(3)]
    (dataset / "test.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in cases), encoding="utf-8",
    )
    generated = tmp_path / "final_code.jsonl"
    sources = ["int plus_one(int x){return x+1;}", "int plus_one(int x){return x;}", "bad syntax"]
    generated.write_text("".join(json.dumps({"id": case["task_id"], "dataset": "humanevalpack",
                                             "prompt": case["prompt"], "final_code": source}) + "\n"
                                 for case, source in zip(cases, sources)), encoding="utf-8")
    correctness = tmp_path / "correctness.jsonl"
    assert _module().main(["--language", "cpp", "--dataset", "humanevalpack",
                           "--dataset-path", str(tmp_path / "datasets"),
                           "--generated-input", str(generated),
                           "--correctness-output", str(correctness)]) == 0
    rows = [json.loads(line) for line in correctness.read_text(encoding="utf-8").splitlines()]
    assert [row["is_correct"] for row in rows] == [True, False, False]
    assert all(row["posthoc_only"] for row in rows)
    report = json.loads(capsys.readouterr().out)
    assert report["total_count"] == report["sample_count"] == 3
    assert report["passed_count"] == 1
    assert report["value"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("language,prefix", [("java", "Java"), ("js", "JavaScript")])
def test_native_generated_input_preserves_all_failures(tmp_path, capsys, language, prefix):
    from wfcllm.evaluation.humanevalpack_execution import require_native_toolchain

    try:
        require_native_toolchain(language)
    except ValueError as error:
        pytest.skip(str(error))
    (tmp_path / "fixture-manifest.json").write_text(
        json.dumps({"formal_eligible": False, "diagnostic_only": True,
                    "diagnostic_test_backend": True, "not_official_method": True}) + "\n",
        encoding="utf-8",
    )
    if language == "java":
        test = "public class Main { public static void main(String[] a) { if(new Solution().plusOne(2)!=3) throw new AssertionError(); } }"
        sources = ["class Solution { int plusOne(int x) { return x+1; } }",
                   "class Solution { int plusOne(int x) { return x; } }", "invalid syntax"]
    else:
        test = "console.assert(plusOne(2) === 3);"
        sources = ["const plusOne=x=>x+1;", "const plusOne=x=>x;", "const plusOne = ;"]
    dataset = tmp_path / "datasets/humanevalpack" / language
    dataset.mkdir(parents=True)
    cases = [{"task_id": f"{prefix}/{i}", "prompt": "implement plusOne", "test": test} for i in range(3)]
    (dataset / "test.jsonl").write_text("".join(json.dumps(row) + "\n" for row in cases), encoding="utf-8")
    generated = tmp_path / "final_code.jsonl"
    generated.write_text("".join(json.dumps({"id": case["task_id"], "dataset": "humanevalpack",
                                             "prompt": case["prompt"], "final_code": source}) + "\n"
                                 for case, source in zip(cases, sources)), encoding="utf-8")
    correctness = tmp_path / "correctness.jsonl"
    before = generated.read_bytes()
    assert _module().main(["--language", language, "--dataset", "humanevalpack",
                           "--dataset-path", str(tmp_path / "datasets"),
                           "--generated-input", str(generated), "--correctness-output", str(correctness)]) == 0
    assert generated.read_bytes() == before
    rows = [json.loads(line) for line in correctness.read_text(encoding="utf-8").splitlines()]
    assert [row["is_correct"] for row in rows] == [True, False, False]
    assert all(row["posthoc_only"] and row["language"] == language for row in rows)
    report = json.loads(capsys.readouterr().out)
    assert report["total_count"] == report["sample_count"] == 3
    assert report["passed_count"] == 1
    assert report["value"] == pytest.approx(1 / 3)
