from __future__ import annotations

import json
from pathlib import Path

import scripts.wfcllm_sanitize_final_code as sanitize_cli


def test_sanitize_final_code_writes_exact_four_fields(tmp_path: Path) -> None:
    source = tmp_path / "legacy.jsonl"
    output = tmp_path / "nested" / "final_code.jsonl"
    source.write_text(
        "\n"
        + json.dumps(
            {
                "id": "HumanEval/0",
                "dataset": "humaneval",
                "prompt": "def foo():\n",
                "generated_code": "def foo():\n    return 1\n",
                "watermark_params": {"secret_key": "blocked"},
                "token_channel": {"blocked": True},
                "blocks": [{"blocked": True}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rc = sanitize_cli.main(["--input", str(source), "--output", str(output)])

    assert rc == 0
    row = json.loads(output.read_text(encoding="utf-8").strip())
    assert row == {
        "id": "HumanEval/0",
        "dataset": "humaneval",
        "prompt": "def foo():\n",
        "final_code": "def foo():\n    return 1\n",
    }


def test_sanitize_final_code_prefers_final_code_and_legacy_defaults(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy.jsonl"
    output = tmp_path / "final_code.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "HumanEval/1",
                "final_code": "def foo():\n    return 2\n",
                "generated_code": "def foo():\n    return 1\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rc = sanitize_cli.main(["--input", str(source), "--output", str(output)])

    assert rc == 0
    row = json.loads(output.read_text(encoding="utf-8").strip())
    assert row == {
        "id": "HumanEval/1",
        "dataset": "humaneval",
        "prompt": "",
        "final_code": "def foo():\n    return 2\n",
    }


def test_sanitize_final_code_returns_one_for_invalid_row(
    tmp_path: Path,
    capsys,
) -> None:
    source = tmp_path / "legacy.jsonl"
    output = tmp_path / "final_code.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "HumanEval/2",
                "dataset": "unsupported",
                "prompt": "",
                "generated_code": "def foo():\n    return 1\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rc = sanitize_cli.main(["--input", str(source), "--output", str(output)])

    assert rc == 1
    assert "[错误]" in capsys.readouterr().err
