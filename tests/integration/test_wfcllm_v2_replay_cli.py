from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import scripts.wfcllm_v2_replay as replay_cli


def _row(sample_id: str, score: float) -> dict[str, str]:
    return {
        "id": sample_id,
        "dataset": "humaneval",
        "prompt": "def target():\n",
        "final_code": f"def target():\n    return {score}\n",
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class _Scorer:
    def __init__(self, offset: float = 0.0) -> None:
        self._offset = offset

    def score_code(self, final_code: str):
        base = float(final_code.split("return ", 1)[1].strip())
        value = base + self._offset
        return SimpleNamespace(
            raw_score=value,
            unit_count=1,
            duplicate_count=0,
            total_bits=16,
            matched_bits=round(value * 16),
        )


def _args(
    positive: Path,
    negative: Path,
    output: Path,
    model_path: Path,
) -> list[str]:
    return [
        "--positive-input",
        str(positive),
        "--negative-input",
        str(negative),
        "--output",
        str(output),
        "--encoder-model-path",
        str(model_path),
        "--signature-bits",
        "16",
        "--encoder-use-lora",
        "--encoder-use-bf16",
    ]


def test_replay_cli_compares_saved_final_code_wrong_key_and_negatives(
    tmp_path: Path,
    monkeypatch,
) -> None:
    positive_path = tmp_path / "positive.jsonl"
    negative_path = tmp_path / "negative.jsonl"
    output_path = tmp_path / "replay.json"
    model_path = tmp_path / "encoder"
    model_path.mkdir()
    positives = [_row("HumanEval/0", 0.75), _row("HumanEval/1", 0.5)]
    negatives = [_row("HumanEval/2", 0.25)]
    _write_jsonl(positive_path, positives)
    _write_jsonl(negative_path, negatives)
    monkeypatch.setenv("WFCLLM_SECRET_KEY", "private-test-key")

    with patch(
        "scripts.wfcllm_v2_replay.load_v2_signature_scorer",
        side_effect=[_Scorer(), _Scorer(offset=-0.1)],
    ) as load:
        rc = replay_cli.main(
            _args(positive_path, negative_path, output_path, model_path)
        )

    assert rc == 0
    assert load.call_count == 2
    assert load.call_args_list[0].kwargs["secret_key"] == "private-test-key"
    assert load.call_args_list[1].kwargs["secret_key"] != "private-test-key"
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "wfcllm-v2-replay/v2"
    assert report["positive_sample_count"] == 2
    assert report["negative_sample_count"] == 1
    assert report["final_code_replay_rate"] == 1.0
    assert report["positive_correct_key_scores"] == [0.75, 0.5]
    assert report["positive_wrong_key_scores"] == [0.65, 0.4]
    assert report["negative_scores"] == [0.25]
    assert "private-test-key" not in output_path.read_text(encoding="utf-8")
    assert "secret_key" not in output_path.read_text(encoding="utf-8")


def test_replay_cli_checks_selected_retry_ledger_against_final_code(
    tmp_path: Path,
    monkeypatch,
) -> None:
    positive_path = tmp_path / "positive.jsonl"
    negative_path = tmp_path / "negative.jsonl"
    ledger_path = tmp_path / "ledger.jsonl"
    output_path = tmp_path / "replay.json"
    model_path = tmp_path / "encoder"
    model_path.mkdir()
    row = _row("HumanEval/0", 0.75)
    _write_jsonl(positive_path, [row])
    _write_jsonl(negative_path, [_row("HumanEval/2", 0.25)])
    _write_jsonl(
        ledger_path,
        [
            {
                "schema_version": "wfcllm-v2-retry-ledger/v2",
                "id": "HumanEval/0",
                "selected": True,
                "generation_score": 0.75,
                "final_code_sha256": replay_cli._sha256(row["final_code"]),
            }
        ],
    )
    monkeypatch.setenv("WFCLLM_SECRET_KEY", "private-test-key")

    with patch(
        "scripts.wfcllm_v2_replay.load_v2_signature_scorer",
        side_effect=[_Scorer(), _Scorer(offset=-0.1)],
    ):
        rc = replay_cli.main(
            [
                *_args(positive_path, negative_path, output_path, model_path),
                "--retry-ledger",
                str(ledger_path),
            ]
        )

    assert rc == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["ledger_selected_count"] == 1
    assert report["ledger_final_code_match_rate"] == 1.0
    assert report["ledger_score_replay_rate"] == 1.0
    assert report["per_positive"][0]["ledger_final_code_match"] is True
    assert report["per_positive"][0]["ledger_score_replay_equal"] is True


def test_replay_cli_missing_secret_fails_before_model_load(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    positive_path = tmp_path / "positive.jsonl"
    negative_path = tmp_path / "negative.jsonl"
    model_path = tmp_path / "encoder"
    model_path.mkdir()
    _write_jsonl(positive_path, [_row("HumanEval/0", 0.5)])
    _write_jsonl(negative_path, [_row("HumanEval/1", 0.5)])
    monkeypatch.delenv("WFCLLM_SECRET_KEY", raising=False)

    with patch("scripts.wfcllm_v2_replay.load_v2_signature_scorer") as load:
        rc = replay_cli.main(
            _args(positive_path, negative_path, tmp_path / "out.json", model_path)
        )

    assert rc == 1
    assert "WFCLLM_SECRET_KEY" in capsys.readouterr().err
    load.assert_not_called()
