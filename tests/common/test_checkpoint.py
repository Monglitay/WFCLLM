from pathlib import Path

import pytest

from wfcllm.common.checkpoint import (
    find_latest_jsonl,
    load_processed_ids,
    resolve_resume_path,
)


def test_find_latest_jsonl_empty_dir_returns_none(tmp_path: Path):
    assert find_latest_jsonl(tmp_path) is None


def test_load_processed_ids_skips_blank_and_bad_lines(tmp_path: Path):
    path = tmp_path / "partial.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"id":"HumanEval/0"}',
                '{"id":"HumanEval/1"}',
                '{"id":"HumanEval/2"}',
                '{"id":"HumanEval/3"}',
                '{"id":"HumanEval/4"}',
                '{"id":"HumanEval/5"}',
                '{"id":"HumanEval/6"}',
                '{"id":"HumanEval/7"}',
                '{"id":"HumanEval/8"}',
                "",
                '{"id": }',
            ]
        ),
        encoding="utf-8",
    )
    assert load_processed_ids(path) == {
        "HumanEval/0",
        "HumanEval/1",
        "HumanEval/2",
        "HumanEval/3",
        "HumanEval/4",
        "HumanEval/5",
        "HumanEval/6",
        "HumanEval/7",
        "HumanEval/8",
    }


def test_load_processed_ids_raises_when_corruption_ratio_too_high(tmp_path: Path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"id":"ok"}\n{"id": }\n{"id": }\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Too many malformed lines"):
        load_processed_ids(path)


def test_resolve_resume_path_latest_without_match_returns_non_resume(tmp_path: Path):
    assert resolve_resume_path("latest", tmp_path) == (None, False)


def test_resolve_resume_path_explicit_missing_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        resolve_resume_path(str(tmp_path / "missing.jsonl"), tmp_path)
