from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from wfcllm.sawr.config import SawrGenerationConfig, SawrPipelineConfig, SawrRuleConfig
from wfcllm.sawr.generator import SawrGenerateResult
from wfcllm.sawr.pipeline import FORBIDDEN_FINAL_FIELDS, SawrPipeline
from wfcllm.sawr.state_machine import AuditEvent


class FakeGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, int, int]] = []

    def generate(
        self,
        sample_id: str,
        prompt: str,
        dataset: str,
        max_group_statements: int,
        retry_budget: int,
    ) -> SawrGenerateResult:
        self.calls.append(
            (sample_id, prompt, dataset, max_group_statements, retry_budget)
        )
        return SawrGenerateResult(
            final_code=prompt + "    return 1\n",
            accepted_hit_count=1,
            closed_without_hit_count=0,
            fallback_count=0,
            candidate_count=1,
            audit_events=[
                AuditEvent(
                    sample_id=sample_id,
                    position_id="module.foo.body",
                    event="accepted_generation_time_group",
                    candidate_type="simple_statement",
                    group_statement_count=1,
                    final_flush=False,
                    rule_name="hash",
                    decision="hit",
                    reason="forced",
                    candidate_hash="abc",
                )
            ],
        )


class FailingGenerator:
    def generate(
        self,
        sample_id: str,
        prompt: str,
        dataset: str,
        max_group_statements: int,
        retry_budget: int,
    ) -> SawrGenerateResult:
        raise RuntimeError("boom")


class UnsupportedAuditGenerator:
    def generate(
        self,
        sample_id: str,
        prompt: str,
        dataset: str,
        max_group_statements: int,
        retry_budget: int,
    ) -> SawrGenerateResult:
        return SawrGenerateResult(
            final_code=prompt + "    return 1\n",
            accepted_hit_count=0,
            closed_without_hit_count=0,
            fallback_count=0,
            candidate_count=1,
            audit_events=[
                AuditEvent(
                    sample_id=sample_id,
                    position_id="module.foo.body",
                    event="unsupported_event",
                    candidate_type="simple_statement",
                    group_statement_count=1,
                    final_flush=False,
                    rule_name="hash",
                    decision="miss",
                    reason="bad",
                    candidate_hash="abc",
                )
            ],
        )


def _config(tmp_path: Path, **overrides: object) -> SawrPipelineConfig:
    model_path = tmp_path / "model"
    model_path.mkdir(exist_ok=True)
    values = {
        "dataset": "humaneval",
        "dataset_path": "data/datasets",
        "output_dir": str(tmp_path / "sawr"),
        "generation": SawrGenerationConfig(model_path=str(model_path), device="cpu"),
        "rule": SawrRuleConfig(target_accept_rate=1.0),
        "sample_limit": None,
        "sample_offset": None,
        "max_group_statements": 2,
        "retry_budget": 1,
        "resume": None,
    }
    values.update(overrides)
    return SawrPipelineConfig(**values)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_pipeline_writes_final_rows_without_forbidden_fields(tmp_path: Path) -> None:
    prompts = [{"id": "HumanEval/0", "prompt": "def foo():\n"}]
    generator = FakeGenerator()
    pipeline = SawrPipeline(generator=generator, config=_config(tmp_path))

    with patch("wfcllm.sawr.pipeline.load_prompts", return_value=prompts):
        final_path = Path(pipeline.run())

    rows = _read_jsonl(final_path)
    assert len(rows) == 1
    row = rows[0]
    assert set(row) == {
        "artifact_type",
        "schema_version",
        "id",
        "dataset",
        "prompt",
        "final_code",
        "scientific_claims_enabled",
    }
    assert row["artifact_type"] == "sawr_final_code"
    assert row["schema_version"] == "sawr-smoke/v1"
    assert row["id"] == "HumanEval/0"
    assert row["dataset"] == "humaneval"
    assert row["prompt"] == "def foo():\n"
    assert row["final_code"] == "def foo():\n    return 1\n"
    assert row["scientific_claims_enabled"] is False
    assert not (FORBIDDEN_FINAL_FIELDS & set(row))


def test_pipeline_writes_audit_rows_as_audit_only(tmp_path: Path) -> None:
    prompts = [{"id": "HumanEval/0", "prompt": "def foo():\n"}]
    generator = FakeGenerator()
    pipeline = SawrPipeline(generator=generator, config=_config(tmp_path))

    with patch("wfcllm.sawr.pipeline.load_prompts", return_value=prompts):
        final_path = Path(pipeline.run())

    audit_path = Path(str(final_path).replace("_final_", "_audit_"))
    rows = _read_jsonl(audit_path)
    assert rows == [
        {
            "artifact_type": "sawr_audit_event",
            "schema_version": "sawr-smoke/v1",
            "id": "HumanEval/0",
            "audit_only": True,
            "detector_input_allowed": False,
            "scientific_claims_enabled": False,
            "position_id": "module.foo.body",
            "event": "accepted_generation_time_group",
            "candidate_type": "simple_statement",
            "group_statement_count": 1,
            "final_flush": False,
            "rule_name": "hash",
            "decision": "hit",
            "reason": "forced",
            "candidate_hash": "abc",
        }
    ]


def test_pipeline_passes_sample_slice_to_loader(tmp_path: Path) -> None:
    prompts = [{"id": "HumanEval/0", "prompt": "def foo():\n"}]
    config = _config(tmp_path, sample_limit=7, sample_offset=3)
    pipeline = SawrPipeline(generator=FakeGenerator(), config=config)

    with patch("wfcllm.sawr.pipeline.load_prompts", return_value=prompts) as loader:
        pipeline.run()

    loader.assert_called_once_with(
        "humaneval",
        "data/datasets",
        sample_limit=7,
        sample_offset=3,
    )


def test_pipeline_resume_latest_skips_completed_ids(tmp_path: Path) -> None:
    out_dir = tmp_path / "sawr"
    out_dir.mkdir()
    final_path = out_dir / "humaneval_sawr_final_20260611_010101.jsonl"
    audit_path = out_dir / "humaneval_sawr_audit_20260611_010101.jsonl"
    final_path.write_text(
        json.dumps(
            {
                "artifact_type": "sawr_final_code",
                "schema_version": "sawr-smoke/v1",
                "id": "HumanEval/0",
                "dataset": "humaneval",
                "prompt": "def done():\n",
                "final_code": "def done():\n    pass\n",
                "scientific_claims_enabled": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    audit_path.write_text("", encoding="utf-8")
    prompts = [
        {"id": "HumanEval/0", "prompt": "def done():\n"},
        {"id": "HumanEval/1", "prompt": "def next_one():\n"},
    ]
    generator = FakeGenerator()
    pipeline = SawrPipeline(generator=generator, config=_config(tmp_path, resume="latest"))

    with patch("wfcllm.sawr.pipeline.load_prompts", return_value=prompts):
        resumed_path = Path(pipeline.run())

    assert resumed_path == final_path
    assert generator.calls == [("HumanEval/1", "def next_one():\n", "humaneval", 2, 1)]
    rows = _read_jsonl(final_path)
    assert [row["id"] for row in rows] == ["HumanEval/0", "HumanEval/1"]


def test_pipeline_resume_latest_requires_paired_audit_file(tmp_path: Path) -> None:
    out_dir = tmp_path / "sawr"
    out_dir.mkdir()
    final_path = out_dir / "humaneval_sawr_final_20260611_010101.jsonl"
    final_path.write_text(
        json.dumps(
            {
                "artifact_type": "sawr_final_code",
                "schema_version": "sawr-smoke/v1",
                "id": "HumanEval/0",
                "dataset": "humaneval",
                "prompt": "def done():\n",
                "final_code": "def done():\n    pass\n",
                "scientific_claims_enabled": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pipeline = SawrPipeline(generator=FakeGenerator(), config=_config(tmp_path, resume="latest"))

    with pytest.raises(ValueError, match="paired audit file missing"):
        pipeline.run()


def test_pipeline_records_sample_failed_and_continues(tmp_path: Path) -> None:
    prompts = [
        {"id": "HumanEval/0", "prompt": "def bad():\n"},
        {"id": "HumanEval/1", "prompt": "def still_runs():\n"},
    ]
    generator = FailingGenerator()
    pipeline = SawrPipeline(generator=generator, config=_config(tmp_path))

    with patch("wfcllm.sawr.pipeline.load_prompts", return_value=prompts):
        final_path = Path(pipeline.run())

    assert _read_jsonl(final_path) == []
    audit_rows = _read_jsonl(Path(str(final_path).replace("_final_", "_audit_")))
    assert [row["event"] for row in audit_rows] == ["sample_failed", "sample_failed"]
    assert [row["id"] for row in audit_rows] == ["HumanEval/0", "HumanEval/1"]
    assert all(row["audit_only"] is True for row in audit_rows)
    assert all(row["detector_input_allowed"] is False for row in audit_rows)
    assert all(row["scientific_claims_enabled"] is False for row in audit_rows)


def test_pipeline_rejects_unsupported_audit_events(tmp_path: Path) -> None:
    prompts = [{"id": "HumanEval/0", "prompt": "def foo():\n"}]
    pipeline = SawrPipeline(generator=UnsupportedAuditGenerator(), config=_config(tmp_path))

    with patch("wfcllm.sawr.pipeline.load_prompts", return_value=prompts):
        with pytest.raises(ValueError, match="unsupported SAWR audit event"):
            pipeline.run()
