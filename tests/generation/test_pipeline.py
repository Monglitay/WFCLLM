from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from wfcllm.generation.generator import WFCLLMGenerateResult
from wfcllm.generation.pipeline import FORBIDDEN_FINAL_FIELDS, WFCLLMGenerationPipeline
from wfcllm.generation.state_machine import AuditEvent
from wfcllm.method.config import (
    WFCLLMGenerationConfig,
    WFCLLMPipelineConfig,
    WFCLLMRuleConfig,
)


class FakeGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, int, int, int, int]] = []

    def generate(
        self,
        sample_id: str,
        prompt: str,
        dataset: str,
        max_group_statements: int,
        retry_budget: int,
        global_rollback_budget: int,
        max_total_sampled_tokens: int,
    ) -> WFCLLMGenerateResult:
        self.calls.append(
            (
                sample_id,
                prompt,
                dataset,
                max_group_statements,
                retry_budget,
                global_rollback_budget,
                max_total_sampled_tokens,
            )
        )
        return WFCLLMGenerateResult(
            final_code=prompt + "    return 1\n",
            accepted_hit_count=1,
            closed_without_hit_count=0,
            fallback_count=0,
            candidate_count=1,
            audit_events=[
                AuditEvent(
                    sample_id=sample_id,
                    position_id="module.foo.body",
                    event="accepted_generation_time_window",
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
        global_rollback_budget: int,
        max_total_sampled_tokens: int,
    ) -> WFCLLMGenerateResult:
        raise RuntimeError("boom")


class UnsupportedAuditGenerator:
    def generate(
        self,
        sample_id: str,
        prompt: str,
        dataset: str,
        max_group_statements: int,
        retry_budget: int,
        global_rollback_budget: int,
        max_total_sampled_tokens: int,
    ) -> WFCLLMGenerateResult:
        return WFCLLMGenerateResult(
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


class EvidenceRetryGenerator:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        sample_id: str,
        prompt: str,
        dataset: str,
        max_group_statements: int,
        retry_budget: int,
        global_rollback_budget: int,
        max_total_sampled_tokens: int,
        seed_override: int | None = None,
    ) -> WFCLLMGenerateResult:
        self.calls.append(
            {
                "sample_id": sample_id,
                "seed_override": seed_override,
            }
        )
        attempt_index = len(self.calls) - 1
        accepted_hits = [0, 2, 1][attempt_index]
        return WFCLLMGenerateResult(
            final_code=prompt + f"    return {attempt_index}\n",
            accepted_hit_count=accepted_hits,
            closed_without_hit_count=2 - accepted_hits,
            fallback_count=attempt_index,
            candidate_count=attempt_index + 1,
            audit_events=[
                AuditEvent(
                    sample_id=sample_id,
                    position_id="module.foo.body",
                    event="accepted_generation_time_window",
                    candidate_type="simple_statement",
                    group_statement_count=1,
                    final_flush=False,
                    rule_name="hash",
                    decision="hit",
                    reason=f"attempt={attempt_index}",
                    candidate_hash="abc",
                )
            ],
        )


def _config(tmp_path: Path, **overrides: object) -> WFCLLMPipelineConfig:
    model_path = tmp_path / "model"
    model_path.mkdir(exist_ok=True)
    values = {
        "dataset": "humaneval",
        "dataset_path": "data/datasets",
        "output_dir": str(tmp_path / "sawr"),
        "generation": WFCLLMGenerationConfig(model_path=str(model_path), device="cpu"),
        "rule": WFCLLMRuleConfig(target_accept_rate=1.0),
        "sample_limit": None,
        "sample_offset": None,
        "max_group_statements": 2,
        "retry_budget": 1,
        "global_rollback_budget": 2,
        "max_total_sampled_tokens": 100,
        "evidence_retry_attempts": 1,
        "evidence_retry_seed_stride": 1009,
        "resume": None,
    }
    values.update(overrides)
    return WFCLLMPipelineConfig(**values)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_pipeline_writes_final_rows_without_forbidden_fields(tmp_path: Path) -> None:
    prompts = [{"id": "HumanEval/0", "prompt": "def foo():\n"}]
    generator = FakeGenerator()
    pipeline = WFCLLMGenerationPipeline(generator=generator, config=_config(tmp_path))

    with patch("wfcllm.generation.pipeline.load_prompts", return_value=prompts):
        final_path = Path(pipeline.run())

    rows = _read_jsonl(final_path)
    assert len(rows) == 1
    row = rows[0]
    assert set(row) == {"id", "dataset", "prompt", "final_code"}
    assert row["id"] == "HumanEval/0"
    assert row["dataset"] == "humaneval"
    assert row["prompt"] == "def foo():\n"
    assert row["final_code"] == "def foo():\n    return 1\n"
    assert "artifact_type" not in row
    assert "schema_version" not in row
    assert "scientific_claims_enabled" not in row
    assert not (FORBIDDEN_FINAL_FIELDS & set(row))


def test_pipeline_evidence_retry_selects_by_generation_evidence_only(
    tmp_path: Path,
) -> None:
    prompts = [{"id": "HumanEval/0", "prompt": "def foo():\n"}]
    generator = EvidenceRetryGenerator()
    pipeline = WFCLLMGenerationPipeline(
        generator=generator,
        config=_config(
            tmp_path,
            evidence_retry_attempts=3,
            evidence_retry_seed_stride=11,
        ),
    )

    with patch("wfcllm.generation.pipeline.load_prompts", return_value=prompts):
        final_path = Path(pipeline.run())

    rows = _read_jsonl(final_path)
    assert rows == [
        {
            "id": "HumanEval/0",
            "dataset": "humaneval",
            "prompt": "def foo():\n",
            "final_code": "def foo():\n    return 1\n",
        }
    ]
    assert generator.calls == [
        {"sample_id": "HumanEval/0", "seed_override": 0},
        {"sample_id": "HumanEval/0", "seed_override": 11},
        {"sample_id": "HumanEval/0", "seed_override": 22},
    ]
    assert not (FORBIDDEN_FINAL_FIELDS & set(rows[0]))


def test_pipeline_final_rows_remain_detector_clean_with_structure_audit(
    tmp_path: Path,
) -> None:
    prompts = [{"id": "HumanEval/0", "prompt": "def foo():\n"}]
    generator = FakeGenerator()
    pipeline = WFCLLMGenerationPipeline(generator=generator, config=_config(tmp_path))

    with patch("wfcllm.generation.pipeline.load_prompts", return_value=prompts):
        final_path = Path(pipeline.run())

    row = _read_jsonl(final_path)[0]
    assert set(row) == {"id", "dataset", "prompt", "final_code"}
    assert not (FORBIDDEN_FINAL_FIELDS & set(row))


def test_pipeline_writes_audit_rows_as_audit_only(tmp_path: Path) -> None:
    prompts = [{"id": "HumanEval/0", "prompt": "def foo():\n"}]
    generator = FakeGenerator()
    pipeline = WFCLLMGenerationPipeline(generator=generator, config=_config(tmp_path))

    with patch("wfcllm.generation.pipeline.load_prompts", return_value=prompts):
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
            "event": "accepted_generation_time_window",
            "candidate_type": "simple_statement",
            "group_statement_count": 1,
            "final_flush": False,
            "rule_name": "hash",
            "decision": "hit",
            "reason": "forced",
            "candidate_hash": "abc",
        }
    ]
    assert "normalized_text" not in rows[0]
    assert "normalized_text_hash" not in rows[0]
    assert "parent_node_type" not in rows[0]
    assert "ordinal" not in rows[0]


def test_pipeline_writes_candidate_text_sidecar_when_requested(
    tmp_path: Path,
) -> None:
    prompts = [{"id": "HumanEval/0", "prompt": "def foo():\n"}]
    event = AuditEvent(
        sample_id="HumanEval/0",
        position_id="module.foo.body",
        event="accepted_generation_time_window",
        candidate_type="simple_statement",
        group_statement_count=1,
        final_flush=False,
        rule_name="semantic_lsh",
        decision="hit",
        reason=(
            "lsh_signature=(1, 0, 1, 0);"
            "in_valid_set=True;"
            "min_margin=0.420000000000;"
            "margin=0.000000000000;"
            "parent_node_type=function_definition;"
            "ordinal=None;"
            "k=4;"
            "gamma_target=0.250000000000;"
            "gamma_effective=0.250000000000"
        ),
        candidate_hash="f3d1b0b8c1ec9f4c6a4b85a161e70c45fa7c6ce8d68378d06ca222597e6e1731",
        node_type="return_statement",
        parent_node_type="function_definition",
        ordinal=0,
        normalized_text="return 1",
        normalized_text_hash=(
            "f3d1b0b8c1ec9f4c6a4b85a161e70c45fa7c6ce8d68378d06ca222597e6e1731"
        ),
    )
    generator = FakeGenerator()
    generator.generate = lambda **kwargs: WFCLLMGenerateResult(
        final_code="def foo():\n    return 1\n",
        accepted_hit_count=1,
        closed_without_hit_count=0,
        fallback_count=0,
        candidate_count=1,
        audit_events=[event],
    )
    sidecar_path = tmp_path / "candidate_sidecar.jsonl"
    pipeline = WFCLLMGenerationPipeline(
        generator=generator,
        config=_config(tmp_path, candidate_sidecar_output=str(sidecar_path)),
    )

    with patch("wfcllm.generation.pipeline.load_prompts", return_value=prompts):
        final_path = Path(pipeline.run())

    final_row = _read_jsonl(final_path)[0]
    assert "normalized_text" not in final_row
    audit_row = _read_jsonl(Path(str(final_path).replace("_final_", "_audit_")))[0]
    assert "normalized_text" not in audit_row

    rows = _read_jsonl(sidecar_path)
    assert rows == [
        {
            "artifact_type": "sawr_generation_candidate_text_sidecar",
            "schema_version": "sawr-e1-e2-diagnostic/v1",
            "id": "HumanEval/0",
            "audit_only": True,
            "detector_input_allowed": False,
            "scientific_claims_enabled": False,
            "event": "accepted_generation_time_window",
            "decision": "hit",
            "candidate_hash": event.candidate_hash,
            "candidate_type": "simple_statement",
            "position_id": "module.foo.body",
            "node_type": "return_statement",
            "parent_node_type": "function_definition",
            "ordinal": 0,
            "group_statement_count": 1,
            "normalized_text": "return 1",
            "normalized_text_hash": event.normalized_text_hash,
            "rule_name": "semantic_lsh",
            "reason": event.reason,
            "lsh_signature_from_audit_reason": [1, 0, 1, 0],
            "in_valid_set_from_audit_reason": True,
            "min_margin_from_audit_reason": 0.42,
            "k_from_audit_reason": 4,
            "gamma_target_from_audit_reason": 0.25,
            "gamma_effective_from_audit_reason": 0.25,
            "final_flush": False,
        }
    ]


def test_pipeline_allows_structure_aware_audit_events(tmp_path: Path) -> None:
    prompts = [{"id": "HumanEval/0", "prompt": "def foo():\n"}]
    event_names = [
        "compound_layer_started",
        "simple_candidate_observed",
        "compound_layer_window_observed",
        "layer_window_rule_miss",
        "accepted_generation_time_window",
        "statement_retry_requested",
        "window_retry_requested",
        "layer_retry_requested",
        "retry_layer_early_closed_after_hit",
        "layer_disappeared_without_hit",
        "layer_closed_with_child_hit",
        "layer_closed_with_direct_hit",
        "layer_fallback_committed_without_hit",
        "global_rollback_budget_exhausted",
        "absolute_sampled_token_budget_exhausted",
    ]
    generator = FakeGenerator()
    generator.generate = lambda **kwargs: WFCLLMGenerateResult(
        final_code="def foo():\n    return 1\n",
        accepted_hit_count=1,
        closed_without_hit_count=0,
        fallback_count=0,
        candidate_count=1,
        audit_events=[
            AuditEvent(
                sample_id="HumanEval/0",
                position_id="module.foo.body",
                event=name,
                candidate_type="simple_statement",
                group_statement_count=1,
                final_flush=False,
                rule_name="hash",
                decision="hit",
                reason="test",
                candidate_hash="abc",
            )
            for name in event_names
        ],
    )
    pipeline = WFCLLMGenerationPipeline(generator=generator, config=_config(tmp_path))

    with patch("wfcllm.generation.pipeline.load_prompts", return_value=prompts):
        final_path = Path(pipeline.run())

    audit_path = Path(str(final_path).replace("_final_", "_audit_"))
    rows = _read_jsonl(audit_path)
    assert [row["event"] for row in rows] == event_names
    assert all(row["audit_only"] is True for row in rows)
    assert all(row["detector_input_allowed"] is False for row in rows)


def test_pipeline_passes_sample_slice_to_loader(tmp_path: Path) -> None:
    prompts = [{"id": "HumanEval/0", "prompt": "def foo():\n"}]
    config = _config(tmp_path, sample_limit=7, sample_offset=3)
    pipeline = WFCLLMGenerationPipeline(generator=FakeGenerator(), config=config)

    with patch("wfcllm.generation.pipeline.load_prompts", return_value=prompts) as loader:
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
                "id": "HumanEval/0",
                "dataset": "humaneval",
                "prompt": "def done():\n",
                "final_code": "def done():\n    pass\n",
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
    pipeline = WFCLLMGenerationPipeline(
        generator=generator,
        config=_config(tmp_path, resume="latest"),
    )

    with patch("wfcllm.generation.pipeline.load_prompts", return_value=prompts):
        resumed_path = Path(pipeline.run())

    assert resumed_path == final_path
    assert generator.calls == [
        ("HumanEval/1", "def next_one():\n", "humaneval", 2, 1, 2, 100)
    ]
    rows = _read_jsonl(final_path)
    assert [row["id"] for row in rows] == ["HumanEval/0", "HumanEval/1"]


def test_pipeline_resume_latest_requires_paired_audit_file(tmp_path: Path) -> None:
    out_dir = tmp_path / "sawr"
    out_dir.mkdir()
    final_path = out_dir / "humaneval_sawr_final_20260611_010101.jsonl"
    final_path.write_text(
        json.dumps(
            {
                "id": "HumanEval/0",
                "dataset": "humaneval",
                "prompt": "def done():\n",
                "final_code": "def done():\n    pass\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pipeline = WFCLLMGenerationPipeline(
        generator=FakeGenerator(),
        config=_config(tmp_path, resume="latest"),
    )

    with pytest.raises(ValueError, match="paired audit file missing"):
        pipeline.run()


def test_pipeline_records_sample_failed_and_continues(tmp_path: Path) -> None:
    prompts = [
        {"id": "HumanEval/0", "prompt": "def bad():\n"},
        {"id": "HumanEval/1", "prompt": "def still_runs():\n"},
    ]
    generator = FailingGenerator()
    pipeline = WFCLLMGenerationPipeline(generator=generator, config=_config(tmp_path))

    with patch("wfcllm.generation.pipeline.load_prompts", return_value=prompts):
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
    pipeline = WFCLLMGenerationPipeline(
        generator=UnsupportedAuditGenerator(),
        config=_config(tmp_path),
    )

    with patch("wfcllm.generation.pipeline.load_prompts", return_value=prompts):
        with pytest.raises(ValueError, match="unsupported SAWR audit event"):
            pipeline.run()
