from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from wfcllm.sawr.config import SawrPipelineConfig
from wfcllm.sawr.generator import SawrGenerateResult, SawrGenerator
from wfcllm.sawr.state_machine import AuditEvent

FORBIDDEN_FINAL_FIELDS = {
    "generated_code",
    "watermark_params",
    "blocks",
    "embed_rate",
    "p_value",
    "z_score",
    "is_watermarked",
    "tpr",
    "fpr",
    "correctness_result",
    "pass_cost",
    "detector_score",
    "generation_ledger",
    "retry_ledger",
}

ALLOWED_AUDIT_EVENTS = {
    "candidate_observed",
    "group_rule_miss",
    "accepted_generation_time_group",
    "rollback_requested",
    "fallback_committed_without_hit",
    "closed_without_hit",
    "sample_failed",
    "compound_layer_started",
    "simple_candidate_observed",
    "layer_window_rule_miss",
    "accepted_generation_time_window",
    "layer_retry_requested",
    "retry_layer_early_closed_after_hit",
    "layer_disappeared_without_hit",
    "layer_closed_with_child_hit",
    "layer_closed_with_direct_hit",
    "layer_fallback_committed_without_hit",
    "absolute_sampled_token_budget_exhausted",
    "global_rollback_budget_exhausted",
}


def load_prompts(
    dataset: str,
    dataset_path: str,
    sample_limit: int | None = None,
    sample_offset: int | None = None,
) -> list[dict[str, Any]]:
    from wfcllm.datasets.loaders.local import load_prompts as _load_prompts

    return _load_prompts(
        dataset,
        dataset_path,
        sample_limit=sample_limit,
        sample_offset=sample_offset,
    )


class SawrPipeline:
    """Batch SAWR smoke pipeline over local HumanEval or MBPP prompts."""

    def __init__(self, generator: SawrGenerator, config: SawrPipelineConfig) -> None:
        self._generator = generator
        self._config = config

    def run(self) -> str:
        out_dir = Path(self._config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        final_path, audit_path, mode = self._resolve_output_paths(out_dir)
        processed_ids = self._load_processed_ids(final_path) if mode == "a" else set()

        prompts = load_prompts(
            self._config.dataset,
            self._config.dataset_path,
            sample_limit=self._config.sample_limit,
            sample_offset=self._config.sample_offset,
        )

        with final_path.open(mode, encoding="utf-8") as final_file:
            with audit_path.open(mode, encoding="utf-8") as audit_file:
                for item in prompts:
                    sample_id = str(item["id"])
                    if sample_id in processed_ids:
                        continue

                    prompt = str(item["prompt"])
                    try:
                        result = self._generator.generate(
                            sample_id=sample_id,
                            prompt=prompt,
                            dataset=self._config.dataset,
                            max_group_statements=self._config.max_group_statements,
                            retry_budget=self._config.retry_budget,
                            global_rollback_budget=int(
                                self._config.global_rollback_budget
                            ),
                            max_total_sampled_tokens=int(
                                self._config.max_total_sampled_tokens
                            ),
                        )
                    except Exception as exc:
                        audit_file.write(
                            json.dumps(
                                self._build_sample_failed_audit_row(
                                    sample_id=sample_id,
                                    reason=str(exc),
                                ),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        audit_file.flush()
                        print(
                            f"[warning] {sample_id} SAWR smoke failed: {exc}",
                            file=sys.stderr,
                        )
                        continue

                    final_row = self._build_final_row(sample_id, prompt, result)
                    audit_rows = [
                        self._build_audit_row(sample_id, event)
                        for event in result.audit_events
                    ]
                    forbidden = FORBIDDEN_FINAL_FIELDS & set(final_row)
                    if forbidden:
                        raise ValueError(
                            "SAWR final row contains forbidden fields: "
                            f"{sorted(forbidden)}"
                        )

                    final_file.write(json.dumps(final_row, ensure_ascii=False) + "\n")
                    for audit_row in audit_rows:
                        audit_file.write(
                            json.dumps(audit_row, ensure_ascii=False) + "\n"
                        )
                    final_file.flush()
                    audit_file.flush()

        return str(final_path)

    def _resolve_output_paths(self, out_dir: Path) -> tuple[Path, Path, str]:
        if self._config.resume == "latest":
            final_path = self._find_latest_final_path(out_dir)
            if final_path is not None:
                audit_path = self._paired_audit_path(final_path)
                if not audit_path.exists():
                    raise ValueError(f"paired audit file missing for resume: {audit_path}")
                return final_path, audit_path, "a"

        final_path, audit_path = self._new_output_paths(out_dir)
        return final_path, audit_path, "w"

    def _new_output_paths(self, out_dir: Path) -> tuple[Path, Path]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return (
            out_dir / f"{self._config.dataset}_sawr_final_{timestamp}.jsonl",
            out_dir / f"{self._config.dataset}_sawr_audit_{timestamp}.jsonl",
        )

    def _find_latest_final_path(self, out_dir: Path) -> Path | None:
        matches = sorted(out_dir.glob(f"{self._config.dataset}_sawr_final_*.jsonl"))
        if not matches:
            return None
        return matches[-1]

    @staticmethod
    def _paired_audit_path(final_path: Path) -> Path:
        return final_path.with_name(final_path.name.replace("_final_", "_audit_"))

    @staticmethod
    def _load_processed_ids(final_path: Path) -> set[str]:
        if not final_path.exists():
            return set()

        processed: set[str] = set()
        with final_path.open(encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                sample_id = payload.get("id")
                if isinstance(sample_id, str):
                    processed.add(sample_id)
        return processed

    def _build_final_row(
        self,
        sample_id: str,
        prompt: str,
        result: SawrGenerateResult,
    ) -> dict[str, object]:
        return {
            "artifact_type": "sawr_final_code",
            "schema_version": "sawr-smoke/v1",
            "id": sample_id,
            "dataset": self._config.dataset,
            "prompt": prompt,
            "final_code": result.final_code,
            "scientific_claims_enabled": False,
        }

    @staticmethod
    def _build_audit_row(sample_id: str, event: AuditEvent) -> dict[str, object]:
        if event.event not in ALLOWED_AUDIT_EVENTS:
            raise ValueError(f"unsupported SAWR audit event: {event.event}")

        payload: dict[str, Any] = asdict(event)
        payload.pop("sample_id", None)
        return {
            "artifact_type": "sawr_audit_event",
            "schema_version": "sawr-smoke/v1",
            "id": sample_id,
            "audit_only": True,
            "detector_input_allowed": False,
            "scientific_claims_enabled": False,
            **payload,
        }

    @staticmethod
    def _build_sample_failed_audit_row(
        sample_id: str,
        reason: str,
    ) -> dict[str, object]:
        return {
            "artifact_type": "sawr_audit_event",
            "schema_version": "sawr-smoke/v1",
            "id": sample_id,
            "audit_only": True,
            "detector_input_allowed": False,
            "scientific_claims_enabled": False,
            "position_id": None,
            "event": "sample_failed",
            "candidate_type": None,
            "group_statement_count": 0,
            "final_flush": True,
            "rule_name": None,
            "decision": None,
            "reason": reason,
            "candidate_hash": None,
        }
