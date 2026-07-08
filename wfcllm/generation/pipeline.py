from __future__ import annotations

import json
import sys
from ast import literal_eval
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from wfcllm.generation.generator import WFCLLMGenerateResult, WFCLLMGenerator
from wfcllm.generation.retry import evidence_retry_key
from wfcllm.generation.state_machine import AuditEvent
from wfcllm.method.artifacts import FinalCodeRecord
from wfcllm.method.config import WFCLLMPipelineConfig

FORBIDDEN_FINAL_FIELDS = {
    "artifact_type",
    "schema_version",
    "generated_code",
    "watermark_params",
    "blocks",
    "audit",
    "audit_only",
    "detector_input_allowed",
    "generation_ledger",
    "retry_ledger",
    "p_value",
    "z_score",
    "detector_score",
    "score",
    "is_watermarked",
    "pass",
    "passed",
    "correctness_result",
    "scientific_claims_enabled",
    "embed_rate",
    "tpr",
    "fpr",
    "pass_cost",
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


class WFCLLMGenerationPipeline:
    """Batch SAWR smoke pipeline over local HumanEval or MBPP prompts."""

    def __init__(self, generator: WFCLLMGenerator, config: WFCLLMPipelineConfig) -> None:
        self._generator = generator
        self._config = config

    def run(self) -> str:
        out_dir = Path(self._config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        final_path, audit_path, mode = self._resolve_output_paths(out_dir)
        processed_ids = self._load_processed_ids(final_path) if mode == "a" else set()
        candidate_sidecar_file = self._open_candidate_sidecar(mode)

        prompts = load_prompts(
            self._config.dataset,
            self._config.dataset_path,
            sample_limit=self._config.sample_limit,
            sample_offset=self._config.sample_offset,
        )

        try:
            with final_path.open(mode, encoding="utf-8") as final_file:
                with audit_path.open(mode, encoding="utf-8") as audit_file:
                    for item in prompts:
                        sample_id = str(item["id"])
                        if sample_id in processed_ids:
                            continue

                        prompt = str(item["prompt"])
                        try:
                            generate_kwargs: dict[str, object] = {
                                "sample_id": sample_id,
                                "prompt": prompt,
                                "dataset": self._config.dataset,
                                "max_group_statements": (
                                    self._config.max_group_statements
                                ),
                                "retry_budget": self._config.retry_budget,
                                "global_rollback_budget": int(
                                    self._config.global_rollback_budget
                                ),
                                "max_total_sampled_tokens": int(
                                    self._config.max_total_sampled_tokens
                                ),
                            }
                            if self._config.statement_retry_budget is not None:
                                generate_kwargs["statement_retry_budget"] = (
                                    self._config.statement_retry_budget
                                )
                            if self._config.window_retry_budget is not None:
                                generate_kwargs["window_retry_budget"] = (
                                    self._config.window_retry_budget
                                )
                            if self._config.compound_retry_budget is not None:
                                generate_kwargs["compound_retry_budget"] = (
                                    self._config.compound_retry_budget
                                )
                            result = self._generate_with_evidence_retry(
                                generate_kwargs,
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

                        final_file.write(
                            json.dumps(final_row, ensure_ascii=False) + "\n"
                        )
                        for audit_row in audit_rows:
                            audit_file.write(
                                json.dumps(audit_row, ensure_ascii=False) + "\n"
                            )
                        if candidate_sidecar_file is not None:
                            for sidecar_row in self._build_candidate_sidecar_rows(
                                sample_id,
                                result.audit_events,
                            ):
                                candidate_sidecar_file.write(
                                    json.dumps(sidecar_row, ensure_ascii=False) + "\n"
                                )
                        final_file.flush()
                        audit_file.flush()
                        if candidate_sidecar_file is not None:
                            candidate_sidecar_file.flush()
        finally:
            if candidate_sidecar_file is not None:
                candidate_sidecar_file.close()

        return str(final_path)

    def _generate_with_evidence_retry(
        self,
        generate_kwargs: dict[str, object],
    ) -> WFCLLMGenerateResult:
        attempts = int(self._config.evidence_retry_attempts)
        if attempts == 1:
            return self._generator.generate(**generate_kwargs)

        base_seed = int(self._config.generation.seed)
        seed_stride = int(self._config.evidence_retry_seed_stride)
        best_result: WFCLLMGenerateResult | None = None
        best_key: tuple[int, int, int, int, int] | None = None
        for attempt_index in range(attempts):
            result = self._generator.generate(
                **generate_kwargs,
                seed_override=base_seed + attempt_index * seed_stride,
            )
            key = self._evidence_retry_key(result, attempt_index)
            if best_key is None or key > best_key:
                best_result = result
                best_key = key
        if best_result is None:
            raise ValueError("evidence retry produced no generation attempts")
        return best_result

    @staticmethod
    def _evidence_retry_key(
        result: WFCLLMGenerateResult,
        attempt_index: int,
    ) -> tuple[int, int, int, int, int]:
        return evidence_retry_key(result, attempt_index)

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
        result: WFCLLMGenerateResult,
    ) -> dict[str, str]:
        return FinalCodeRecord(
            id=sample_id,
            dataset=self._config.dataset,
            prompt=prompt,
            final_code=result.final_code,
        ).to_dict()

    def _open_candidate_sidecar(self, mode: str):
        if self._config.candidate_sidecar_output is None:
            return None
        path = Path(self._config.candidate_sidecar_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.open(mode, encoding="utf-8")

    @staticmethod
    def _build_audit_row(sample_id: str, event: AuditEvent) -> dict[str, object]:
        if event.event not in ALLOWED_AUDIT_EVENTS:
            raise ValueError(f"unsupported SAWR audit event: {event.event}")

        payload: dict[str, Any] = asdict(event)
        payload.pop("sample_id", None)
        for diagnostic_field in (
            "node_type",
            "parent_node_type",
            "ordinal",
            "normalized_text",
            "normalized_text_hash",
        ):
            payload.pop(diagnostic_field, None)
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
    def _build_candidate_sidecar_rows(
        sample_id: str,
        events: list[AuditEvent],
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for event in events:
            if event.normalized_text is None:
                continue
            reason_fields = _parse_semantic_lsh_reason(event.reason)
            rows.append(
                {
                    "artifact_type": "sawr_generation_candidate_text_sidecar",
                    "schema_version": "sawr-e1-e2-diagnostic/v1",
                    "id": sample_id,
                    "audit_only": True,
                    "detector_input_allowed": False,
                    "scientific_claims_enabled": False,
                    "event": event.event,
                    "decision": event.decision,
                    "candidate_hash": event.candidate_hash,
                    "candidate_type": event.candidate_type,
                    "position_id": event.position_id,
                    "node_type": event.node_type,
                    "parent_node_type": event.parent_node_type,
                    "ordinal": event.ordinal,
                    "group_statement_count": event.group_statement_count,
                    "normalized_text": event.normalized_text,
                    "normalized_text_hash": event.normalized_text_hash,
                    "rule_name": event.rule_name,
                    "reason": event.reason,
                    "lsh_signature_from_audit_reason": reason_fields[
                        "lsh_signature"
                    ],
                    "in_valid_set_from_audit_reason": reason_fields["in_valid_set"],
                    "min_margin_from_audit_reason": reason_fields["min_margin"],
                    "k_from_audit_reason": reason_fields["k"],
                    "gamma_target_from_audit_reason": reason_fields[
                        "gamma_target"
                    ],
                    "gamma_effective_from_audit_reason": reason_fields[
                        "gamma_effective"
                    ],
                    "final_flush": event.final_flush,
                }
            )
        return rows

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


def _parse_semantic_lsh_reason(reason: str | None) -> dict[str, object]:
    parsed: dict[str, object] = {
        "lsh_signature": None,
        "in_valid_set": None,
        "min_margin": None,
        "k": None,
        "gamma_target": None,
        "gamma_effective": None,
    }
    if not reason:
        return parsed

    fields: dict[str, str] = {}
    for item in reason.split(";"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        fields[key.strip()] = value.strip()

    if "lsh_signature" in fields:
        try:
            value = literal_eval(fields["lsh_signature"])
        except (SyntaxError, ValueError):
            value = fields["lsh_signature"]
        if isinstance(value, tuple):
            parsed["lsh_signature"] = list(value)
        else:
            parsed["lsh_signature"] = value
    if "in_valid_set" in fields:
        if fields["in_valid_set"] == "True":
            parsed["in_valid_set"] = True
        elif fields["in_valid_set"] == "False":
            parsed["in_valid_set"] = False
    for key in ("min_margin", "gamma_target", "gamma_effective"):
        if key in fields:
            try:
                parsed[key] = float(fields[key])
            except ValueError:
                parsed[key] = fields[key]
    if "k" in fields:
        try:
            parsed["k"] = int(fields["k"])
        except ValueError:
            parsed["k"] = fields["k"]
    return parsed


SawrPipeline = WFCLLMGenerationPipeline
