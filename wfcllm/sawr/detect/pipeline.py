from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from wfcllm.sawr.detect.calibration import (
    ARTIFACT_TYPE,
    CalibrationArtifact,
    ContextCalibrationInput,
    SCHEMA_VERSION,
    build_calibration_artifact,
    context_score_from_null,
    empirical_upper_tail_p,
    null_for_context_from_artifact,
    percentile_threshold,
)
from wfcllm.sawr.detect.config import DETECTOR_MODE, SawrDetectionConfig
from wfcllm.sawr.detect.proxy_windows import (
    StructureContext,
    extract_structure_contexts,
)
from wfcllm.sawr.detect.scoring import SawrWindowScorer, WindowEvidence


FORBIDDEN_DETECTOR_OUTPUT_FIELDS = {
    "generation_checkpoint",
    "generation_candidate_id",
    "generation_window_id",
    "generation_layer_id",
    "audit_event_id",
    "retry_trace",
    "rollback_trace",
    "watermark_params",
    "blocks",
}
COMPATIBLE_ARTIFACT_CONFIG_FIELDS = (
    "secret_key_sha256",
    "lsh_d",
    "gamma",
    "k",
    "gamma_effective",
    "semantic_margin",
    "max_group_statements",
    "min_scoreable_contexts",
    "min_proxy_windows",
    "target_fpr",
    "use_ordinal_keying",
    "evidence_mode",
    "statistic",
    "structure_aware",
    "detector_mode",
    "bucket_edges",
)


@dataclass(frozen=True)
class ContextDetectionSummary:
    context_id: str
    structure_type: str
    parent_node_type: str
    context_raw: float
    context_score: float
    calibration_bucket_level: str
    proxy_windows: int
    direct_statements: int


@dataclass(frozen=True)
class SawrDetectionResult:
    id: str
    is_watermarked: bool
    score: float
    threshold_5fpr: float
    p_value: float
    fpr_target: float
    scoreable_contexts: int
    proxy_windows: int
    insufficient_evidence: bool
    detector_mode: str
    context_summaries: tuple[ContextDetectionSummary, ...]
    code_chars: int
    direct_statements: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["context_summaries"] = [
            asdict(summary) for summary in self.context_summaries
        ]
        forbidden_fields = FORBIDDEN_DETECTOR_OUTPUT_FIELDS & set(payload)
        if forbidden_fields:
            raise ValueError(
                "forbidden detector output fields: "
                f"{sorted(forbidden_fields)}"
            )
        try:
            json.dumps(payload, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("detector output must be JSON-safe") from exc
        return payload


@dataclass(frozen=True)
class _ScoredContext:
    context: StructureContext
    evidence: list[WindowEvidence]
    sample_proxy_windows: int


class SawrDetectionPipeline:
    """Calibrate and apply the SAWR final-code detector."""

    def __init__(
        self,
        *,
        config: SawrDetectionConfig,
        scorer: SawrWindowScorer,
    ) -> None:
        self._config = config
        self._scorer = scorer

    def calibrate(self, records: list[dict[str, Any]]) -> CalibrationArtifact:
        scored_samples = [self._score_record_contexts(record) for record in records]
        calibration_samples = [
            [
                self._context_calibration_input(scored)
                for scored in scored_contexts
            ]
            for scored_contexts in scored_samples
        ]
        artifact = build_calibration_artifact(
            calibration_samples,
            config=self._config,
        )
        sample_scores = [
            self._score_sample(scored_contexts, artifact=artifact)[0]
            for scored_contexts in scored_samples
        ]
        threshold_5fpr = (
            percentile_threshold(sample_scores, self._config.target_fpr)
            if sample_scores
            else 0.0
        )
        return replace(
            artifact,
            sample_scores=sample_scores,
            threshold_5fpr=threshold_5fpr,
        )

    def detect_one(
        self,
        record: dict[str, Any],
        *,
        artifact: CalibrationArtifact,
    ) -> SawrDetectionResult:
        self._validate_artifact_compatible(artifact)
        return self._detect_one(record, artifact=artifact)

    def _detect_one(
        self,
        record: dict[str, Any],
        *,
        artifact: CalibrationArtifact,
    ) -> SawrDetectionResult:
        code = code_from_record(record)
        scored_contexts = self._score_record_contexts(record)
        score, context_summaries = self._score_sample(
            scored_contexts,
            artifact=artifact,
        )
        proxy_windows = sum(
            len(scored.context.proxy_windows) for scored in scored_contexts
        )
        direct_statements = sum(
            len(scored.context.direct_statements) for scored in scored_contexts
        )
        insufficient_evidence = (
            len(scored_contexts) < self._config.min_scoreable_contexts
            or proxy_windows < self._config.min_proxy_windows
        )
        is_watermarked = (
            not insufficient_evidence and score >= artifact.threshold_5fpr
        )

        return SawrDetectionResult(
            id=str(record.get("id", "")),
            is_watermarked=is_watermarked,
            score=score,
            threshold_5fpr=artifact.threshold_5fpr,
            p_value=empirical_upper_tail_p(score, artifact.sample_scores),
            fpr_target=self._config.target_fpr,
            scoreable_contexts=len(scored_contexts),
            proxy_windows=proxy_windows,
            insufficient_evidence=insufficient_evidence,
            detector_mode=DETECTOR_MODE,
            context_summaries=tuple(context_summaries),
            code_chars=len(code),
            direct_statements=direct_statements,
        )

    def detect_to_jsonl(
        self,
        records: list[dict[str, Any]],
        *,
        artifact: CalibrationArtifact,
        output_path: str | Path,
    ) -> str:
        self._validate_artifact_compatible(artifact)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                payload = self._detect_one(record, artifact=artifact).to_dict()
                handle.write(
                    json.dumps(payload, allow_nan=False, ensure_ascii=False) + "\n"
                )
        return str(path)

    def _validate_artifact_compatible(
        self,
        artifact: CalibrationArtifact,
    ) -> None:
        expected_config = self._config.to_public_dict()
        mismatches: list[str] = []

        if artifact.artifact_type != ARTIFACT_TYPE:
            mismatches.append("artifact_type")
        if artifact.schema_version != SCHEMA_VERSION:
            mismatches.append("schema_version")
        if artifact.detector_mode != DETECTOR_MODE:
            mismatches.append("detector_mode")
        if artifact.bucket_edges != expected_config["bucket_edges"]:
            mismatches.append("bucket_edges")

        for field_name in COMPATIBLE_ARTIFACT_CONFIG_FIELDS:
            if artifact.config.get(field_name) != expected_config[field_name]:
                mismatches.append(field_name)

        if mismatches:
            unique_mismatches = sorted(set(mismatches))
            raise ValueError(
                "calibration artifact is incompatible with detector config: "
                f"{unique_mismatches}"
            )

    def _score_record_contexts(self, record: dict[str, Any]) -> list[_ScoredContext]:
        final_code = code_from_record(record)
        prompt = record.get("prompt")
        contexts = extract_structure_contexts(
            final_code,
            prompt=prompt if isinstance(prompt, str) else None,
            max_group_statements=self._config.max_group_statements,
        )
        sample_proxy_windows = sum(len(context.proxy_windows) for context in contexts)
        return [
            _ScoredContext(
                context=context,
                evidence=[
                    self._scorer.score_window(window)
                    for window in context.proxy_windows
                ],
                sample_proxy_windows=sample_proxy_windows,
            )
            for context in contexts
        ]

    def _context_calibration_input(
        self,
        scored: _ScoredContext,
    ) -> ContextCalibrationInput:
        raw_values = [evidence.window_raw for evidence in scored.evidence]
        return ContextCalibrationInput(
            context_id=scored.context.context_id,
            structure_type=scored.context.structure_type,
            parent_node_type=scored.context.parent_node_type,
            context_raw=max(raw_values) if raw_values else 0.0,
            context_window_count=len(scored.context.proxy_windows),
            context_statement_count=len(scored.context.direct_statements),
            window_length_mix=_window_length_mix(scored.context),
            sample_proxy_windows=scored.sample_proxy_windows,
        )

    def _score_sample(
        self,
        scored_contexts: list[_ScoredContext],
        *,
        artifact: CalibrationArtifact,
    ) -> tuple[float, list[ContextDetectionSummary]]:
        context_scores: list[float] = []
        context_summaries: list[ContextDetectionSummary] = []

        for scored in scored_contexts:
            calibration_input = self._context_calibration_input(scored)
            context_score, calibration_level = self._score_context(
                calibration_input,
                scored.evidence,
                artifact,
            )
            context_scores.append(context_score)
            context_summaries.append(
                ContextDetectionSummary(
                    context_id=scored.context.context_id,
                    structure_type=scored.context.structure_type,
                    parent_node_type=scored.context.parent_node_type,
                    context_raw=calibration_input.context_raw,
                    context_score=context_score,
                    calibration_bucket_level=calibration_level,
                    proxy_windows=len(scored.context.proxy_windows),
                    direct_statements=len(scored.context.direct_statements),
                )
            )

        sample_score = (
            sum(context_scores) / len(context_scores) if context_scores else 0.0
        )
        return sample_score, context_summaries

    def _score_context(
        self,
        calibration_input: ContextCalibrationInput,
        evidence: list[WindowEvidence],
        artifact: CalibrationArtifact,
    ) -> tuple[float, str]:
        null_values, calibration_level = null_for_context_from_artifact(
            calibration_input,
            config=self._config,
            artifact=artifact,
        )
        if self._config.statistic == "raw_context_max":
            return calibration_input.context_raw, calibration_level
        if self._config.statistic == "context_mean_window_evidence":
            if not evidence:
                return 0.0, calibration_level
            return (
                sum(item.window_raw for item in evidence) / len(evidence),
                calibration_level,
            )
        return (
            context_score_from_null(calibration_input.context_raw, null_values),
            calibration_level,
        )


def _window_length_mix(context: StructureContext) -> str:
    counts: dict[int, int] = {}
    for window in context.proxy_windows:
        counts[window.window_length] = counts.get(window.window_length, 0) + 1
    return "|".join(
        f"{window_length}:{counts[window_length]}"
        for window_length in sorted(counts)
    )


def code_from_record(record: dict[str, Any]) -> str:
    if "final_code" in record:
        final_code = record["final_code"]
        if not isinstance(final_code, str):
            raise ValueError("final_code must be a string")
        return final_code

    if "generated_code" in record:
        generated_code = record["generated_code"]
        if not isinstance(generated_code, str):
            raise ValueError("generated_code must be a string")
        return generated_code

    raise ValueError("record must contain final_code or generated_code")


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    records_path = Path(path)
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        records_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid JSONL record {records_path}:{line_number}"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(
                f"JSONL record must be an object {records_path}:{line_number}"
            )
        records.append(record)
    return records
