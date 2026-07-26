"""Final-code-only calibration and detection for formal gated windows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from wfcllm.detection.code_only import (
    reject_forbidden_detector_output_fields,
    validate_final_code_record_exact,
)
from wfcllm.detection.config import GATED_DETECTOR_MODE
from wfcllm.detection.pipeline import load_jsonl_records
from wfcllm.detection.scoring import GatedSampleScore, GatedWindowScorer
from wfcllm.gate.input import GATE_INPUT_CONTRACT_VERSION
from wfcllm.windowing.contracts import (
    WINDOW_CONTRACT_VERSION,
    is_supported_window_contract,
)

GATED_METHOD_NAME = "gated_semantic_window_v1"
GATED_CALIBRATION_SCHEMA_VERSION = "wfcllm-gated-calibration/v1"
EMPIRICAL_P_VALUE_RULE = "right_tail_plus_one/v1"
BINOMIAL_P_VALUE_RULE = "pooled_negative_binomial_right_tail/v1"
EMPIRICAL_BINOMIAL_SURPRISAL_RULE = (
    "pooled_negative_empirical_binomial_surprisal/v1"
)
EMPIRICAL_STANDARDIZED_HIT_SURPLUS_RULE = (
    "pooled_negative_empirical_standardized_hit_surplus/v1"
)
_INSUFFICIENT_STATISTIC_FLOOR = -1e300
QUANTILE_THRESHOLD_RULE = "pooled_negative_quantile_threshold/v1"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class GatedDetectionBindings:
    gate_bundle_sha256: str
    semantic_encoder_sha256: str
    lsh_config_sha256: str
    key_identifier_sha256: str
    negative_corpus_manifest_sha256: str
    window_contract_version: str = WINDOW_CONTRACT_VERSION
    gate_input_contract_version: str = GATE_INPUT_CONTRACT_VERSION
    detector_mode: str = GATED_DETECTOR_MODE

    def __post_init__(self) -> None:
        for name in (
            "gate_bundle_sha256",
            "semantic_encoder_sha256",
            "lsh_config_sha256",
            "key_identifier_sha256",
            "negative_corpus_manifest_sha256",
        ):
            if _DIGEST.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if not is_supported_window_contract(self.window_contract_version):
            raise ValueError("window contract mismatch")
        if self.gate_input_contract_version != GATE_INPUT_CONTRACT_VERSION:
            raise ValueError("gate input contract mismatch")
        if self.detector_mode != GATED_DETECTOR_MODE:
            raise ValueError("detector mode mismatch")


@dataclass(frozen=True)
class GatedCalibrationArtifact:
    schema_version: str
    method_name: str
    detector_mode: str
    window_contract_version: str
    gate_input_contract_version: str
    gate_bundle_sha256: str
    semantic_encoder_sha256: str
    lsh_config_sha256: str
    key_identifier_sha256: str
    negative_corpus_manifest_sha256: str
    empirical_p_value_rule: str
    target_fpr: float
    minimum_reliable_windows: int
    null_hit_probability: float | None
    reliable_window_count_buckets: dict[str, tuple[float, ...]]
    thresholds_by_reliable_window_count: dict[str, float]

    def __post_init__(self) -> None:
        if self.schema_version != GATED_CALIBRATION_SCHEMA_VERSION:
            raise ValueError("gated calibration schema mismatch")
        if self.method_name != GATED_METHOD_NAME:
            raise ValueError("gated calibration method mismatch")
        GatedDetectionBindings(
            gate_bundle_sha256=self.gate_bundle_sha256,
            semantic_encoder_sha256=self.semantic_encoder_sha256,
            lsh_config_sha256=self.lsh_config_sha256,
            key_identifier_sha256=self.key_identifier_sha256,
            negative_corpus_manifest_sha256=self.negative_corpus_manifest_sha256,
            window_contract_version=self.window_contract_version,
            gate_input_contract_version=self.gate_input_contract_version,
            detector_mode=self.detector_mode,
        )
        if self.empirical_p_value_rule not in {
            EMPIRICAL_P_VALUE_RULE,
            BINOMIAL_P_VALUE_RULE,
            EMPIRICAL_BINOMIAL_SURPRISAL_RULE,
            EMPIRICAL_STANDARDIZED_HIT_SURPLUS_RULE,
            QUANTILE_THRESHOLD_RULE,
        }:
            raise ValueError("empirical p-value rule mismatch")
        if not isinstance(self.target_fpr, (int, float)) or not 0 < self.target_fpr < 1:
            raise ValueError("target_fpr must be in (0, 1)")
        if type(self.minimum_reliable_windows) is not int or self.minimum_reliable_windows < 1:
            raise ValueError("minimum_reliable_windows must be positive")
        if set(self.reliable_window_count_buckets) != set(
            self.thresholds_by_reliable_window_count
        ):
            raise ValueError("calibration buckets and thresholds must match")
        if not self.reliable_window_count_buckets:
            raise ValueError("calibration requires reliable-window buckets")
        count_statistic_rule = self.empirical_p_value_rule in {
            EMPIRICAL_BINOMIAL_SURPRISAL_RULE,
            EMPIRICAL_STANDARDIZED_HIT_SURPLUS_RULE,
        }
        surprisal_rule = (
            self.empirical_p_value_rule
            == EMPIRICAL_BINOMIAL_SURPRISAL_RULE
        )
        if count_statistic_rule:
            if (
                isinstance(self.null_hit_probability, bool)
                or not isinstance(self.null_hit_probability, (int, float))
                or not 0.0 < self.null_hit_probability < 1.0
            ):
                raise ValueError(
                    "count-statistic calibration requires null_hit_probability"
                )
        elif self.null_hit_probability is not None:
            raise ValueError(
                "null_hit_probability is reserved for count-statistic calibration"
            )
        for label, values in self.reliable_window_count_buckets.items():
            if not label.isdigit() or int(label) < self.minimum_reliable_windows:
                raise ValueError("invalid reliable-window bucket")
            if not values or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or (surprisal_rule and float(value) < 0.0)
                or (
                    not count_statistic_rule
                    and not 0.0 <= float(value) <= 1.0
                )
                for value in values
            ):
                raise ValueError("invalid background statistic distribution")
            threshold = self.thresholds_by_reliable_window_count[label]
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))
                or not math.isfinite(float(threshold))
                or (surprisal_rule and float(threshold) < 0.0)
                or (
                    not count_statistic_rule
                    and not 0.0 <= float(threshold) <= 1.0
                )
            ):
                raise ValueError("invalid empirical threshold")

    @classmethod
    def from_dict(cls, value: object) -> GatedCalibrationArtifact:
        fields = set(cls.__dataclass_fields__)
        if not isinstance(value, Mapping):
            raise ValueError("gated calibration artifact schema mismatch")
        payload = dict(value)
        # Pre-multichannel artifacts (nocarrier/cpp runs) predate this field;
        # absence means the non-count-statistic rules, i.e. None.
        if "null_hit_probability" not in payload:
            payload["null_hit_probability"] = None
        if set(payload) != fields:
            raise ValueError("gated calibration artifact schema mismatch")
        buckets = payload.get("reliable_window_count_buckets")
        if not isinstance(buckets, Mapping):
            raise ValueError("calibration buckets must be an object")
        payload["reliable_window_count_buckets"] = {
            str(key): tuple(values) for key, values in buckets.items()
        }
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reliable_window_count_buckets"] = {
            key: list(values)
            for key, values in self.reliable_window_count_buckets.items()
        }
        return payload


@dataclass(frozen=True)
class GatedDetectionResult:
    id: str
    method_name: str
    detector_mode: str
    decision: str
    is_watermarked: bool
    insufficient_evidence: bool
    hit_count: int
    miss_count: int
    abstain_count: int
    reliable_window_count: int
    hit_rate: float
    threshold_at_target_fpr: float | None
    p_value: float | None
    fpr_target: float
    windows: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["windows"] = [dict(item) for item in self.windows]
        reject_forbidden_detector_output_fields(payload)
        json.dumps(payload, allow_nan=False)
        return payload


class GatedDetectionPipeline:
    """Use the same final-code extractor and semantic scorer for both populations."""

    def __init__(
        self,
        *,
        extractor: object,
        scorer: GatedWindowScorer,
        bindings: GatedDetectionBindings,
        target_fpr: float = 0.05,
        calibration_group_by: str = "reliable_window_count",
    ) -> None:
        if not callable(getattr(extractor, "extract", None)):
            raise ValueError("extractor must expose final-code extract")
        if not isinstance(scorer, GatedWindowScorer):
            raise ValueError("scorer must be GatedWindowScorer")
        if not isinstance(bindings, GatedDetectionBindings):
            raise ValueError("bindings must be GatedDetectionBindings")
        if (
            getattr(extractor, "window_contract_version", None)
            != bindings.window_contract_version
        ):
            raise ValueError("extractor and detection bindings window contracts differ")
        if isinstance(target_fpr, bool) or not isinstance(target_fpr, (int, float)) or not 0 < target_fpr < 1:
            raise ValueError("target_fpr must be in (0, 1)")
        if calibration_group_by not in {
            "reliable_window_count",
            "pooled_reliable_hit_rate",
            "pooled_reliable_hit_rate_quantile",
            "pooled_binomial_tail",
            "pooled_empirical_binomial_surprisal",
            "pooled_empirical_standardized_hit_surplus",
        }:
            raise ValueError("unsupported gated calibration grouping")
        self._extractor = extractor
        self._scorer = scorer
        self._bindings = bindings
        self._target_fpr = float(target_fpr)
        self._calibration_group_by = calibration_group_by

    def calibrate(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        output_path: str | Path | None = None,
    ) -> GatedCalibrationArtifact:
        scores: list[GatedSampleScore] = []
        for record in records:
            validate_final_code_record_exact(record)
            scores.append(self._score_record(record))
        buckets: dict[str, list[float]] = {}
        null_hit_probability: float | None = None
        count_statistic_group = self._calibration_group_by in {
            "pooled_empirical_binomial_surprisal",
            "pooled_empirical_standardized_hit_surplus",
        }
        if count_statistic_group:
            null_hits = sum(score.hit_count for score in scores)
            null_trials = sum(score.reliable_window_count for score in scores)
            if null_trials <= 0:
                raise ValueError(
                    "calibration has no reliable windows for null estimate"
                )
            null_hit_probability = (null_hits + 1.0) / (
                null_trials + 2.0
            )
            bucket = str(self._scorer.minimum_reliable_windows)
            buckets[bucket] = [
                (
                    (
                        _binomial_surprisal(
                            score.hit_count,
                            score.reliable_window_count,
                            null_hit_probability,
                        )
                        if self._calibration_group_by
                        == "pooled_empirical_binomial_surprisal"
                        else _standardized_hit_surplus(
                            score.hit_count,
                            score.reliable_window_count,
                            null_hit_probability,
                        )
                    )
                    if score.reliable_window_count
                    >= self._scorer.minimum_reliable_windows
                    else _INSUFFICIENT_STATISTIC_FLOOR
                )
                for score in scores
            ]
        for score in (
            ()
            if count_statistic_group
            else scores
        ):
            if score.reliable_window_count < self._scorer.minimum_reliable_windows:
                if self._calibration_group_by in {
                    "pooled_reliable_hit_rate",
                    "pooled_reliable_hit_rate_quantile",
                }:
                    bucket = str(self._scorer.minimum_reliable_windows)
                    buckets.setdefault(bucket, []).append(0.0)
                continue
            if self._calibration_group_by == "pooled_binomial_tail":
                bucket = str(self._scorer.minimum_reliable_windows)
                buckets.setdefault(bucket, []).extend(
                    [1.0] * score.hit_count + [0.0] * score.miss_count
                )
            else:
                bucket = (
                    str(self._scorer.minimum_reliable_windows)
                    if self._calibration_group_by
                    in {
                        "pooled_reliable_hit_rate",
                        "pooled_reliable_hit_rate_quantile",
                    }
                    else str(score.reliable_window_count)
                )
                buckets.setdefault(bucket, []).append(score.hit_rate)
        if not buckets:
            raise ValueError("calibration has no samples with enough reliable windows")
        frozen_buckets = {key: tuple(values) for key, values in sorted(buckets.items(), key=lambda x: int(x[0]))}
        artifact = GatedCalibrationArtifact(
            schema_version=GATED_CALIBRATION_SCHEMA_VERSION,
            method_name=GATED_METHOD_NAME,
            detector_mode=self._bindings.detector_mode,
            window_contract_version=self._bindings.window_contract_version,
            gate_input_contract_version=self._bindings.gate_input_contract_version,
            gate_bundle_sha256=self._bindings.gate_bundle_sha256,
            semantic_encoder_sha256=self._bindings.semantic_encoder_sha256,
            lsh_config_sha256=self._bindings.lsh_config_sha256,
            key_identifier_sha256=self._bindings.key_identifier_sha256,
            negative_corpus_manifest_sha256=self._bindings.negative_corpus_manifest_sha256,
            empirical_p_value_rule=(
                BINOMIAL_P_VALUE_RULE
                if self._calibration_group_by == "pooled_binomial_tail"
                else EMPIRICAL_BINOMIAL_SURPRISAL_RULE
                if self._calibration_group_by
                == "pooled_empirical_binomial_surprisal"
                else EMPIRICAL_STANDARDIZED_HIT_SURPLUS_RULE
                if self._calibration_group_by
                == "pooled_empirical_standardized_hit_surplus"
                else QUANTILE_THRESHOLD_RULE
                if self._calibration_group_by
                == "pooled_reliable_hit_rate_quantile"
                else EMPIRICAL_P_VALUE_RULE
            ),
            target_fpr=self._target_fpr,
            minimum_reliable_windows=self._scorer.minimum_reliable_windows,
            null_hit_probability=null_hit_probability,
            reliable_window_count_buckets=frozen_buckets,
            thresholds_by_reliable_window_count=(
                {key: self._target_fpr for key in frozen_buckets}
                if self._calibration_group_by == "pooled_binomial_tail"
                else (
                    {
                        key: _inclusive_empirical_threshold(
                            values,
                            self._target_fpr,
                        )
                        for key, values in frozen_buckets.items()
                    }
                    if self._calibration_group_by
                    == "pooled_reliable_hit_rate_quantile"
                    else {
                        key: _empirical_threshold(values, self._target_fpr)
                        for key, values in frozen_buckets.items()
                    }
                )
            ),
        )
        if output_path is not None:
            write_gated_calibration_artifact(output_path, artifact)
        return artifact

    def calibrate_jsonl(
        self, path: str | Path, *, output_path: str | Path | None = None
    ) -> GatedCalibrationArtifact:
        records = load_jsonl_records(path)
        return self.calibrate(records, output_path=output_path)

    def detect(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        artifact: GatedCalibrationArtifact,
        output_path: str | Path | None = None,
    ) -> list[GatedDetectionResult]:
        self._validate_artifact(artifact)
        results = [self._detect_one(record, artifact) for record in records]
        if output_path is not None:
            _write_details(output_path, results)
        return results

    def detect_jsonl(
        self,
        path: str | Path,
        *,
        artifact: GatedCalibrationArtifact,
        output_path: str | Path | None = None,
    ) -> list[GatedDetectionResult]:
        records = load_jsonl_records(path)
        return self.detect(records, artifact=artifact, output_path=output_path)

    def _score_record(self, record: Mapping[str, Any]) -> GatedSampleScore:
        validate_final_code_record_exact(record)
        extraction = self._extractor.extract(record["final_code"])
        windows = getattr(extraction, "windows", None)
        if not isinstance(windows, (tuple, list)):
            raise ValueError("gated extractor must return windows")
        return self._scorer.score(windows)

    def _detect_one(
        self, record: Mapping[str, Any], artifact: GatedCalibrationArtifact
    ) -> GatedDetectionResult:
        validate_final_code_record_exact(record)
        score = self._score_record(record)
        insufficient = score.reliable_window_count < artifact.minimum_reliable_windows
        if insufficient:
            threshold = p_value = None
            decision = "insufficient_evidence"
        else:
            background, threshold = _select_bucket(artifact, score.reliable_window_count)
            if artifact.empirical_p_value_rule == BINOMIAL_P_VALUE_RULE:
                null_hit_probability = (sum(background) + 1.0) / (
                    len(background) + 2.0
                )
                p_value = binomial_right_tail(
                    score.hit_count,
                    score.reliable_window_count,
                    null_hit_probability,
                )
                threshold = _binomial_hit_rate_threshold(
                    score.reliable_window_count,
                    null_hit_probability,
                    artifact.target_fpr,
                )
                observed_statistic = score.hit_rate
            elif (
                artifact.empirical_p_value_rule
                == EMPIRICAL_BINOMIAL_SURPRISAL_RULE
            ):
                assert artifact.null_hit_probability is not None
                observed_statistic = _binomial_surprisal(
                    score.hit_count,
                    score.reliable_window_count,
                    artifact.null_hit_probability,
                )
                p_value = empirical_right_tail_plus_one(
                    observed_statistic, background
                )
            elif (
                artifact.empirical_p_value_rule
                == EMPIRICAL_STANDARDIZED_HIT_SURPLUS_RULE
            ):
                assert artifact.null_hit_probability is not None
                observed_statistic = _standardized_hit_surplus(
                    score.hit_count,
                    score.reliable_window_count,
                    artifact.null_hit_probability,
                )
                p_value = empirical_right_tail_plus_one(
                    observed_statistic, background
                )
            else:
                observed_statistic = score.hit_rate
                p_value = empirical_right_tail_plus_one(score.hit_rate, background)
            if artifact.empirical_p_value_rule == QUANTILE_THRESHOLD_RULE:
                decision = (
                    "watermarked"
                    if score.hit_rate >= threshold
                    else "not_watermarked"
                )
            else:
                decision = (
                    "watermarked"
                    if p_value <= artifact.target_fpr
                    and observed_statistic >= threshold
                    else "not_watermarked"
                )
        return GatedDetectionResult(
            id=str(record["id"]),
            method_name=GATED_METHOD_NAME,
            detector_mode=GATED_DETECTOR_MODE,
            decision=decision,
            is_watermarked=decision == "watermarked",
            insufficient_evidence=insufficient,
            hit_count=score.hit_count,
            miss_count=score.miss_count,
            abstain_count=score.abstain_count,
            reliable_window_count=score.reliable_window_count,
            hit_rate=score.hit_rate,
            threshold_at_target_fpr=threshold,
            p_value=p_value,
            fpr_target=artifact.target_fpr,
            windows=tuple(asdict(item) for item in score.evidence),
        )

    def _validate_artifact(self, artifact: GatedCalibrationArtifact) -> None:
        if not isinstance(artifact, GatedCalibrationArtifact):
            raise ValueError("gated calibration artifact is required")
        expected = asdict(self._bindings)
        mismatches = [
            name for name, value in expected.items() if getattr(artifact, name) != value
        ]
        if artifact.target_fpr != self._target_fpr:
            mismatches.append("target_fpr")
        if artifact.minimum_reliable_windows != self._scorer.minimum_reliable_windows:
            mismatches.append("minimum_reliable_windows")
        if mismatches:
            raise ValueError(
                "gated calibration artifact hash/config mismatch: "
                + ", ".join(sorted(mismatches))
            )


def empirical_right_tail_plus_one(value: float, background: Sequence[float]) -> float:
    if not background:
        raise ValueError("empirical background must not be empty")
    return (1 + sum(item >= value for item in background)) / (len(background) + 1)


def binomial_right_tail(hits: int, trials: int, hit_probability: float) -> float:
    if type(hits) is not int or type(trials) is not int or not 0 <= hits <= trials:
        raise ValueError("binomial counts are invalid")
    if not 0.0 < hit_probability < 1.0:
        raise ValueError("binomial hit probability must be in (0, 1)")
    return sum(
        math.comb(trials, count)
        * hit_probability**count
        * (1.0 - hit_probability) ** (trials - count)
        for count in range(hits, trials + 1)
    )


def _binomial_surprisal(
    hits: int, trials: int, hit_probability: float
) -> float:
    tail = binomial_right_tail(hits, trials, hit_probability)
    return -math.log10(max(tail, 1e-300))


def _standardized_hit_surplus(
    hits: int, trials: int, hit_probability: float
) -> float:
    if type(hits) is not int or type(trials) is not int or not 0 <= hits <= trials:
        raise ValueError("binomial counts are invalid")
    if trials <= 0:
        raise ValueError("standardized hit surplus requires positive trials")
    if not 0.0 < hit_probability < 1.0:
        raise ValueError("binomial hit probability must be in (0, 1)")
    return (hits - hit_probability * trials) / math.sqrt(trials)


def _binomial_hit_rate_threshold(
    trials: int, hit_probability: float, target_fpr: float
) -> float:
    for hits in range(trials + 1):
        if binomial_right_tail(hits, trials, hit_probability) <= target_fpr:
            return hits / trials
    return 1.0


def _empirical_threshold(values: Sequence[float], target_fpr: float) -> float:
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil((1.0 - target_fpr) * len(ordered)) - 1)
    return ordered[index]


def _inclusive_empirical_threshold(
    values: Sequence[float],
    target_fpr: float,
) -> float:
    ordered = tuple(sorted(float(value) for value in values))
    allowed_false_positives = math.floor(target_fpr * len(ordered))
    if allowed_false_positives < 1:
        raise ValueError(
            "quantile calibration requires at least ceil(1 / target_fpr) negatives"
        )
    for threshold in sorted(set(ordered)):
        if sum(value >= threshold for value in ordered) <= allowed_false_positives:
            return threshold
    raise ValueError(
        "negative score ties prevent an inclusive threshold at the target FPR"
    )


def _select_bucket(
    artifact: GatedCalibrationArtifact, reliable_count: int
) -> tuple[tuple[float, ...], float]:
    labels = sorted((int(key), key) for key in artifact.reliable_window_count_buckets)
    _, label = min(labels, key=lambda item: (abs(item[0] - reliable_count), item[0]))
    return (
        artifact.reliable_window_count_buckets[label],
        artifact.thresholds_by_reliable_window_count[label],
    )


def write_gated_calibration_artifact(
    path: str | Path, artifact: GatedCalibrationArtifact
) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact.to_dict(), allow_nan=False, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return str(output)


def load_gated_calibration_artifact(path: str | Path) -> GatedCalibrationArtifact:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid gated calibration artifact") from exc
    return GatedCalibrationArtifact.from_dict(value)


def _write_details(path: str | Path, results: Sequence[GatedDetectionResult]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.to_dict(), allow_nan=False, ensure_ascii=False) + "\n")
    return str(output)


def hash_negative_corpus_manifest(records: Sequence[Mapping[str, Any]]) -> str:
    for record in records:
        validate_final_code_record_exact(record)
    canonical = json.dumps(
        list(records), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def calibrate_gated_detector(
    records: Sequence[Mapping[str, Any]],
    *,
    pipeline: GatedDetectionPipeline,
    output_dir: str | Path | None = None,
) -> GatedCalibrationArtifact:
    """Convenience entry point used by artifact-producing integrations."""

    if not isinstance(pipeline, GatedDetectionPipeline):
        raise ValueError("pipeline must be a GatedDetectionPipeline")
    output_path = (
        Path(output_dir) / "gated_calibration.json"
        if output_dir is not None
        else None
    )
    return pipeline.calibrate(records, output_path=output_path)


__all__ = [
    "BINOMIAL_P_VALUE_RULE",
    "EMPIRICAL_BINOMIAL_SURPRISAL_RULE",
    "EMPIRICAL_STANDARDIZED_HIT_SURPLUS_RULE",
    "EMPIRICAL_P_VALUE_RULE",
    "QUANTILE_THRESHOLD_RULE",
    "GATED_CALIBRATION_SCHEMA_VERSION",
    "GATED_METHOD_NAME",
    "GatedCalibrationArtifact",
    "GatedDetectionBindings",
    "GatedDetectionPipeline",
    "GatedDetectionResult",
    "calibrate_gated_detector",
    "binomial_right_tail",
    "empirical_right_tail_plus_one",
    "hash_negative_corpus_manifest",
    "load_gated_calibration_artifact",
    "write_gated_calibration_artifact",
]
