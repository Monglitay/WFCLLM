"""Batch watermark extraction pipeline over a JSONL watermark dataset."""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]

try:
    from statsmodels.stats.proportion import proportion_confint as _proportion_confint
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False

from wfcllm.common.checkpoint import load_processed_ids, resolve_resume_path
from wfcllm.extract.detector import WatermarkDetector


@dataclass
class ExtractPipelineConfig:
    """Configuration for batch watermark extraction pipeline."""

    input_file: str     # Path to input JSONL produced by WatermarkPipeline
    output_dir: str     # Directory for report JSON output
    resume: str | None = None
    summary_metadata: dict | None = None


class ExtractPipeline:
    """Batch watermark detection and statistical reporting."""

    def __init__(self, detector: WatermarkDetector, config: ExtractPipelineConfig):
        self._detector = detector
        self._config = config

    @staticmethod
    def summary_path_for_details(details_path: Path) -> Path:
        base_stem = details_path.stem
        if base_stem.endswith("_details"):
            base_stem = base_stem[: -len("_details")]
        return details_path.parent / f"{base_stem}_summary.json"

    @staticmethod
    def _build_embed_rate_map(records: list[dict]) -> dict[str, float]:
        return {record["id"]: record.get("embed_rate", 0.0) for record in records}

    def _validate_resume_path(self, resume_path: Path) -> None:
        expected_name = f"{Path(self._config.input_file).stem}_details.jsonl"
        if resume_path.name != expected_name:
            raise ValueError(
                f"Resume file {resume_path.name} does not match input file {self._config.input_file}"
            )

    def _generate_summary(
        self,
        details_path: Path,
        embed_rate_by_id: dict[str, float],
    ) -> Path:
        rows: list[dict] = []
        with open(details_path, encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if line:
                    rows.append(json.loads(line))

        total = len(rows)
        invalid_rows = [row for row in rows if row.get("contract_valid") is False]
        scored_rows = rows
        if self._exclude_invalid_samples():
            scored_rows = [row for row in rows if row.get("contract_valid") is not False]

        watermarked = sum(1 for row in scored_rows if row.get("semantic_prediction", row["is_watermarked"]))
        z_scores = [row["z_score"] for row in scored_rows]
        p_values = [row["p_value"] for row in scored_rows]
        block_counts = [row["independent_blocks"] for row in scored_rows]
        embed_rates = [embed_rate_by_id.get(row["id"], 0.0) for row in scored_rows]
        lexical_z_scores = [row["lexical_z_score"] for row in scored_rows if "lexical_z_score" in row]
        green_fractions = [row["green_fraction"] for row in scored_rows if "green_fraction" in row]
        joint_scores = [row["joint_score"] for row in scored_rows if "joint_score" in row]
        joint_predictions = [row["joint_prediction"] for row in scored_rows if "joint_prediction" in row]

        summary = {
            "meta": {
                "input_file": self._config.input_file,
                "total_samples": total,
                "scored_samples": len(scored_rows),
                "invalid_samples": len(invalid_rows),
                **(self._config.summary_metadata or {}),
            },
            "summary": {
                "watermark_rate": watermarked / len(scored_rows) if scored_rows else 0.0,
                "watermark_rate_ci_95": self._proportion_ci(watermarked, len(scored_rows)),
                "mean_z_score": _mean(z_scores),
                "std_z_score": _std(z_scores),
                "mean_p_value": _mean(p_values),
                "mean_blocks": _mean(block_counts),
                "embed_rate_distribution": _distribution_stats(embed_rates),
                "mean_lexical_z_score": _mean(lexical_z_scores),
                "mean_green_fraction": _mean(green_fractions),
                "mean_joint_score": _mean(joint_scores),
                "joint_prediction_rate": (
                    sum(1 for prediction in joint_predictions if prediction) / len(joint_predictions)
                    if joint_predictions
                    else 0.0
                ),
                "mode_counts": self._mode_counts(rows),
                "invalid_reason_counts": self._invalid_reason_counts(rows),
            },
        }

        summary_path = self.summary_path_for_details(details_path)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary_path

    def _exclude_invalid_samples(self) -> bool:
        detector_config = getattr(self._detector, "_config", None)
        adaptive_config = getattr(detector_config, "adaptive_detection", None)
        return bool(getattr(adaptive_config, "exclude_invalid_samples", False))

    @staticmethod
    def _mode_counts(rows: list[dict]) -> dict[str, int]:
        return {
            "fixed": sum(1 for row in rows if row.get("mode", "fixed") == "fixed"),
            "adaptive": sum(1 for row in rows if row.get("mode") == "adaptive"),
        }

    @staticmethod
    def _invalid_reason_counts(rows: list[dict]) -> dict[str, int]:
        alignment_failed = 0
        adaptive_contract_invalid = 0

        for row in rows:
            alignment = row.get("contract_alignment") or {}
            if alignment.get("structure_mismatch"):
                alignment_failed += 1
                continue
            if (
                row.get("mode") == "adaptive"
                and row.get("contract_valid") is False
                and alignment.get("numeric_mismatch")
            ):
                adaptive_contract_invalid += 1

        return {
            "alignment_failed": alignment_failed,
            "adaptive_contract_invalid": adaptive_contract_invalid,
        }

    def run(self) -> str:
        """Run batch extraction. Returns path to output details JSONL."""
        input_stem = Path(self._config.input_file).stem
        out_dir = Path(self._config.output_dir)
        resume_path, is_resume = resolve_resume_path(
            self._config.resume,
            out_dir,
            default_pattern=f"{input_stem}_details.jsonl",
        )

        all_records = self._load_jsonl()
        processed_ids: set[str] = set()
        if is_resume and resume_path is not None:
            self._validate_resume_path(resume_path)
            processed_ids = load_processed_ids(resume_path)

        records = [item for item in all_records if item["id"] not in processed_ids]

        iterator = (
            tqdm(records, desc="Extracting", unit="sample")
            if tqdm is not None
            else records
        )

        embed_rate_by_id = self._build_embed_rate_map(all_records)
        out_dir.mkdir(parents=True, exist_ok=True)
        details_path = (
            resume_path
            if is_resume and resume_path is not None
            else out_dir / f"{input_stem}_details.jsonl"
        )
        mode = "a" if is_resume and resume_path is not None else "w"

        if not records:
            self._generate_summary(details_path, embed_rate_by_id)
            return str(details_path)

        with open(details_path, mode, encoding="utf-8") as f:
            for item in iterator:
                if "blocks" in item:
                    result = self._detector.detect(
                        item["generated_code"],
                        watermark_metadata=item,
                    )
                else:
                    result = self._detector.detect(item["generated_code"])
                row = {
                    "id": item["id"],
                    "mode": result.mode,
                    "is_watermarked": result.is_watermarked,
                    "semantic_prediction": result.is_watermarked,
                    "z_score": result.z_score,
                    "p_value": result.p_value,
                    "independent_blocks": result.independent_blocks,
                    "hits": result.hit_blocks,
                }
                if result.contract_valid is not None:
                    row["contract_valid"] = result.contract_valid
                if result.alignment_report is not None:
                    row["alignment_ok"] = result.alignment_ok
                    row["contract_alignment"] = result.alignment_report.to_dict()
                lexical_result = getattr(result, "lexical_result", None)
                if lexical_result is not None:
                    row["num_positions_scored"] = lexical_result.num_positions_scored
                    row["num_green_hits"] = lexical_result.num_green_hits
                    row["green_fraction"] = lexical_result.green_fraction
                    row["lexical_z_score"] = lexical_result.lexical_z_score
                    row["lexical_p_value"] = lexical_result.lexical_p_value
                joint_result = getattr(result, "joint_result", None)
                if joint_result is not None:
                    row["joint_score"] = joint_result.joint_score
                    row["p_joint"] = joint_result.p_joint
                    row["joint_prediction"] = joint_result.prediction
                    row["confidence"] = joint_result.confidence
                    row["rationale"] = joint_result.rationale
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()

                summary_line = (
                    f"  ✓ {item['id']} | "
                    f"z={result.z_score:.2f} | "
                    f"blocks={result.independent_blocks} | "
                    f"watermarked={result.is_watermarked}"
                )
                print(summary_line, file=sys.stderr)

        self._generate_summary(details_path, embed_rate_by_id)
        return str(details_path)

    def _load_jsonl(self) -> list[dict]:
        path = Path(self._config.input_file)
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    @staticmethod
    def _proportion_ci(k: int, n: int, confidence: float = 0.95) -> list[float]:
        """Wilson score confidence interval for a proportion."""
        if n == 0:
            return [0.0, 0.0]
        if _HAS_STATSMODELS:
            lo, hi = _proportion_confint(k, n, alpha=1 - confidence, method="wilson")
            return [round(float(lo), 6), round(float(hi), 6)]
        # Fallback: Wilson score interval (no external deps)
        p = k / n
        z = 1.96  # 95% CI
        denom = 1 + z ** 2 / n
        centre = (p + z ** 2 / (2 * n)) / denom
        margin = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
        return [round(max(0.0, centre - margin), 6), round(min(1.0, centre + margin), 6)]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def _distribution_stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "std": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0}
    sorted_v = sorted(values)
    n = len(sorted_v)

    def percentile(p: float) -> float:
        idx = p * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        return sorted_v[lo] + (sorted_v[hi] - sorted_v[lo]) * (idx - lo)

    return {
        "mean": round(_mean(values), 6),
        "std": round(_std(values), 6),
        "p25": round(percentile(0.25), 6),
        "p50": round(percentile(0.50), 6),
        "p75": round(percentile(0.75), 6),
    }
