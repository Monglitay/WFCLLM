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

from wfcllm.extract.detector import WatermarkDetector


@dataclass
class ExtractPipelineConfig:
    """Configuration for batch watermark extraction pipeline."""

    input_file: str     # Path to input JSONL produced by WatermarkPipeline
    output_dir: str     # Directory for report JSON output


class ExtractPipeline:
    """Batch watermark detection and statistical reporting."""

    def __init__(self, detector: WatermarkDetector, config: ExtractPipelineConfig):
        self._detector = detector
        self._config = config

    def run(self) -> str:
        """Run batch extraction. Returns path to output report JSON."""
        records = self._load_jsonl()
        total = len(records)

        iterator = (
            tqdm(records, desc="Extracting", unit="sample")
            if tqdm is not None
            else records
        )

        per_sample = []
        z_scores = []
        p_values = []
        block_counts = []
        embed_rates = []

        for item in iterator:
            result = self._detector.detect(item["generated_code"])
            per_sample.append({
                "id": item["id"],
                "is_watermarked": result.is_watermarked,
                "z_score": result.z_score,
                "p_value": result.p_value,
                "independent_blocks": result.independent_blocks,
                "hits": result.hit_blocks,
            })
            z_scores.append(result.z_score)
            p_values.append(result.p_value)
            block_counts.append(result.independent_blocks)
            embed_rates.append(item.get("embed_rate", 0.0))

            summary_line = (
                f"  ✓ {item['id']} | "
                f"z={result.z_score:.2f} | "
                f"blocks={result.independent_blocks} | "
                f"watermarked={result.is_watermarked}"
            )
            print(summary_line, file=sys.stderr)

        watermarked = sum(1 for s in per_sample if s["is_watermarked"])
        watermark_rate = watermarked / total if total > 0 else 0.0

        report = {
            "meta": {
                "input_file": self._config.input_file,
                "total_samples": total,
            },
            "summary": {
                "watermark_rate": watermark_rate,
                "watermark_rate_ci_95": self._proportion_ci(watermarked, total),
                "mean_z_score": _mean(z_scores),
                "std_z_score": _std(z_scores),
                "mean_p_value": _mean(p_values),
                "mean_blocks": _mean(block_counts),
                "embed_rate_distribution": _distribution_stats(embed_rates),
            },
            "per_sample": per_sample,
        }

        out_dir = Path(self._config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(self._config.input_file).stem
        out_path = out_dir / f"{stem}_report.json"
        out_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return str(out_path)

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
