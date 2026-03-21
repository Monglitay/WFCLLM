"""Batch watermarking pipeline over HumanEval/MBPP datasets."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]

import torch

from wfcllm.common.checkpoint import load_processed_ids, resolve_resume_path
from wfcllm.watermark.generator import WatermarkGenerator
from wfcllm.common.dataset_loader import SUPPORTED_DATASETS, load_prompts


@dataclass
class WatermarkPipelineConfig:
    """Configuration for batch watermark embedding pipeline."""

    dataset: str            # "humaneval" or "mbpp"
    output_dir: str         # e.g. "data/watermarked"
    dataset_path: str       # local datasets root, e.g. "data/datasets"
    resume: str | None = None
    sample_limit: int | None = None

    def __post_init__(self):
        if self.dataset not in SUPPORTED_DATASETS:
            raise ValueError(
                f"dataset must be one of {SUPPORTED_DATASETS}, got '{self.dataset}'"
            )


class WatermarkPipeline:
    """Batch watermark embedding over a HumanEval or MBPP dataset."""

    def __init__(self, generator: WatermarkGenerator, config: WatermarkPipelineConfig):
        self._generator = generator
        self._config = config

    def _load_prompts(self) -> list[dict]:
        """Load prompts from local dataset. Returns list of {"id", "prompt"}."""
        return load_prompts(self._config.dataset, self._config.dataset_path)

    def _validate_resume_path(self, resume_path: Path) -> None:
        expected_prefix = f"{self._config.dataset}_"
        if not resume_path.name.startswith(expected_prefix):
            raise ValueError(
                f"Resume file {resume_path.name} does not match dataset {self._config.dataset}"
            )

    @staticmethod
    def _build_public_watermark_params(generator: WatermarkGenerator) -> dict:
        generator_config = getattr(generator, "config", None)
        if generator_config is None:
            raise ValueError(
                "Generator must expose watermark config via .config"
            )
        params = {
            "lsh_d": generator_config.lsh_d,
            "lsh_gamma": generator_config.lsh_gamma,
            "margin_base": generator_config.margin_base,
            "margin_alpha": generator_config.margin_alpha,
        }
        adaptive_gamma = WatermarkPipeline._build_public_adaptive_gamma_params(generator)
        if adaptive_gamma is not None:
            params["adaptive_gamma"] = adaptive_gamma
        return params

    @staticmethod
    def _build_public_adaptive_gamma_params(
        generator: WatermarkGenerator,
    ) -> dict | None:
        generator_config = getattr(generator, "config", None)
        adaptive_config = getattr(generator_config, "adaptive_gamma", None)
        profile = getattr(generator, "_entropy_profile", None)
        if adaptive_config is None or not getattr(adaptive_config, "enabled", False):
            return None
        if profile is None:
            return None
        return {
            "strategy": getattr(adaptive_config, "strategy", "piecewise_quantile"),
            "profile_id": getattr(adaptive_config, "profile_id", None),
            "anchors": dict(getattr(adaptive_config, "anchors", {}) or {}),
            "profile": {
                "language": profile.language,
                "model_family": profile.model_family,
                "quantiles_units": dict(profile.quantiles_units_map),
                "strategy": profile.strategy,
            },
        }

    def run(self) -> str:
        """Run batch watermarking. Returns path to output JSONL file."""
        out_dir = Path(self._config.output_dir)
        resume_path, is_resume = resolve_resume_path(
            self._config.resume,
            out_dir,
            default_pattern=f"{self._config.dataset}_*.jsonl",
        )

        processed_ids: set[str] = set()
        if is_resume and resume_path is not None:
            self._validate_resume_path(resume_path)
            processed_ids = load_processed_ids(resume_path)

        all_prompts = self._load_prompts()
        if self._config.sample_limit is not None:
            all_prompts = all_prompts[: self._config.sample_limit]
        prompts = [item for item in all_prompts if item["id"] not in processed_ids]
        if is_resume and resume_path is not None and not prompts:
            print("All samples already processed", file=sys.stderr)
            return str(resume_path)

        out_dir.mkdir(parents=True, exist_ok=True)
        if is_resume and resume_path is not None:
            out_path = resume_path
            mode = "a"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"{self._config.dataset}_{timestamp}.jsonl"
            mode = "w"

        iterator = (
            tqdm(prompts, desc=f"Watermarking {self._config.dataset}", unit="prompt")
            if tqdm is not None
            else prompts
        )

        with open(out_path, mode, encoding="utf-8") as f:
            for item in iterator:
                result = self._generator.generate(item["prompt"])
                embed_rate = (
                    result.embedded_blocks / result.total_blocks
                    if result.total_blocks > 0
                    else 0.0
                )
                record = {
                    "id": item["id"],
                    "dataset": self._config.dataset,
                    "prompt": item["prompt"],
                    "generated_code": result.code,
                    "blocks": [asdict(contract) for contract in result.block_contracts],
                    "adaptive_mode": result.adaptive_mode,
                    "profile_id": result.profile_id,
                    "alignment_summary": result.alignment_summary,
                    "total_blocks": result.total_blocks,
                    "embedded_blocks": result.embedded_blocks,
                    "failed_blocks": result.failed_blocks,
                    "fallback_blocks": result.fallback_blocks,
                    "embed_rate": embed_rate,
                }
                record["watermark_params"] = self._build_public_watermark_params(
                    self._generator
                )
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                summary = (
                    f"  ✓ {item['id']} | "
                    f"blocks: {result.embedded_blocks}/{result.total_blocks} | "
                    f"failed: {result.failed_blocks} | "
                    f"fallback: {result.fallback_blocks} | "
                    f"embed_rate: {embed_rate:.1%}"
                )
                print(summary, file=sys.stderr)

        return str(out_path)
