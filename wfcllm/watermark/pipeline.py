"""Batch watermarking pipeline over HumanEval/MBPP datasets."""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]

import torch
from datasets import load_dataset

from wfcllm.watermark.generator import WatermarkGenerator

SUPPORTED_DATASETS = ("humaneval", "mbpp")


@dataclass
class WatermarkPipelineConfig:
    """Configuration for batch watermark embedding pipeline."""

    dataset: str            # "humaneval" or "mbpp"
    output_dir: str         # e.g. "data/watermarked"
    dataset_path: str       # local datasets root, e.g. "data/datasets"

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
        dataset_path = str(Path(self._config.dataset_path) / self._config.dataset)

        if self._config.dataset == "humaneval":
            ds = load_dataset(
                "openai/openai_humaneval",
                cache_dir=dataset_path,
                download_mode="reuse_cache_if_exists",
            )
            prompts = []
            for split in ds:
                for item in ds[split]:
                    prompts.append({
                        "id": item["task_id"],
                        "prompt": item["prompt"],
                    })
            return prompts

        # mbpp
        ds = load_dataset(
            "google-research-datasets/mbpp",
            "full",
            cache_dir=dataset_path,
            download_mode="reuse_cache_if_exists",
        )
        prompts = []
        for split in ds:
            for item in ds[split]:
                prompts.append({
                    "id": f"mbpp/{item['task_id']}",
                    "prompt": item["text"],
                })
        return prompts

    def run(self) -> str:
        """Run batch watermarking. Returns path to output JSONL file."""
        prompts = self._load_prompts()

        out_dir = Path(self._config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"{self._config.dataset}_{timestamp}.jsonl"

        iterator = (
            tqdm(prompts, desc=f"Watermarking {self._config.dataset}", unit="prompt")
            if tqdm is not None
            else prompts
        )

        with open(out_path, "w", encoding="utf-8") as f:
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
                    "total_blocks": result.total_blocks,
                    "embedded_blocks": result.embedded_blocks,
                    "failed_blocks": result.failed_blocks,
                    "fallback_blocks": result.fallback_blocks,
                    "embed_rate": embed_rate,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

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
