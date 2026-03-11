"""Shared dataset loading utility for HumanEval and MBPP."""
from __future__ import annotations

from pathlib import Path

from datasets import load_dataset

SUPPORTED_DATASETS = ("humaneval", "mbpp")


def load_prompts(dataset: str, dataset_path: str) -> list[dict]:
    """Load prompts from local HumanEval or MBPP dataset.

    Args:
        dataset: One of "humaneval" or "mbpp".
        dataset_path: Root directory containing local dataset caches.

    Returns:
        List of dicts with keys "id" (str) and "prompt" (str).

    Raises:
        ValueError: If dataset is not in SUPPORTED_DATASETS.
    """
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"dataset must be one of {SUPPORTED_DATASETS}, got '{dataset}'"
        )

    path = str(Path(dataset_path) / dataset)

    if dataset == "humaneval":
        ds = load_dataset(
            "openai/openai_humaneval",
            cache_dir=path,
            download_mode="reuse_cache_if_exists",
        )
        prompts = []
        for split in ds:
            for item in ds[split]:
                prompts.append({"id": item["task_id"], "prompt": item["prompt"]})
        return prompts

    # mbpp
    ds = load_dataset(
        "google-research-datasets/mbpp",
        "full",
        cache_dir=path,
        download_mode="reuse_cache_if_exists",
    )
    prompts = []
    for split in ds:
        for item in ds[split]:
            prompts.append({"id": f"mbpp/{item['task_id']}", "prompt": item["text"]})
    return prompts
