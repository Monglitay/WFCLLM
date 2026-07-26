"""Shared dataset loading utility for HumanEval and MBPP."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset

from wfcllm.datasets.constants import SUPPORTED_DATASETS
from wfcllm.datasets.mbpp_interface import build_mbpp_generation_prompt


def load_prompts(
    dataset: str,
    dataset_path: str,
    sample_limit: int | None = None,
    sample_offset: int | None = None,
) -> list[dict]:
    """Load prompts from local HumanEval or MBPP dataset.

    Args:
        dataset: One of "humaneval" or "mbpp".
        dataset_path: Root directory containing local dataset caches.
        sample_limit: Optional maximum number of prompt rows to return.
        sample_offset: Optional number of prompt rows to skip before limiting.

    Returns:
        List of dicts with keys "id" (str) and "prompt" (str).

    Raises:
        ValueError: If dataset is not in SUPPORTED_DATASETS or slicing args
            are negative.
    """
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"dataset must be one of {SUPPORTED_DATASETS}, got '{dataset}'"
        )
    if sample_offset is not None and sample_offset < 0:
        raise ValueError("sample_offset must be non-negative")
    if sample_limit is not None and sample_limit < 0:
        raise ValueError("sample_limit must be non-negative")

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
        return _slice_prompt_rows(prompts, sample_limit, sample_offset)

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
    return _slice_prompt_rows(prompts, sample_limit, sample_offset)


def _slice_prompt_rows(
    rows: list[dict],
    sample_limit: int | None,
    sample_offset: int | None,
) -> list[dict]:
    if sample_offset is not None:
        rows = rows[sample_offset:]
    if sample_limit is not None:
        rows = rows[:sample_limit]
    return rows


def load_mbpp_generation_samples(dataset_path: str) -> list[dict]:
    """Load MBPP rows with leakage-safe callable interface prompts."""
    path = str(Path(dataset_path) / "mbpp")
    ds = load_dataset(
        "google-research-datasets/mbpp",
        "full",
        cache_dir=path,
        download_mode="reuse_cache_if_exists",
    )
    rows = []
    for split in ds:
        for item in ds[split]:
            prompt, interface = build_mbpp_generation_prompt(
                item["text"],
                item["test_list"],
                reference_code=item["code"],
            )
            rows.append(
                {
                    "id": f"mbpp/{item['task_id']}",
                    "prompt": prompt,
                    "generated_code": item["code"],
                    "interface_extraction_status": (
                        "interface_aware" if interface is not None else "fallback"
                    ),
                    "interface_function_name": (
                        interface.function_name if interface is not None else None
                    ),
                    "interface_positional_arities": (
                        list(interface.positional_arities)
                        if interface is not None
                        else []
                    ),
                    "interface_parameter_names": (
                        list(interface.parameter_names)
                        if interface is not None
                        else []
                    ),
                    "interface_helper_classes": (
                        list(interface.helper_classes)
                        if interface is not None
                        else []
                    ),
                }
            )
    return rows


def load_reference_solutions(dataset: str, dataset_path: str) -> list[dict]:
    """Load prompt/reference-solution pairs from local datasets."""
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
        rows = []
        for split in ds:
            for item in ds[split]:
                rows.append(
                    {
                        "id": item["task_id"],
                        "prompt": item["prompt"],
                        "generated_code": item["canonical_solution"],
                    }
                )
        return rows

    ds = load_dataset(
        "google-research-datasets/mbpp",
        "full",
        cache_dir=path,
        download_mode="reuse_cache_if_exists",
    )
    rows = []
    for split in ds:
        for item in ds[split]:
            rows.append(
                {
                    "id": f"mbpp/{item['task_id']}",
                    "prompt": item["text"],
                    "generated_code": item["code"],
                }
            )
    return rows


@dataclass
class TestCase:
    """A single test case for code execution evaluation."""

    task_id: str
    entry_point: str | None
    test_code: str


def load_test_cases(dataset: str, dataset_path: str) -> dict[str, TestCase]:
    """Load test cases from local HumanEval or MBPP dataset.

    Args:
        dataset: One of "humaneval" or "mbpp".
        dataset_path: Root directory containing local dataset caches.

    Returns:
        Dict mapping task_id to TestCase.

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
        cases: dict[str, TestCase] = {}
        for split in ds:
            for item in ds[split]:
                tid = item["task_id"]
                cases[tid] = TestCase(
                    task_id=tid,
                    entry_point=item["entry_point"],
                    test_code=item["test"],
                )
        return cases

    # mbpp
    ds = load_dataset(
        "google-research-datasets/mbpp",
        "full",
        cache_dir=path,
        download_mode="reuse_cache_if_exists",
    )
    cases = {}
    for split in ds:
        for item in ds[split]:
            tid = f"mbpp/{item['task_id']}"
            cases[tid] = TestCase(
                task_id=tid,
                entry_point=None,
                test_code="\n".join(item["test_list"]),
            )
    return cases
