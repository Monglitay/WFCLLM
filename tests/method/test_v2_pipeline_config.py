from __future__ import annotations

from pathlib import Path

import pytest

from wfcllm.method.config import (
    WFCLLMGenerationConfig,
    WFCLLMPipelineConfig,
)


def _config(tmp_path: Path, **overrides) -> WFCLLMPipelineConfig:
    model_path = tmp_path / "model"
    model_path.mkdir(exist_ok=True)
    values = {
        "dataset": "humaneval",
        "dataset_path": "data/datasets",
        "output_dir": str(tmp_path / "run"),
        "generation": WFCLLMGenerationConfig(
            model_path=str(model_path),
            device="cpu",
        ),
        "method_version": "v2",
        "evidence_retry_attempts": 20,
    }
    values.update(overrides)
    return WFCLLMPipelineConfig(**values)


def test_v2_pipeline_config_requires_retry_twenty(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly 20"):
        _config(tmp_path, evidence_retry_attempts=19)


def test_v2_pipeline_config_accepts_explicit_unique_sample_ids(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        sample_ids=("HumanEval/2", "HumanEval/0"),
        retry_attempt_ledger_output=str(tmp_path / "ledger.jsonl"),
    )

    assert config.method_version == "v2"
    assert config.evidence_retry_attempts == 20
    assert config.sample_ids == ("HumanEval/2", "HumanEval/0")


@pytest.mark.parametrize(
    "sample_ids",
    [
        "HumanEval/0",
        (),
        ("",),
        ("HumanEval/0", "HumanEval/0"),
    ],
)
def test_pipeline_config_rejects_invalid_sample_ids(
    tmp_path: Path,
    sample_ids,
) -> None:
    with pytest.raises(ValueError, match="sample_ids"):
        _config(tmp_path, sample_ids=sample_ids)


def test_v1_pipeline_config_keeps_existing_retry_default(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()

    config = WFCLLMPipelineConfig(
        dataset="humaneval",
        dataset_path="data/datasets",
        output_dir=str(tmp_path / "run"),
        generation=WFCLLMGenerationConfig(
            model_path=str(model_path),
            device="cpu",
        ),
    )

    assert config.method_version == "v1"
    assert config.evidence_retry_attempts == 1
    assert config.sample_ids is None
    assert config.retry_attempt_ledger_output is None


def test_v1_rejects_v2_retry_ledger_output(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()

    with pytest.raises(ValueError, match="v2"):
        WFCLLMPipelineConfig(
            dataset="humaneval",
            dataset_path="data/datasets",
            output_dir=str(tmp_path / "run"),
            generation=WFCLLMGenerationConfig(
                model_path=str(model_path),
                device="cpu",
            ),
            retry_attempt_ledger_output=str(tmp_path / "ledger.jsonl"),
        )
