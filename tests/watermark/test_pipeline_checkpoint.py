import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from wfcllm.watermark.generator import EmbedStats, GenerateResult
from wfcllm.watermark.pipeline import WatermarkPipeline, WatermarkPipelineConfig


def _result(code: str) -> GenerateResult:
    return GenerateResult(
        code=code,
        stats=EmbedStats(
            total_blocks=2,
            embedded_blocks=1,
            failed_blocks=0,
            fallback_blocks=0,
        ),
    )


def test_resume_latest_appends_only_remaining_samples(tmp_path: Path):
    partial = tmp_path / "humaneval_20260318_120000.jsonl"
    partial.write_text('{"id":"HumanEval/0","generated_code":"done"}\n', encoding="utf-8")

    cfg = WatermarkPipelineConfig(
        dataset="humaneval",
        output_dir=str(tmp_path),
        dataset_path="data/datasets",
        resume="latest",
    )
    generator = MagicMock()
    generator.generate.side_effect = [_result("code-1"), _result("code-2")]
    generator.config = SimpleNamespace(
        lsh_d=4,
        lsh_gamma=0.75,
        margin_base=0.1,
        margin_alpha=0.05,
    )
    pipeline = WatermarkPipeline(generator=generator, config=cfg)

    prompts = [
        {"id": "HumanEval/0", "prompt": "p0"},
        {"id": "HumanEval/1", "prompt": "p1"},
        {"id": "HumanEval/2", "prompt": "p2"},
    ]
    with patch.object(pipeline, "_load_prompts", return_value=prompts):
        out_path = pipeline.run()

    assert out_path == str(partial)
    lines = partial.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["id"] for line in lines] == [
        "HumanEval/0",
        "HumanEval/1",
        "HumanEval/2",
    ]
    assert generator.generate.call_count == 2


def test_resume_all_processed_returns_existing_file_without_generation(tmp_path: Path):
    partial = tmp_path / "humaneval_20260318_120000.jsonl"
    partial.write_text('{"id":"HumanEval/0"}\n{"id":"HumanEval/1"}\n', encoding="utf-8")
    cfg = WatermarkPipelineConfig(
        dataset="humaneval",
        output_dir=str(tmp_path),
        dataset_path="data/datasets",
        resume=str(partial),
    )
    generator = MagicMock()
    pipeline = WatermarkPipeline(generator=generator, config=cfg)
    with patch.object(
        pipeline,
        "_load_prompts",
        return_value=[
            {"id": "HumanEval/0", "prompt": "p0"},
            {"id": "HumanEval/1", "prompt": "p1"},
        ],
    ):
        assert pipeline.run() == str(partial)
    generator.generate.assert_not_called()


def test_resume_explicit_with_wrong_dataset_prefix_raises(tmp_path: Path):
    wrong = tmp_path / "mbpp_20260318_120000.jsonl"
    wrong.write_text('{"id":"mbpp/1"}\n', encoding="utf-8")
    pipeline = WatermarkPipeline(
        generator=MagicMock(),
        config=WatermarkPipelineConfig(
            dataset="humaneval",
            output_dir=str(tmp_path),
            dataset_path="data/datasets",
            resume=str(wrong),
        ),
    )
    with pytest.raises(ValueError, match="does not match dataset"):
        pipeline.run()
