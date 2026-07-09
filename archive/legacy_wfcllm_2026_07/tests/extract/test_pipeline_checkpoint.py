import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from wfcllm.extract.config import DetectionResult
from wfcllm.extract.pipeline import ExtractPipeline, ExtractPipelineConfig


def _detect(is_watermarked: bool, z_score: float) -> DetectionResult:
    return DetectionResult(
        is_watermarked=is_watermarked,
        z_score=z_score,
        p_value=0.001 if is_watermarked else 0.5,
        total_blocks=10,
        independent_blocks=8,
        hit_blocks=7 if is_watermarked else 4,
        block_details=[],
    )


def test_resume_latest_appends_remaining_details_and_rebuilds_summary(
    tmp_path: Path,
):
    input_file = tmp_path / "humaneval.jsonl"
    input_file.write_text(
        "\n".join(
            [
                '{"id":"HumanEval/0","generated_code":"c0","embed_rate":0.1}',
                '{"id":"HumanEval/1","generated_code":"c1","embed_rate":0.2}',
                '{"id":"HumanEval/2","generated_code":"c2","embed_rate":0.3}',
            ]
        ),
        encoding="utf-8",
    )
    details = tmp_path / "humaneval_details.jsonl"
    details.write_text(
        '{"id":"HumanEval/0","is_watermarked":true,"z_score":4.0,"p_value":0.001,"independent_blocks":8,"hits":7}\n',
        encoding="utf-8",
    )

    detector = MagicMock()
    detector.detect.side_effect = [_detect(True, 3.0), _detect(False, 1.0)]
    pipeline = ExtractPipeline(
        detector=detector,
        config=ExtractPipelineConfig(
            input_file=str(input_file),
            output_dir=str(tmp_path),
            resume="latest",
        ),
    )

    out_path = pipeline.run()
    summary_path = tmp_path / "humaneval_summary.json"

    assert out_path == str(details)
    assert summary_path.exists()
    assert detector.detect.call_count == 2


def test_resume_all_processed_regenerates_summary(tmp_path: Path):
    input_file = tmp_path / "humaneval.jsonl"
    input_file.write_text(
        '{"id":"HumanEval/0","generated_code":"c0","embed_rate":0.1}\n',
        encoding="utf-8",
    )
    details = tmp_path / "humaneval_details.jsonl"
    details.write_text(
        '{"id":"HumanEval/0","is_watermarked":true,"z_score":4.0,"p_value":0.001,"independent_blocks":8,"hits":7}\n',
        encoding="utf-8",
    )

    pipeline = ExtractPipeline(
        detector=MagicMock(),
        config=ExtractPipelineConfig(
            input_file=str(input_file),
            output_dir=str(tmp_path),
            resume=str(details),
        ),
    )

    out_path = pipeline.run()
    assert out_path == str(details)
    assert (tmp_path / "humaneval_summary.json").exists()


def test_resume_explicit_with_wrong_input_stem_raises(tmp_path: Path):
    input_file = tmp_path / "humaneval.jsonl"
    input_file.write_text(
        '{"id":"HumanEval/0","generated_code":"c0","embed_rate":0.1}\n',
        encoding="utf-8",
    )
    wrong = tmp_path / "mbpp_details.jsonl"
    wrong.write_text(
        '{"id":"mbpp/1","is_watermarked":true,"z_score":4.0,"p_value":0.001,"independent_blocks":8,"hits":7}\n',
        encoding="utf-8",
    )

    pipeline = ExtractPipeline(
        detector=MagicMock(),
        config=ExtractPipelineConfig(
            input_file=str(input_file),
            output_dir=str(tmp_path),
            resume=str(wrong),
        ),
    )

    with pytest.raises(ValueError, match="does not match input file"):
        pipeline.run()
