import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from wfcllm.watermark.generator import EmbedStats, GenerateResult
from wfcllm.watermark.pipeline import WatermarkPipeline, WatermarkPipelineConfig


def test_watermark_metadata_roundtrip_is_non_secret(tmp_path):
    cfg = WatermarkPipelineConfig(
        dataset="humaneval",
        output_dir=str(tmp_path),
        dataset_path="data/datasets",
    )

    generator = MagicMock()
    generator.generate.return_value = GenerateResult(
        code="def foo():\n    return 1\n",
        stats=EmbedStats(
            total_blocks=2,
            embedded_blocks=1,
            failed_blocks=1,
            fallback_blocks=0,
        ),
    )
    generator.config = SimpleNamespace(
        lsh_d=4,
        lsh_gamma=0.75,
        margin_base=0.1,
        margin_alpha=0.05,
        secret_key="super-secret",
    )

    pipeline = WatermarkPipeline(generator=generator, config=cfg)
    with patch.object(pipeline, "_load_prompts", return_value=[
        {"id": "HumanEval/0", "prompt": "def foo():\n"},
    ]):
        output_path = pipeline.run()

    raw_text = Path(output_path).read_text(encoding="utf-8")
    row = json.loads(raw_text.splitlines()[0])
    assert row["watermark_params"] == {
        "lsh_d": 4,
        "lsh_gamma": 0.75,
        "margin_base": 0.1,
        "margin_alpha": 0.05,
    }
    assert "super-secret" not in raw_text
