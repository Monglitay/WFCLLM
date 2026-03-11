"""Watermark extraction and verification module."""

from wfcllm.extract.calibrator import ThresholdCalibrator
from wfcllm.extract.config import BlockScore, DetectionResult, ExtractConfig
from wfcllm.extract.detector import WatermarkDetector
from wfcllm.extract.dp_selector import DPSelector
from wfcllm.extract.hypothesis import HypothesisTester
from wfcllm.extract.scorer import BlockScorer

__all__ = [
    "ExtractConfig",
    "DetectionResult",
    "BlockScore",
    "WatermarkDetector",
    "BlockScorer",
    "DPSelector",
    "HypothesisTester",
    "ThresholdCalibrator",
]

from wfcllm.extract.pipeline import ExtractPipeline, ExtractPipelineConfig
