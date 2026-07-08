"""Generation-time watermark embedding module."""

from wfcllm.watermark.config import WatermarkConfig
from wfcllm.watermark.adaptive_gamma.entropy import NodeEntropyEstimator
from wfcllm.watermark.orchestrator import GenerateResult, WatermarkGenerator
from wfcllm.watermark.interceptor import InterceptEvent, StatementInterceptor
from wfcllm.semantic.keying import WatermarkKeying
from wfcllm.generation.kv_cache import CacheSnapshot, KVCacheManager
from wfcllm.semantic.verifier import ProjectionVerifier, VerifyResult

__all__ = [
    "WatermarkConfig",
    "NodeEntropyEstimator",
    "WatermarkGenerator",
    "GenerateResult",
    "StatementInterceptor",
    "InterceptEvent",
    "WatermarkKeying",
    "KVCacheManager",
    "CacheSnapshot",
    "ProjectionVerifier",
    "VerifyResult",
]

from wfcllm.watermark.pipeline import WatermarkPipeline, WatermarkPipelineConfig
