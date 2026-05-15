"""LSH verification + adaptive-gamma resolution slice of WatermarkGenerator.

Holds a back-reference to the orchestrator and reads its mutable attributes
(_config, _entropy_est, _verifier, _keying, _entropy_profile, _gamma_schedule)
directly. This preserves the existing test pattern of monkey-patching
`generator._verify_block`, `generator._verifier.verify`, etc.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wfcllm.watermark.orchestrator import WatermarkGenerator


class SemanticChannel:
    def __init__(self, orchestrator: "WatermarkGenerator") -> None:
        self._orch = orchestrator
