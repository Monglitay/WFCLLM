"""CLI runner for the pretrain phase."""
from __future__ import annotations

import argparse

from wfcllm.orchestration.state import RunStateManager
from wfcllm.pretrain.config import PretrainConfig
from wfcllm.pretrain.pipeline import PretrainPipeline


def run_pretrain(args: argparse.Namespace, state: RunStateManager) -> int:
    """Phase entry point. Reads --stages from args (None → default both)."""
    stages = getattr(args, "stages", None)
    cfg = PretrainConfig(stages=list(stages)) if stages else PretrainConfig()
    print(f"=== Phase: pretrain (stages={cfg.stages}) ===")
    return PretrainPipeline(cfg).run(args, state)
