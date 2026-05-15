"""Pretrain pipeline: dispatches encoder + lexical stages in canonical order."""
from __future__ import annotations

import argparse
import sys

from wfcllm.cli.runners import run_encoder, run_token_channel_train
from wfcllm.orchestration.state import RunStateManager
from wfcllm.pretrain.config import PretrainConfig


class PretrainPipeline:
    """Sequential dispatcher for pretrain stages.

    For each stage in cfg.stages (already canonically ordered):
      - "encoder": call run_encoder
      - "lexical": fail-fast if encoder not done, else call run_token_channel_train
    Returns the first non-zero exit code, or 0 if all stages succeed.
    """

    def __init__(self, cfg: PretrainConfig):
        self.cfg = cfg

    def run(self, args: argparse.Namespace, state: RunStateManager) -> int:
        for stage in self.cfg.stages:
            if stage == "encoder":
                rc = run_encoder(args, state)
                if rc != 0:
                    return rc
            elif stage == "lexical":
                if not state.is_done("encoder"):
                    print(
                        "[错误] pretrain --stages lexical 需要先完成 encoder 阶段（encoder 未完成）",
                        file=sys.stderr,
                    )
                    return 1
                rc = run_token_channel_train(args, state)
                if rc != 0:
                    return rc
            else:  # defensive — PretrainConfig validates
                print(f"[错误] unknown stage: {stage}", file=sys.stderr)
                return 1
        return 0
