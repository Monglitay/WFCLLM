"""统一运行入口：支持全流程、单阶段运行与断点续跑。

流程概述：
    阶段一（encoder）  — 对比学习预训练鲁棒语义编码器
    阶段二（watermark）— 遍历 HumanEval/MBPP 数据集，批量生成含水印代码并输出 JSONL
    阶段三（extract）  — 读取水印 JSONL，批量检测并输出 details JSONL + summary JSON

用法示例：
    python run.py                          # 全流程（自动跳过已完成阶段）
    python run.py --phase encoder          # 只跑阶段一
    python run.py --status                 # 查看各阶段完成情况
    python run.py --reset                  # 清除断点状态，重头开始
    python run.py --phase encoder --force  # 强制重跑（忽略已完成标记）

    # 阶段二：对 humaneval 数据集批量嵌入水印
    python run.py --phase watermark \
        --lm-model-path data/models/deepseek-coder-7b \
        --secret-key mysecret \
        --dataset humaneval

    # 阶段三：检测水印 JSONL，输出 details JSONL + 统计摘要
    python run.py --phase extract \
        --secret-key mysecret \
        --input-file data/watermarked/humaneval_20260309_120000.jsonl

    # 阶段二：恢复最新 watermark 输出
    python run.py --phase watermark \
        --lm-model-path data/models/deepseek-coder-7b \
        --secret-key mysecret \
        --dataset humaneval \
        --resume latest

    # 阶段三：恢复最新 extract details 文件
    python run.py --phase extract \
        --secret-key mysecret \
        --input-file data/watermarked/humaneval_20260318_120000.jsonl \
        --resume latest

    # 阶段三（先用负样本语料自动校准 FPR 阈值，再检测）
    python run.py --phase extract \
        --secret-key mysecret \
        --calibration-corpus data/negative_corpus.jsonl \
        --fpr 0.01
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PHASES = ["encoder", "watermark", "extract"]
OPTIONAL_PHASES = ["generate-negative", "token-channel-train"]
ALL_PHASES = PHASES + OPTIONAL_PHASES
DEFAULT_STATE_FILE = Path("data/run_state.json")

from wfcllm.cli.arguments import build_parser, DEFAULT_CONFIG_FILE  # noqa: E402
from wfcllm.cli.config_resolver import (  # noqa: E402
    parse_optional_bool,
    load_config,
    resolve_extract_lsh_params,
    resolve_adaptive_gamma_config,
    resolve_extract_adaptive_gamma_config,
    resolve_token_channel_config,
    _apply_token_channel_cli_overrides,
    build_extract_calibration_contract_builder,
    resolve_adaptive_detection_config,
)
from wfcllm.cli.runners import (  # noqa: E402
    run_phase,
    run_encoder,
    run_watermark,
    run_offline_analysis,
    run_extract,
    run_generate_negative,
    resolve_token_channel_train_config,
    validate_token_channel_train_config,
    run_token_channel_train,
    get_config,
    configured_extract_input,
    is_compare_only_mode,
    has_explicit_extract_input,
    validate_compare_only_mode,
    COMPARE_ONLY_REQUIRED_FLAGS,
    COMPARE_ONLY_OPTIONAL_WATERMARKED_FLAGS,
)


class RunState:
    """断点状态管理：读写 data/run_state.json。"""

    def __init__(self, path: Path = DEFAULT_STATE_FILE):
        self._path = path
        self._data: dict = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            with open(self._path, encoding="utf-8") as f:
                return json.load(f)
        return {phase: {"done": False} for phase in ALL_PHASES}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def is_done(self, phase: str) -> bool:
        return self._data.get(phase, {}).get("done", False)

    def get(self, phase: str, key: str) -> str | None:
        return self._data.get(phase, {}).get(key)

    def mark_done(self, phase: str, **kwargs) -> None:
        self._data[phase] = {
            "done": True,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            **kwargs,
        }
        self._save()

    def reset(self) -> None:
        self._data = {phase: {"done": False} for phase in ALL_PHASES}
        self._save()

    def status(self) -> dict:
        return {
            phase: {
                "done": self._data.get(phase, {}).get("done", False),
                **{k: v for k, v in self._data.get(phase, {}).items() if k != "done"},
            }
            for phase in ALL_PHASES
        }


def cmd_status(state: RunState) -> None:
    print("=== WFCLLM 阶段状态 ===")
    for phase in ALL_PHASES:
        info = state.status()[phase]
        done_str = "✓ 完成" if info["done"] else "○ 未完成"
        extras = {k: v for k, v in info.items() if k not in ("done", "completed_at")}
        extra_str = "  " + str(extras) if extras else ""
        print(f"  {phase:10s} {done_str}{extra_str}")


def cmd_reset(state: RunState) -> None:
    state.reset()
    print("已重置所有阶段状态。")


def should_skip_completed_phase(args: argparse.Namespace, phase: str, state: RunState) -> bool:
    if not state.is_done(phase):
        return False
    if args.force or args.eval_only or is_compare_only_mode(args):
        return False
    if phase == "extract" and has_explicit_extract_input(args):
        return False
    return True



def main() -> int:
    log_level = logging.DEBUG if os.environ.get("WFCLLM_DEBUG") else logging.WARNING
    logging.basicConfig(level=log_level, format="%(name)s %(levelname)s %(message)s")

    parser = build_parser()
    args = parser.parse_args()
    state = RunState()

    if args.status:
        cmd_status(state)
        return 0

    if args.reset:
        cmd_reset(state)
        return 0

    compare_only_error = validate_compare_only_mode(args)
    if compare_only_error is not None:
        print(compare_only_error, file=sys.stderr)
        return 1

    phases_to_run = [args.phase] if args.phase else PHASES

    for phase in phases_to_run:
        if should_skip_completed_phase(args, phase, state):
            print(f"[跳过] {phase}（已完成，使用 --force 强制重跑）")
            continue
        rc = run_phase(phase, args, state)
        if rc != 0:
            print(f"[失败] {phase} 阶段退出码 {rc}", file=sys.stderr)
            return rc

    return 0


if __name__ == "__main__":
    sys.exit(main())
