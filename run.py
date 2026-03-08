"""统一运行入口：支持全流程、单阶段运行与断点续跑。

用法：
    python run.py                          # 全流程
    python run.py --phase encoder          # 只跑阶段一
    python run.py --status                 # 查看状态
    python run.py --reset                  # 清除断点
    python run.py --phase encoder --force  # 强制重跑
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PHASES = ["encoder", "watermark", "extract"]
DEFAULT_STATE_FILE = Path("data/run_state.json")


class RunState:
    """断点状态管理：读写 data/run_state.json。"""

    def __init__(self, path: Path = DEFAULT_STATE_FILE):
        self._path = path
        self._data: dict = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            with open(self._path, encoding="utf-8") as f:
                return json.load(f)
        return {phase: {"done": False} for phase in PHASES}

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
        self._data = {phase: {"done": False} for phase in PHASES}
        self._save()

    def status(self) -> dict:
        return {
            phase: {
                "done": self._data.get(phase, {}).get("done", False),
                **{k: v for k, v in self._data.get(phase, {}).items() if k != "done"},
            }
            for phase in PHASES
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WFCLLM 统一运行入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        choices=PHASES,
        help="运行指定阶段（不指定则运行全流程）",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="查看各阶段完成情况",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="清除断点状态，重头开始",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重跑指定阶段（忽略已完成标记）",
    )
    # Encoder 参数
    parser.add_argument("--model-name", default=None, help="CodeT5 模型名称或本地路径")
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--margin", type=float, default=None)
    parser.add_argument("--no-lora", action="store_true", help="禁用 LoRA")
    parser.add_argument("--no-bf16", action="store_true", help="禁用 BF16")
    # Watermark 参数
    parser.add_argument("--secret-key", default=None, help="水印密钥")
    parser.add_argument("--lm-model-path", default=None, help="代码生成 LLM 路径")
    parser.add_argument("--prompt", default=None, help="生成代码的输入提示")
    parser.add_argument("--output-file", default=None, help="保存生成代码的路径")
    # Extract 参数
    parser.add_argument("--code-file", default=None, help="待检测代码文件路径")
    parser.add_argument("--z-threshold", type=float, default=None, help="Z 分数阈值")
    return parser


def cmd_status(state: RunState) -> None:
    print("=== WFCLLM 阶段状态 ===")
    for phase in PHASES:
        info = state.status()[phase]
        done_str = "✓ 完成" if info["done"] else "○ 未完成"
        extras = {k: v for k, v in info.items() if k not in ("done", "completed_at")}
        extra_str = "  " + str(extras) if extras else ""
        print(f"  {phase:10s} {done_str}{extra_str}")


def cmd_reset(state: RunState) -> None:
    state.reset()
    print("已重置所有阶段状态。")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    state = RunState()

    if args.status:
        cmd_status(state)
        return 0

    if args.reset:
        cmd_reset(state)
        return 0

    phases_to_run = [args.phase] if args.phase else PHASES

    for phase in phases_to_run:
        if state.is_done(phase) and not args.force:
            print(f"[跳过] {phase}（已完成，使用 --force 强制重跑）")
            continue
        rc = run_phase(phase, args, state)
        if rc != 0:
            print(f"[失败] {phase} 阶段退出码 {rc}", file=sys.stderr)
            return rc

    return 0


def run_phase(phase: str, args: argparse.Namespace, state: RunState) -> int:
    """分发到各阶段 runner，返回退出码。"""
    runners = {
        "encoder": run_encoder,
        "watermark": run_watermark,
        "extract": run_extract,
    }
    return runners[phase](args, state)


def run_encoder(args: argparse.Namespace, state: RunState) -> int:
    """占位：阶段一实现在 Task 4。"""
    print("[encoder] 未实现")
    return 1


def run_watermark(args: argparse.Namespace, state: RunState) -> int:
    """占位：阶段二实现在 Task 5。"""
    print("[watermark] 未实现")
    return 1


def run_extract(args: argparse.Namespace, state: RunState) -> int:
    """占位：阶段三实现在 Task 6。"""
    print("[extract] 未实现")
    return 1


if __name__ == "__main__":
    sys.exit(main())
