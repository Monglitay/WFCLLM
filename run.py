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
    """阶段一：训练语义编码器。"""
    import glob

    from wfcllm.encoder.config import EncoderConfig
    from wfcllm.encoder.train import main as encoder_main

    print("=== 阶段一：语义编码器预训练 ===")

    config = EncoderConfig()
    if args.model_name:
        config.model_name = args.model_name
    if args.embed_dim:
        config.embed_dim = args.embed_dim
    if args.lr:
        config.lr = args.lr
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.epochs = args.epochs
    if args.margin:
        config.margin = args.margin
    if args.no_lora:
        config.use_lora = False
    if args.no_bf16:
        config.use_bf16 = False

    # 若用户未指定 --model-name，自动检测本地
    if args.model_name is None:
        local_codet5 = Path(config.local_model_dir) / "codet5-base"
        if local_codet5.exists() and (local_codet5 / "config.json").exists():
            config.model_name = str(local_codet5)
            print(f"[自动] 使用本地模型: {config.model_name}")
        else:
            print(f"[回退] 使用 HF Hub 模型: {config.model_name}")

    try:
        encoder_main(config)
    except Exception as e:
        print(f"[错误] 编码器训练失败：{e}", file=sys.stderr)
        return 1

    # 找到最新的 checkpoint 文件
    ckpt_pattern = str(Path(config.checkpoint_dir) / "encoder_epoch*.pt")
    checkpoints = sorted(glob.glob(ckpt_pattern))
    checkpoint_path = checkpoints[-1] if checkpoints else config.checkpoint_dir

    state.mark_done("encoder", checkpoint=checkpoint_path)
    print(f"[完成] 编码器训练完毕，checkpoint: {checkpoint_path}")
    return 0


def run_watermark(args: argparse.Namespace, state: RunState) -> int:
    """阶段二：生成含水印代码。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from wfcllm.encoder.config import EncoderConfig
    from wfcllm.encoder.model import SemanticEncoder
    from wfcllm.watermark.config import WatermarkConfig
    from wfcllm.watermark.generator import WatermarkGenerator

    print("=== 阶段二：生成时水印嵌入 ===")

    # 前置检查
    if not state.is_done("encoder"):
        print("[错误] 请先完成阶段一（encoder）", file=sys.stderr)
        return 1
    if not args.secret_key:
        print("[错误] --secret-key 为必填参数", file=sys.stderr)
        return 1
    if not args.lm_model_path:
        print("[错误] --lm-model-path 为必填参数", file=sys.stderr)
        return 1
    prompt = args.prompt or "def solution():"

    encoder_checkpoint = state.get("encoder", "checkpoint")
    embed_dim = args.embed_dim or 128

    # 加载编码器
    enc_config = EncoderConfig(embed_dim=embed_dim)
    local_codet5 = Path(enc_config.local_model_dir) / "codet5-base"
    if local_codet5.exists() and (local_codet5 / "config.json").exists():
        enc_config.model_name = str(local_codet5)
        print(f"[自动] 编码器使用本地模型: {enc_config.model_name}")
    else:
        print(f"[回退] 编码器使用 HF Hub: {enc_config.model_name}")
    encoder = SemanticEncoder(config=enc_config)
    if encoder_checkpoint and Path(encoder_checkpoint).exists():
        ckpt = torch.load(encoder_checkpoint, map_location="cpu")
        encoder.load_state_dict(ckpt["model_state_dict"])
    encoder_tokenizer = AutoTokenizer.from_pretrained(enc_config.model_name)

    # 加载代码生成 LLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lm_tokenizer = AutoTokenizer.from_pretrained(args.lm_model_path)
    lm_model = AutoModelForCausalLM.from_pretrained(args.lm_model_path).to(device)

    config = WatermarkConfig(
        secret_key=args.secret_key,
        encoder_embed_dim=embed_dim,
        encoder_device=device,
    )
    generator = WatermarkGenerator(lm_model, lm_tokenizer, encoder, encoder_tokenizer, config)

    try:
        result = generator.generate(prompt)
    except Exception as e:
        print(f"[错误] 水印生成失败：{e}", file=sys.stderr)
        return 1

    # 保存结果
    output_file = args.output_file or "data/results/watermarked.py"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_file).write_text(result.code, encoding="utf-8")

    state.mark_done(
        "watermark",
        output_file=output_file,
        embedded_blocks=result.embedded_blocks,
        total_blocks=result.total_blocks,
    )
    print(f"[完成] 生成代码已保存至 {output_file}")
    print(f"       嵌入块: {result.embedded_blocks}/{result.total_blocks}")
    return 0


def run_extract(args: argparse.Namespace, state: RunState) -> int:
    """阶段三：检测代码水印。"""
    import torch
    from transformers import AutoTokenizer

    from wfcllm.encoder.config import EncoderConfig
    from wfcllm.encoder.model import SemanticEncoder
    from wfcllm.extract.config import ExtractConfig
    from wfcllm.extract.detector import WatermarkDetector

    print("=== 阶段三：水印提取与验证 ===")

    # 前置检查
    if not state.is_done("encoder"):
        print("[错误] 请先完成阶段一（encoder）", file=sys.stderr)
        return 1
    if not args.secret_key:
        print("[错误] --secret-key 为必填参数", file=sys.stderr)
        return 1
    if not args.code_file:
        print("[错误] --code-file 为必填参数", file=sys.stderr)
        return 1
    code_path = Path(args.code_file)
    if not code_path.exists():
        print(f"[错误] 文件不存在：{args.code_file}", file=sys.stderr)
        return 1

    encoder_checkpoint = state.get("encoder", "checkpoint")
    embed_dim = args.embed_dim or 128
    z_threshold = args.z_threshold or 3.0

    # 加载编码器
    enc_config = EncoderConfig(embed_dim=embed_dim)
    local_codet5 = Path(enc_config.local_model_dir) / "codet5-base"
    if local_codet5.exists() and (local_codet5 / "config.json").exists():
        enc_config.model_name = str(local_codet5)
        print(f"[自动] 编码器使用本地模型: {enc_config.model_name}")
    else:
        print(f"[回退] 编码器使用 HF Hub: {enc_config.model_name}")
    encoder = SemanticEncoder(config=enc_config)
    if encoder_checkpoint and Path(encoder_checkpoint).exists():
        ckpt = torch.load(encoder_checkpoint, map_location="cpu")
        encoder.load_state_dict(ckpt["model_state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = encoder.to(device)
    tokenizer = AutoTokenizer.from_pretrained(enc_config.model_name)

    config = ExtractConfig(secret_key=args.secret_key, embed_dim=embed_dim, z_threshold=z_threshold)
    detector = WatermarkDetector(config, encoder, tokenizer, device=device)

    code = code_path.read_text(encoding="utf-8")
    try:
        result = detector.detect(code)
    except Exception as e:
        print(f"[错误] 检测失败：{e}", file=sys.stderr)
        return 1

    verdict = "【含水印】" if result.is_watermarked else "【无水印】"
    print(f"\n{verdict}")
    print(f"  Z 分数:   {result.z_score:.4f}（阈值 {z_threshold}）")
    print(f"  p 值:     {result.p_value:.6f}")
    print(f"  独立块:   {result.independent_blocks}/{result.total_blocks}")
    print(f"  命中块:   {result.hit_blocks}")

    state.mark_done(
        "extract",
        code_file=args.code_file,
        is_watermarked=result.is_watermarked,
        z_score=result.z_score,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
