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
OPTIONAL_PHASES = ["generate-negative"]
ALL_PHASES = PHASES + OPTIONAL_PHASES
DEFAULT_STATE_FILE = Path("data/run_state.json")
DEFAULT_CONFIG_FILE = Path("configs/base_config.json")


def load_config(config_path: Path) -> dict:
    """读取 JSON 配置文件，返回按阶段分组的 dict。"""
    if not config_path.exists():
        print(f"[错误] 配置文件不存在：{config_path}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"[错误] 配置文件解析失败：{e}", file=sys.stderr)
        sys.exit(1)


def resolve_extract_lsh_params(first_record: dict, ext_cfg: dict) -> tuple[int, float]:
    params = first_record.get("watermark_params") or {}
    lsh_d_raw = params.get("lsh_d", ext_cfg.get("lsh_d", 3))
    lsh_gamma_raw = params.get("lsh_gamma", ext_cfg.get("lsh_gamma", 0.5))
    try:
        return int(lsh_d_raw), float(lsh_gamma_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"LSH 参数解析失败：lsh_d={lsh_d_raw!r}, lsh_gamma={lsh_gamma_raw!r}"
        ) from exc


def resolve_adaptive_gamma_config(args: argparse.Namespace, wm_cfg: dict):
    from wfcllm.watermark.config import AdaptiveGammaConfig

    configured = wm_cfg.get("adaptive_gamma") or {}
    defaults = AdaptiveGammaConfig()
    anchors = defaults.anchors.copy()
    raw_anchors = configured.get("anchors")
    if isinstance(raw_anchors, dict):
        anchors.update(raw_anchors)

    enabled = bool(configured.get("enabled", defaults.enabled))
    if (
        getattr(args, "gamma_strategy", None) is not None
        or getattr(args, "entropy_profile", None) is not None
        or getattr(args, "profile_id", None) is not None
    ):
        enabled = True

    return AdaptiveGammaConfig(
        enabled=enabled,
        strategy=(
            getattr(args, "gamma_strategy", None)
            or configured.get("strategy", defaults.strategy)
        ),
        profile_path=(
            getattr(args, "entropy_profile", None)
            if getattr(args, "entropy_profile", None) is not None
            else configured.get("profile_path", defaults.profile_path)
        ),
        profile_id=(
            getattr(args, "profile_id", None)
            if getattr(args, "profile_id", None) is not None
            else configured.get("profile_id", defaults.profile_id)
        ),
        gamma_min=float(configured.get("gamma_min", defaults.gamma_min)),
        gamma_max=float(configured.get("gamma_max", defaults.gamma_max)),
        anchors=anchors,
    )


def resolve_extract_adaptive_gamma_config(args: argparse.Namespace, cfg: dict):
    extract_cfg = cfg.get("extract", {})
    configured = extract_cfg.get("adaptive_gamma")
    if isinstance(configured, dict):
        return resolve_adaptive_gamma_config(
            args,
            {"adaptive_gamma": configured},
        )
    return resolve_adaptive_gamma_config(args, cfg.get("watermark", {}))


def resolve_token_channel_config(section: dict | None):
    from wfcllm.watermark.token_channel.config import TokenChannelConfig

    if section is None:
        configured = {}
    elif isinstance(section, dict):
        configured = section
    else:
        raise ValueError("token_channel must be a JSON object")
    return TokenChannelConfig.from_mapping(configured)


def build_extract_calibration_contract_builder(
    adaptive_detection_config,
    adaptive_gamma_config,
    lsh_d: int,
):
    if not getattr(adaptive_detection_config, "prefer_adaptive", False):
        return None
    if not getattr(adaptive_gamma_config, "enabled", False):
        return None

    from wfcllm.extract.alignment import rebuild_block_contracts

    def builder(code: str) -> dict[str, dict]:
        return {
            contract["block_id"]: contract
            for contract in rebuild_block_contracts(
                code,
                adaptive_gamma_config=adaptive_gamma_config,
                default_lsh_d=lsh_d,
            )
        }

    return builder


def resolve_adaptive_detection_config(args: argparse.Namespace, ext_cfg: dict):
    from wfcllm.extract.config import AdaptiveDetectionConfig

    configured = ext_cfg.get("adaptive_detection") or {}
    defaults = AdaptiveDetectionConfig()

    require_block_contract_check = bool(
        configured.get(
            "require_block_contract_check",
            defaults.require_block_contract_check,
        )
    )
    fail_on_structure_mismatch = bool(
        configured.get(
            "fail_on_structure_mismatch",
            defaults.fail_on_structure_mismatch,
        )
    )
    if getattr(args, "strict_contract", False):
        require_block_contract_check = True
        fail_on_structure_mismatch = True

    return AdaptiveDetectionConfig(
        mode=(
            getattr(args, "adaptive_detection_mode", None)
            or configured.get("mode", defaults.mode)
        ),
        require_block_contract_check=require_block_contract_check,
        fail_on_structure_mismatch=fail_on_structure_mismatch,
        warn_on_numeric_mismatch=bool(
            configured.get(
                "warn_on_numeric_mismatch",
                defaults.warn_on_numeric_mismatch,
            )
        ),
        exclude_invalid_samples=bool(
            configured.get(
                "exclude_invalid_samples",
                defaults.exclude_invalid_samples,
            )
        ),
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WFCLLM 统一运行入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_FILE,
        help=f"配置文件路径（默认: {DEFAULT_CONFIG_FILE}）",
    )
    parser.add_argument(
        "--phase",
        choices=ALL_PHASES,
        help="运行指定阶段（不指定则运行主流程三阶段）",
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
    parser.add_argument(
        "--resume",
        default=None,
        help="样本级断点恢复：latest 或已有 JSONL 文件路径（仅 watermark/extract 阶段有效）",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="只跑评测，不训练（需配合 --phase encoder）",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="评测用的 checkpoint 路径（不传则从 run_state.json 读取）",
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
    parser.add_argument(
        "--dataset",
        default=None,
        choices=["humaneval", "mbpp"],
        help="水印嵌入数据集（humaneval 或 mbpp，默认: humaneval）",
    )
    parser.add_argument("--dataset-path", default=None, help="本地数据集根目录（默认: data/datasets）")
    parser.add_argument("--output-dir", default=None, help="水印 JSONL 输出目录（默认: data/watermarked）")
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="仅处理前 N 条 watermark prompts（调试/子集验证用）",
    )
    parser.add_argument(
        "--gamma-strategy",
        choices=["piecewise_quantile"],
        default=None,
        help="自适应 gamma 调度策略（默认从配置文件读取）",
    )
    parser.add_argument(
        "--entropy-profile",
        default=None,
        help="entropy profile JSON 路径（启用 adaptive gamma 时使用）",
    )
    parser.add_argument(
        "--profile-id",
        default=None,
        help="输出 watermark metadata 时使用的 entropy profile 标识",
    )
    # Extract 参数
    parser.add_argument(
        "--input-file",
        default=None,
        help="待检测的水印 JSONL 文件路径（不传则从 run_state 读取阶段二输出）",
    )
    parser.add_argument("--extract-output-dir", default=None, help="检测报告输出目录（默认: data/results）")
    parser.add_argument("--fpr-threshold", type=float, default=None, help="FPR 阈值 M_r（默认: 3.0，需通过校准脚本生成）")
    parser.add_argument(
        "--calibration-corpus",
        default=None,
        help="负样本校准语料 JSONL 路径（提供则自动运行 ThresholdCalibrator）",
    )
    parser.add_argument(
        "--fpr",
        type=float,
        default=None,
        help="校准目标 FPR（不传则优先读取 config，仅在 --calibration-corpus 指定时生效）",
    )
    parser.add_argument(
        "--adaptive-detection-mode",
        choices=["fixed", "prefer-adaptive", "require-adaptive"],
        default=None,
        help="提取阶段 adaptive hypothesis 的模式（默认从配置文件读取）",
    )
    parser.add_argument(
        "--strict-contract",
        action="store_true",
        help="强制启用 block contract 检查并在结构不匹配时严格失败",
    )
    parser.add_argument("--compare-summary-left", default=None, help="离线对比左侧 summary JSON 路径")
    parser.add_argument("--compare-details-left", default=None, help="离线对比左侧 details JSONL 路径")
    parser.add_argument("--compare-watermarked-left", default=None, help="离线对比左侧 watermarked JSONL 路径")
    parser.add_argument("--compare-summary-right", default=None, help="离线对比右侧 summary JSON 路径")
    parser.add_argument("--compare-details-right", default=None, help="离线对比右侧 details JSONL 路径")
    parser.add_argument("--compare-watermarked-right", default=None, help="离线对比右侧 watermarked JSONL 路径")
    parser.add_argument("--compare-output", default=None, help="离线对比输出 JSON 路径")
    # generate-negative 参数
    parser.add_argument(
        "--negative-output",
        default=None,
        help="负样本语料输出 JSONL 路径（默认从配置文件读取，或 data/negative_corpus.jsonl）",
    )
    parser.add_argument(
        "--negative-limit",
        type=int,
        default=None,
        help="只处理前 N 条 prompt（调试用，默认: 全量）",
    )
    return parser


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


COMPARE_ONLY_REQUIRED_FLAGS = (
    "compare_summary_left",
    "compare_details_left",
    "compare_summary_right",
    "compare_details_right",
    "compare_output",
)
COMPARE_ONLY_OPTIONAL_WATERMARKED_FLAGS = (
    "compare_watermarked_left",
    "compare_watermarked_right",
)


def get_config(args: argparse.Namespace) -> dict:
    cfg = getattr(args, "_config_cache", None)
    if cfg is None:
        cfg = load_config(args.config)
        setattr(args, "_config_cache", cfg)
    return cfg


def configured_extract_input(args: argparse.Namespace) -> str | None:
    return (get_config(args).get("extract") or {}).get("input_file")


def is_compare_only_mode(args: argparse.Namespace) -> bool:
    required_present = all(getattr(args, flag, None) for flag in COMPARE_ONLY_REQUIRED_FLAGS)
    if not required_present:
        return False

    optional_watermarked_flags = tuple(
        getattr(args, flag, None) for flag in COMPARE_ONLY_OPTIONAL_WATERMARKED_FLAGS
    )
    return not any(optional_watermarked_flags) or all(optional_watermarked_flags)


def should_skip_completed_phase(args: argparse.Namespace, phase: str, state: RunState) -> bool:
    if not state.is_done(phase):
        return False
    if args.force or args.eval_only or is_compare_only_mode(args):
        return False
    if phase == "extract" and has_explicit_extract_input(args):
        return False
    return True


def has_explicit_extract_input(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "input_file", None) or configured_extract_input(args))


def validate_compare_only_mode(args: argparse.Namespace) -> str | None:
    compare_flags = COMPARE_ONLY_REQUIRED_FLAGS + COMPARE_ONLY_OPTIONAL_WATERMARKED_FLAGS
    if not any(getattr(args, flag, None) for flag in compare_flags):
        return None
    if args.phase != "extract":
        return "[错误] compare-only 模式仅支持 --phase extract"
    if not is_compare_only_mode(args):
        return "[错误] compare-only 模式要求提供左右 summary/details 和 compare-output；watermarked 必须两侧同时提供或同时省略"
    return None


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


def run_phase(phase: str, args: argparse.Namespace, state: RunState) -> int:
    """分发到各阶段 runner，返回退出码。"""
    runners = {
        "encoder": run_encoder,
        "watermark": run_watermark,
        "extract": run_extract,
        "generate-negative": run_generate_negative,
    }
    return runners[phase](args, state)


def run_encoder(args: argparse.Namespace, state: RunState) -> int:
    """阶段一：训练语义编码器。"""
    import glob

    from wfcllm.encoder.config import EncoderConfig
    from wfcllm.encoder.train import main as encoder_main

    print("=== 阶段一：语义编码器预训练 ===")

    if args.eval_only:
        from wfcllm.encoder.train import evaluate_only

        default_best = str(Path(EncoderConfig().output_model_dir) / "best_model.pt")
        checkpoint = (
            args.checkpoint
            or (default_best if Path(default_best).exists() else None)
            or state.get("encoder", "checkpoint")
        )
        if not checkpoint:
            print("[错误] 未找到 checkpoint，请用 --checkpoint 指定路径", file=sys.stderr)
            return 1
        if not Path(checkpoint).exists():
            print(f"[错误] checkpoint 不存在：{checkpoint}", file=sys.stderr)
            return 1
        print(f"[评测] 使用模型: {checkpoint}")

        config = EncoderConfig()
        if args.model_name:
            config.model_name = args.model_name
        if args.embed_dim:
            config.embed_dim = args.embed_dim
        if args.no_lora:
            config.use_lora = False
        if args.no_bf16:
            config.use_bf16 = False

        try:
            evaluate_only(checkpoint, config)
        except Exception as e:
            print(f"[错误] 评测失败：{e}", file=sys.stderr)
            return 1
        return 0

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

    best_model_path = str(Path(config.output_model_dir) / "best_model.pt")
    ckpt_pattern = str(Path(config.checkpoint_dir) / "encoder_epoch*.pt")
    checkpoints = sorted(glob.glob(ckpt_pattern))
    checkpoint_path = checkpoints[-1] if checkpoints else config.checkpoint_dir

    state.mark_done("encoder", checkpoint=checkpoint_path, best_model_path=best_model_path)
    print(f"[完成] 编码器训练完毕，最优模型: {best_model_path}")
    return 0


def run_watermark(args: argparse.Namespace, state: RunState) -> int:
    """阶段二：批量生成含水印代码（基于数据集）。

    从本地 HumanEval 或 MBPP 数据集逐条加载 prompt，调用 WatermarkGenerator
    生成含水印代码，将结果写入 JSONL 文件（每行一条 JSON 记录），记录字段：
        id, dataset, prompt, generated_code,
        total_blocks, embedded_blocks, failed_blocks, fallback_blocks, embed_rate
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from wfcllm.encoder.config import EncoderConfig
    from wfcllm.encoder.model import SemanticEncoder
    from wfcllm.watermark.config import WatermarkConfig
    from wfcllm.watermark.generator import WatermarkGenerator
    from wfcllm.watermark.pipeline import WatermarkPipeline, WatermarkPipelineConfig

    print("=== 阶段二：生成时水印嵌入 ===")

    if not state.is_done("encoder"):
        print("[错误] 请先完成阶段一（encoder）", file=sys.stderr)
        return 1

    cfg = load_config(args.config)
    wm_cfg = cfg.get("watermark", {})
    dataset = args.dataset or wm_cfg.get("dataset", "humaneval")
    dataset_path = args.dataset_path or wm_cfg.get("dataset_path", "data/datasets")
    output_dir = args.output_dir or wm_cfg.get("output_dir", "data/watermarked")
    sample_limit = args.sample_limit if args.sample_limit is not None else wm_cfg.get("sample_limit")
    embed_dim = args.embed_dim or wm_cfg.get("encoder_embed_dim", 128)
    secret_key = args.secret_key or wm_cfg.get("secret_key", "")
    lm_model_path = args.lm_model_path or wm_cfg.get("lm_model_path", "")

    try:
        token_channel_config = resolve_token_channel_config(wm_cfg.get("token_channel"))
    except ValueError as exc:
        print(f"[错误] token_channel 配置无效：{exc}", file=sys.stderr)
        return 1

    if not secret_key:
        print("[错误] --secret-key 为必填参数", file=sys.stderr)
        return 1
    if not lm_model_path:
        print("[错误] --lm-model-path 为必填参数", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc_config = EncoderConfig(embed_dim=embed_dim)
    local_codet5 = Path(enc_config.local_model_dir) / "codet5-base"
    if local_codet5.exists() and (local_codet5 / "config.json").exists():
        enc_config.model_name = str(local_codet5)
        print(f"[自动] 编码器使用本地模型: {enc_config.model_name}")
    else:
        print(f"[回退] 编码器使用 HF Hub: {enc_config.model_name}")
    encoder = SemanticEncoder(config=enc_config)

    best_model_path = state.get("encoder", "best_model_path") or str(
        Path(enc_config.output_model_dir) / "best_model.pt"
    )
    encoder_checkpoint = state.get("encoder", "checkpoint")
    if Path(best_model_path).exists():
        ckpt = torch.load(best_model_path, map_location="cpu")
        encoder.load_state_dict(ckpt["model_state_dict"])
        print(f"[加载] 编码器权重来自: {best_model_path}")
    elif encoder_checkpoint and Path(encoder_checkpoint).exists():
        ckpt = torch.load(encoder_checkpoint, map_location="cpu")
        encoder.load_state_dict(ckpt["model_state_dict"])
        print(f"[加载] 编码器权重来自 checkpoint（fallback）: {encoder_checkpoint}")
    else:
        print("[警告] 未找到微调权重，使用预训练模型")
    encoder_device = wm_cfg.get("encoder_device", "cpu")
    encoder = encoder.to(encoder_device)
    encoder_tokenizer = AutoTokenizer.from_pretrained(enc_config.model_name)

    import os as _os
    _os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    lm_tokenizer = AutoTokenizer.from_pretrained(lm_model_path)
    lm_model = AutoModelForCausalLM.from_pretrained(
        lm_model_path,
        quantization_config=bnb_config,
        device_map="auto",
    )

    wm_config = WatermarkConfig(
        secret_key=secret_key,
        encoder_embed_dim=embed_dim,
        encoder_device=wm_cfg.get("encoder_device", "cpu"),
        margin_base=wm_cfg.get("margin_base", 0.1),
        margin_alpha=wm_cfg.get("margin_alpha", 0.05),
        max_retries=wm_cfg.get("max_retries", 5),
        temperature=wm_cfg.get("temperature", 0.8),
        top_p=wm_cfg.get("top_p", 0.95),
        top_k=wm_cfg.get("top_k", 50),
        max_new_tokens=wm_cfg.get("max_new_tokens", 512),
        eos_token_id=wm_cfg.get("eos_token_id"),
        enable_cascade=wm_cfg.get("enable_cascade", True),
        cascade_max_depth=wm_cfg.get("cascade_max_depth", 1),
        repetition_penalty=wm_cfg.get("repetition_penalty", 1.3),
        lsh_d=wm_cfg.get("lsh_d", 3),
        lsh_gamma=wm_cfg.get("lsh_gamma", 0.5),
        adaptive_gamma=resolve_adaptive_gamma_config(args, wm_cfg),
        token_channel=token_channel_config,
    )
    generator = WatermarkGenerator(lm_model, lm_tokenizer, encoder, encoder_tokenizer, wm_config)

    resume = args.resume if args.resume is not None else wm_cfg.get("resume")
    pipeline_config = WatermarkPipelineConfig(
        dataset=dataset,
        output_dir=output_dir,
        dataset_path=dataset_path,
        resume=resume,
        sample_limit=sample_limit,
    )
    pipeline = WatermarkPipeline(generator=generator, config=pipeline_config)

    try:
        output_path = pipeline.run()
    except Exception as e:
        print(f"[错误] 水印生成失败：{e}", file=sys.stderr)
        return 1

    state.mark_done("watermark", output_file=output_path, dataset=dataset)
    print(f"[完成] 水印数据集已保存至 {output_path}")
    return 0


def run_offline_analysis(args: argparse.Namespace) -> int:
    from wfcllm.extract.offline_analysis import (
        build_offline_regression_report,
        load_detail_artifact,
        load_summary_artifact,
        load_watermarked_artifact,
        write_offline_regression_report,
    )

    left_watermarked = (
        load_watermarked_artifact(args.compare_watermarked_left)
        if args.compare_watermarked_left
        else None
    )
    right_watermarked = (
        load_watermarked_artifact(args.compare_watermarked_right)
        if args.compare_watermarked_right
        else None
    )

    report = build_offline_regression_report(
        left_summary=load_summary_artifact(args.compare_summary_left),
        left_details=load_detail_artifact(args.compare_details_left),
        left_watermarked=left_watermarked,
        right_summary=load_summary_artifact(args.compare_summary_right),
        right_details=load_detail_artifact(args.compare_details_right),
        right_watermarked=right_watermarked,
    )
    output_path = write_offline_regression_report(args.compare_output, report)
    print(f"[完成] 离线回归报告已保存至 {output_path}")
    return 0


def run_extract(args: argparse.Namespace, state: RunState) -> int:
    """阶段三：批量检测水印（基于 JSONL 水印数据集）。

    读取阶段二输出的 JSONL 文件，对每条记录调用 WatermarkDetector.detect()，
    产出 details JSONL，并基于其重建 summary JSON。
    """
    if is_compare_only_mode(args):
        return run_offline_analysis(args)

    import torch
    from transformers import AutoTokenizer

    from wfcllm.encoder.config import EncoderConfig
    from wfcllm.encoder.model import SemanticEncoder
    from wfcllm.extract.config import ExtractConfig
    from wfcllm.extract.detector import WatermarkDetector
    from wfcllm.extract.pipeline import ExtractPipeline, ExtractPipelineConfig

    print("=== 阶段三：水印提取与验证 ===")

    cfg = get_config(args)
    ext_cfg = cfg.get("extract", {})
    secret_key = args.secret_key or ext_cfg.get("secret_key", "")
    if not secret_key:
        print("[错误] --secret-key 为必填参数", file=sys.stderr)
        return 1
    input_file = args.input_file or ext_cfg.get("input_file") or state.get("watermark", "output_file")
    if not input_file:
        print("[错误] --input-file 为必填参数（或先完成阶段二）", file=sys.stderr)
        return 1
    if not Path(input_file).exists():
        print(f"[错误] 文件不存在：{input_file}", file=sys.stderr)
        return 1
    if not state.is_done("encoder") and args.input_file is None and ext_cfg.get("input_file") is None:
        print("[错误] 请先完成阶段一（encoder）", file=sys.stderr)
        return 1

    output_dir = args.extract_output_dir or ext_cfg.get("output_dir", "data/results")
    embed_dim = args.embed_dim or ext_cfg.get("embed_dim", 128)
    fpr_threshold = args.fpr_threshold or ext_cfg.get("fpr_threshold", 3.0)
    resume = args.resume if args.resume is not None else ext_cfg.get("resume")
    adaptive_detection_config = resolve_adaptive_detection_config(args, ext_cfg)
    adaptive_gamma_config = resolve_extract_adaptive_gamma_config(args, cfg)

    try:
        token_channel_config = resolve_token_channel_config(ext_cfg.get("token_channel"))
    except ValueError as exc:
        print(f"[错误] token_channel 配置无效：{exc}", file=sys.stderr)
        return 1

    try:
        with open(input_file, encoding="utf-8") as f:
            first_line = next((line.strip() for line in f if line.strip()), "")
        first_record = json.loads(first_line) if first_line else {}
        lsh_d, lsh_gamma = resolve_extract_lsh_params(first_record, ext_cfg)
    except json.JSONDecodeError as exc:
        print(f"[错误] 输入文件首条记录 JSON 解析失败：{exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"[错误] 输入文件首条记录 LSH 参数无效：{exc}", file=sys.stderr)
        return 1

    if "watermark_params" in first_record:
        cfg_lsh_d = ext_cfg.get("lsh_d")
        cfg_lsh_gamma = ext_cfg.get("lsh_gamma")
        meta_pair = (lsh_d, lsh_gamma)
        if cfg_lsh_d is not None and cfg_lsh_gamma is not None:
            try:
                cfg_pair = (int(cfg_lsh_d), float(cfg_lsh_gamma))
            except (TypeError, ValueError):
                cfg_pair = None
            if cfg_pair is not None and cfg_pair != meta_pair:
                print(
                    f"[警告] extract 配置 LSH 参数 {cfg_pair} 与输入文件元数据 {meta_pair} 不一致；"
                    f"优先使用输入文件元数据",
                    file=sys.stderr,
                )

    enc_config = EncoderConfig(embed_dim=embed_dim)
    local_codet5 = Path(enc_config.local_model_dir) / "codet5-base"
    if local_codet5.exists() and (local_codet5 / "config.json").exists():
        enc_config.model_name = str(local_codet5)
        print(f"[自动] 编码器使用本地模型: {enc_config.model_name}")
    else:
        print(f"[回退] 编码器使用 HF Hub: {enc_config.model_name}")
    encoder = SemanticEncoder(config=enc_config)

    best_model_path = state.get("encoder", "best_model_path") or str(
        Path(enc_config.output_model_dir) / "best_model.pt"
    )
    encoder_checkpoint = state.get("encoder", "checkpoint")
    if Path(best_model_path).exists():
        ckpt = torch.load(best_model_path, map_location="cpu")
        encoder.load_state_dict(ckpt["model_state_dict"])
        print(f"[加载] 编码器权重来自: {best_model_path}")
    elif encoder_checkpoint and Path(encoder_checkpoint).exists():
        ckpt = torch.load(encoder_checkpoint, map_location="cpu")
        encoder.load_state_dict(ckpt["model_state_dict"])
        print(f"[加载] 编码器权重来自 checkpoint（fallback）: {encoder_checkpoint}")
    else:
        print("[警告] 未找到微调权重，使用预训练模型")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = encoder.to(device)
    tokenizer = AutoTokenizer.from_pretrained(enc_config.model_name)

    calibration_summary_metadata = None
    calibration_corpus_path = (
        getattr(args, "calibration_corpus", None)
        or ext_cfg.get("calibration_corpus")
    )
    if calibration_corpus_path:
        if not Path(calibration_corpus_path).exists():
            print(f"[错误] 校准语料文件不存在：{calibration_corpus_path}", file=sys.stderr)
            return 1
        from wfcllm.extract.calibrator import ThresholdCalibrator
        from wfcllm.extract.scorer import BlockScorer
        from wfcllm.watermark.keying import WatermarkKeying
        from wfcllm.watermark.lsh_space import LSHSpace
        from wfcllm.watermark.verifier import ProjectionVerifier

        fpr_target = getattr(args, "fpr", None) or ext_cfg.get("fpr", 0.01)
        block_contract_builder = build_extract_calibration_contract_builder(
            adaptive_detection_config,
            adaptive_gamma_config,
            lsh_d,
        )
        calibration_mode = "adaptive" if block_contract_builder is not None else "fixed"

        lsh_space = LSHSpace(secret_key, embed_dim, lsh_d)
        keying = WatermarkKeying(secret_key, lsh_d, lsh_gamma)
        verifier = ProjectionVerifier(encoder, tokenizer, lsh_space=lsh_space, device=device)
        scorer = BlockScorer(keying, verifier)

        import json as _calib_json

        corpus = []
        with open(calibration_corpus_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    corpus.append(_calib_json.loads(line))
        print(f"[校准] 加载负样本语料 {len(corpus)} 条，FPR 目标={fpr_target}")

        calibrator = ThresholdCalibrator(
            scorer,
            gamma=lsh_gamma,
            mode=calibration_mode,
            block_contract_builder=block_contract_builder,
        )
        calib_result = calibrator.calibrate(corpus, fpr=fpr_target)
        fpr_threshold = calib_result["fpr_threshold"]
        calibration_summary_metadata = {
            "calibration": {
                "source": str(calibration_corpus_path),
                "fpr": float(fpr_target),
                "threshold": float(fpr_threshold),
                "hypothesis_mode": calibration_mode,
                "statistic_definition": (
                    "sum(gamma_i), sum(gamma_i*(1-gamma_i))"
                    if calibration_mode == "adaptive"
                    else "m * gamma, m * gamma * (1 - gamma)"
                ),
                "decision_rule": "z_score >= threshold",
            }
        }
        print(
            f"[校准] 完成：M_r = {fpr_threshold:.4f}（FPR={fpr_target}，样本数={calib_result['n_samples']}）"
        )

    extract_config = ExtractConfig(
        secret_key=secret_key,
        embed_dim=embed_dim,
        fpr_threshold=fpr_threshold,
        lsh_d=lsh_d,
        lsh_gamma=lsh_gamma,
        adaptive_detection=adaptive_detection_config,
        adaptive_gamma=adaptive_gamma_config,
        token_channel=token_channel_config,
    )
    detector = WatermarkDetector(extract_config, encoder, tokenizer, device=device)

    pipeline_config = ExtractPipelineConfig(
        input_file=input_file,
        output_dir=output_dir,
        resume=resume,
        summary_metadata=calibration_summary_metadata,
    )
    pipeline = ExtractPipeline(detector=detector, config=pipeline_config)

    try:
        details_path = pipeline.run()
    except Exception as e:
        print(f"[错误] 检测失败：{e}", file=sys.stderr)
        return 1

    import json as _json

    summary_path = ExtractPipeline.summary_path_for_details(Path(details_path))
    summary_doc = _json.loads(summary_path.read_text(encoding="utf-8"))
    summary = summary_doc["summary"]
    print(f"\n=== 检测结果摘要 ===")
    print(f"  样本总数:     {summary_doc['meta']['total_samples']}")
    print(
        f"  水印检测率:   {summary['watermark_rate']:.1%}  "
        f"95% CI [{summary['watermark_rate_ci_95'][0]:.3f}, {summary['watermark_rate_ci_95'][1]:.3f}]"
    )
    print(f"  平均 Z 分数:  {summary['mean_z_score']:.4f} ± {summary['std_z_score']:.4f}")
    print(f"  平均 p 值:    {summary['mean_p_value']:.6f}")
    print(f"  报告已保存至: {summary_path}")

    state.mark_done(
        "extract",
        details_file=details_path,
        summary_file=str(summary_path),
        watermark_rate=summary["watermark_rate"],
    )
    return 0


def run_generate_negative(args: argparse.Namespace, state: RunState) -> int:
    """生成负样本语料：支持原生参考解或无水印 LLM 生成。

    输出 JSONL 格式与阶段二水印数据集相同（含 generated_code 字段），
    可直接作为 --calibration-corpus 传给 run.py --phase extract。
    """
    from wfcllm.extract.negative_corpus import NegativeCorpusConfig, NegativeCorpusGenerator

    print("=== 生成负样本语料 ===")

    cfg = load_config(args.config)
    neg_cfg = cfg.get("generate_negative", {})
    source_mode = neg_cfg.get("source_mode", "reference")

    lm_model_path = args.lm_model_path or neg_cfg.get("lm_model_path", "")
    if source_mode == "llm" and not lm_model_path:
        print("[错误] --lm-model-path 为必填参数", file=sys.stderr)
        return 1

    dataset = args.dataset or neg_cfg.get("dataset", "humaneval")
    dataset_path = args.dataset_path or neg_cfg.get("dataset_path", "data/datasets")
    output_path = args.negative_output or neg_cfg.get("output_path", "data/negative_corpus.jsonl")
    limit = args.negative_limit or neg_cfg.get("limit", None)

    config = NegativeCorpusConfig(
        lm_model_path=lm_model_path,
        output_path=output_path,
        dataset=dataset,
        dataset_path=dataset_path,
        max_new_tokens=neg_cfg.get("max_new_tokens", 512),
        temperature=neg_cfg.get("temperature", 0.8),
        top_p=neg_cfg.get("top_p", 0.95),
        top_k=neg_cfg.get("top_k", 50),
        device=neg_cfg.get("device", "cuda"),
        limit=limit,
        source_mode=source_mode,
    )

    try:
        generator = NegativeCorpusGenerator(config)
        out_path = generator.run()
    except Exception as e:
        print(f"[错误] 负样本生成失败：{e}", file=sys.stderr)
        return 1

    state.mark_done("generate-negative", output_file=out_path, dataset=dataset)
    print(f"[完成] 负样本语料已保存至 {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
