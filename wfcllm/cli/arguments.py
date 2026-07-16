"""Argparse definitions for the WFCLLM run.py entry point.

Lifted verbatim from run.py:343-670 (Phase 1 refactor). No behavior changes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from wfcllm.orchestration.state import ALL_PHASES
from wfcllm.cli.config_resolver import parse_optional_bool

DEFAULT_CONFIG_FILE = Path("configs/base_config.json")


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
        help="运行指定阶段（不指定则运行新版 WFCLLM 主流程）",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="允许显式运行归档的 legacy phase",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="WFCLLM run id；不传则按时间和 method 名生成",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="WFCLLM run directory；优先于 config artifacts.run_root",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="detect 阶段 official final_code.jsonl 输入",
    )
    parser.add_argument(
        "--negative-input",
        default=None,
        help="calibrate 阶段 reference negative final_code.jsonl",
    )
    parser.add_argument(
        "--calibration",
        default=None,
        help="detect 阶段 calibration artifact 路径",
    )
    parser.add_argument(
        "--positive-details",
        default=None,
        help="report 阶段 positive details JSONL",
    )
    parser.add_argument(
        "--negative-details",
        default=None,
        help="report 阶段 negative details JSONL",
    )
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="允许运行 diagnostic-selector phase",
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
        help="legacy watermark/extract 阶段使用：样本级断点恢复 latest 或已有 JSONL 文件路径",
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
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["encoder", "lexical"],
        default=None,
        help="legacy pretrain 阶段使用：选择运行哪些 stage，默认两个都跑",
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
    # Legacy watermark 参数
    parser.add_argument("--secret-key", default=None, help="legacy watermark/extract 阶段使用：水印密钥")
    parser.add_argument(
        "--secret-key-file", default=None,
        help="gated 方法部署密钥文件（不得写入 public config）",
    )
    parser.add_argument(
        "--secret-key-env", default=None,
        help="gated 方法部署密钥环境变量名",
    )
    parser.add_argument("--training-key-bank-file", default=None)
    parser.add_argument("--training-key-bank-env", default=None)
    parser.add_argument("--holdout-key-bank-file", default=None)
    parser.add_argument("--holdout-key-bank-env", default=None)
    parser.add_argument("--gate-source-manifest", default=None)
    parser.add_argument("--pilot-feasibility", default=None)
    parser.add_argument("--lm-model-path", default=None, help="legacy watermark 阶段使用：代码生成 LLM 路径")
    parser.add_argument(
        "--dataset",
        default=None,
        choices=["humaneval", "mbpp"],
        help="legacy watermark 阶段使用：水印嵌入数据集（humaneval 或 mbpp，默认: humaneval）",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="legacy watermark 阶段使用：本地数据集根目录（默认: data/datasets）",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="legacy watermark 阶段使用：水印 JSONL 输出目录（默认: data/watermarked）",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="legacy watermark 阶段使用：仅处理前 N 条 prompts（调试/子集验证用）",
    )
    parser.add_argument(
        "--sample-offset",
        type=int,
        default=None,
        help="legacy watermark 阶段使用：从第 N 条 prompt 开始处理（并行分片用）",
    )
    parser.add_argument(
        "--gamma-strategy",
        choices=["piecewise_quantile"],
        default=None,
        help="legacy adaptive-gamma 阶段使用：自适应 gamma 调度策略（默认从配置文件读取）",
    )
    parser.add_argument(
        "--entropy-profile",
        default=None,
        help="legacy adaptive-gamma 阶段使用：entropy profile JSON 路径",
    )
    parser.add_argument(
        "--profile-id",
        default=None,
        help="legacy watermark 阶段使用：输出 metadata 时使用的 entropy profile 标识",
    )
    # build-entropy-profile phase 参数（Phase 3）
    parser.add_argument(
        "--build-profile-input-log",
        default=None,
        help="watermark debug 日志路径（用于 build-entropy-profile 阶段）",
    )
    parser.add_argument(
        "--build-profile-output",
        default=None,
        help="entropy profile JSON 输出路径（用于 build-entropy-profile 阶段）",
    )
    parser.add_argument(
        "--build-profile-language",
        default=None,
        help="entropy profile 语言标签",
    )
    parser.add_argument(
        "--build-profile-model-family",
        default=None,
        help="entropy profile model-family 标签",
    )
    parser.add_argument(
        "--build-profile-strategy",
        default=None,
        help="adaptive gamma 策略标签（默认 piecewise_quantile）",
    )
    parser.add_argument(
        "--build-profile-id",
        default=None,
        help="profile_id 字段的可选标识",
    )
    parser.add_argument(
        "--token-channel-enabled",
        type=parse_optional_bool,
        default=None,
        help="是否启用 token 级词法通道（true/false）",
    )
    parser.add_argument(
        "--token-channel-mode",
        choices=["semantic-only", "lexical-only", "dual-channel"],
        default=None,
        help="token 通道运行模式",
    )
    parser.add_argument(
        "--token-channel-model-path",
        default=None,
        help="token 通道模型产物路径",
    )
    parser.add_argument(
        "--token-channel-cache-path",
        default=None,
        help="token 通道训练缓存路径",
    )
    parser.add_argument(
        "--token-channel-context-width",
        type=int,
        default=None,
        help="token 通道上下文宽度",
    )
    parser.add_argument(
        "--token-channel-hidden-size",
        type=int,
        default=None,
        help="token 通道训练隐藏层维度",
    )
    parser.add_argument(
        "--token-channel-batch-size",
        type=int,
        default=None,
        help="token 通道训练批大小",
    )
    parser.add_argument(
        "--token-channel-epochs",
        type=int,
        default=None,
        help="token 通道训练轮数",
    )
    parser.add_argument(
        "--token-channel-lr",
        type=float,
        default=None,
        help="token 通道训练学习率",
    )
    parser.add_argument(
        "--token-channel-entropy-threshold",
        type=float,
        default=None,
        help="token 通道训练样本熵阈值",
    )
    parser.add_argument(
        "--token-channel-diversity-threshold",
        type=int,
        default=None,
        help="token 通道训练样本多样性阈值",
    )
    parser.add_argument(
        "--token-channel-split-ratio",
        type=float,
        default=None,
        help="token 通道训练集划分比例",
    )
    parser.add_argument(
        "--token-channel-seed",
        type=int,
        default=None,
        help="token 通道训练随机种子",
    )
    parser.add_argument(
        "--token-channel-teacher-batch-size",
        type=int,
        default=None,
        help="token 通道 teacher 模型批量推理大小（默认 16）",
    )
    parser.add_argument(
        "--token-channel-max-variants",
        type=int,
        default=None,
        help="每个样本生成的最大变体数量（包括原始样本，默认无限制）",
    )
    parser.add_argument(
        "--token-channel-top-k-logits",
        type=int,
        default=None,
        help="只保存 top-k logits 以减少缓存大小（默认 100，None 表示保存全部）",
    )
    parser.add_argument(
        "--token-channel-resume-training",
        type=parse_optional_bool,
        default=None,
        help="是否从已有 checkpoint 继续训练（默认 true）",
    )
    parser.add_argument(
        "--token-channel-switch-threshold",
        type=float,
        default=None,
        help="token 通道 gate 阈值",
    )
    parser.add_argument(
        "--token-channel-delta",
        type=float,
        default=None,
        help="token 通道 green 集偏置强度",
    )
    parser.add_argument(
        "--token-channel-ignore-repeated-ngrams",
        type=parse_optional_bool,
        default=None,
        help="提取阶段是否忽略重复 n-gram 位置（true/false）",
    )
    parser.add_argument(
        "--token-channel-ignore-repeated-prefixes",
        type=parse_optional_bool,
        default=None,
        help="提取阶段是否忽略重复 prefix 位置（true/false）",
    )
    parser.add_argument(
        "--token-channel-debug-mode",
        type=parse_optional_bool,
        default=None,
        help="是否启用 token 通道调试模式（true/false）",
    )
    parser.add_argument(
        "--token-channel-lexical-min-block-tokens",
        type=int,
        default=None,
        help="短 block 关闭规则的最小 token 数",
    )
    parser.add_argument(
        "--token-channel-lexical-retry-decay-start",
        type=int,
        default=None,
        help="词法通道重试衰减起始轮次",
    )
    parser.add_argument(
        "--token-channel-lexical-retry-disable-after",
        type=int,
        default=None,
        help="词法通道在多次重试后关闭的轮次",
    )
    parser.add_argument(
        "--token-channel-lexical-gate-probe-tokens",
        type=int,
        default=None,
        help="词法 gate 探针窗口大小",
    )
    parser.add_argument(
        "--token-channel-lexical-gate-min-fraction",
        type=float,
        default=None,
        help="词法 gate 最低命中比例",
    )
    parser.add_argument(
        "--token-channel-joint-semantic-weight",
        type=float,
        default=None,
        help="联合检测中的语义通道权重",
    )
    parser.add_argument(
        "--token-channel-joint-lexical-weight",
        type=float,
        default=None,
        help="联合检测中的词法通道权重",
    )
    parser.add_argument(
        "--token-channel-lexical-full-weight-min-positions",
        type=int,
        default=None,
        help="词法联合权重达到满额所需的最少计分位置数",
    )
    parser.add_argument(
        "--token-channel-joint-threshold",
        type=float,
        default=None,
        help="联合检测判决阈值",
    )
    # Legacy extract 参数
    parser.add_argument(
        "--input-file",
        default=None,
        help="legacy extract 阶段使用：待检测的水印 JSONL 文件路径",
    )
    parser.add_argument(
        "--extract-output-dir",
        default=None,
        help="legacy extract 阶段使用：检测报告输出目录（默认: data/results）",
    )
    parser.add_argument(
        "--fpr-threshold",
        type=float,
        default=None,
        help="legacy extract 阶段使用：FPR 阈值 M_r（默认: 3.0，需通过校准脚本生成）",
    )
    parser.add_argument(
        "--min-blocks",
        type=int,
        default=None,
        help="legacy extract 阶段使用：检测时最小块数阈值，低于此值的样本将被跳过（默认: 2）",
    )
    parser.add_argument(
        "--calibration-corpus",
        default=None,
        help="legacy extract 阶段使用：负样本校准语料 JSONL 路径",
    )
    parser.add_argument(
        "--fpr",
        type=float,
        default=None,
        help="legacy extract 阶段使用：校准目标 FPR",
    )
    parser.add_argument(
        "--adaptive-detection-mode",
        choices=["fixed", "prefer-adaptive", "require-adaptive"],
        default=None,
        help="legacy extract 阶段使用：adaptive hypothesis 的模式（默认从配置文件读取）",
    )
    parser.add_argument(
        "--strict-contract",
        action="store_true",
        help="legacy extract 阶段使用：强制启用 block contract 检查并在结构不匹配时严格失败",
    )
    parser.add_argument("--compare-summary-left", default=None, help="legacy extract compare-only：左侧 summary JSON 路径")
    parser.add_argument("--compare-details-left", default=None, help="legacy extract compare-only：左侧 details JSONL 路径")
    parser.add_argument("--compare-watermarked-left", default=None, help="legacy extract compare-only：左侧 watermarked JSONL 路径")
    parser.add_argument("--compare-summary-right", default=None, help="legacy extract compare-only：右侧 summary JSON 路径")
    parser.add_argument("--compare-details-right", default=None, help="legacy extract compare-only：右侧 details JSONL 路径")
    parser.add_argument("--compare-watermarked-right", default=None, help="legacy extract compare-only：右侧 watermarked JSONL 路径")
    parser.add_argument("--compare-output", default=None, help="legacy extract compare-only：离线对比输出 JSON 路径")
    # Legacy generate-negative 参数
    parser.add_argument(
        "--negative-output",
        default=None,
        help="legacy generate-negative 阶段使用：负样本语料输出 JSONL 路径",
    )
    parser.add_argument(
        "--negative-limit",
        type=int,
        default=None,
        help="legacy generate-negative 阶段使用：只处理前 N 条 prompt（调试用，默认: 全量）",
    )
    return parser
