"""Per-phase runner functions invoked by PhaseOrchestrator.

Lifted from run.py (Phase 1 refactor). Functions are CLI-bound:
they consume argparse.Namespace, build phase configs, invoke pipelines,
and update RunStateManager.

Future refactor phases (Phase 5: pretrain, Phase 8: generator split) will
further decompose these into their phase packages. For Phase 1 they stay
together to avoid scope creep.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from wfcllm.cli.arguments import DEFAULT_CONFIG_FILE
from wfcllm.cli.config_resolver import (
    load_config,
    resolve_extract_lsh_params,
    resolve_adaptive_gamma_config,
    resolve_extract_adaptive_gamma_config,
    resolve_token_channel_config,
    build_extract_calibration_contract_builder,
    resolve_adaptive_detection_config,
)
from wfcllm.orchestration.state import RunStateManager

# ---------------------------------------------------------------------------
# Compare-only mode flag names
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Config helper utilities (used by multiple runners)
# ---------------------------------------------------------------------------


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


def has_explicit_extract_input(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "input_file", None) or configured_extract_input(args))


# ---------------------------------------------------------------------------
# Per-phase runner functions
# ---------------------------------------------------------------------------


def run_phase(phase: str, args: argparse.Namespace, state: RunStateManager) -> int:
    """分发到各阶段 runner，返回退出码。"""
    runners = {
        "encoder": run_encoder,
        "watermark": run_watermark,
        "extract": run_extract,
        "generate-negative": run_generate_negative,
        "token-channel-train": run_token_channel_train,
    }
    return runners[phase](args, state)


def run_encoder(args: argparse.Namespace, state: RunStateManager) -> int:
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


def run_watermark(args: argparse.Namespace, state: RunStateManager) -> int:
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
        token_channel_config = resolve_token_channel_config(wm_cfg.get("token_channel"), args)
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


def run_extract(args: argparse.Namespace, state: RunStateManager) -> int:
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
    min_blocks = args.min_blocks if args.min_blocks is not None else ext_cfg.get("min_blocks", 2)
    resume = args.resume if args.resume is not None else ext_cfg.get("resume")
    adaptive_detection_config = resolve_adaptive_detection_config(args, ext_cfg)
    adaptive_gamma_config = resolve_extract_adaptive_gamma_config(args, cfg)

    try:
        token_channel_config = resolve_token_channel_config(ext_cfg.get("token_channel"), args)
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

    # Load LM tokenizer for token-channel compatibility if enabled
    lm_tokenizer = None
    if token_channel_config.enabled:
        try:
            artifact_metadata_path = Path(token_channel_config.model_path) / "metadata.json"
            if artifact_metadata_path.exists():
                with open(artifact_metadata_path, encoding="utf-8") as f:
                    artifact_metadata = json.load(f)
                lm_tokenizer_name = artifact_metadata.get("tokenizer_name")
                if lm_tokenizer_name:
                    lm_tokenizer = AutoTokenizer.from_pretrained(lm_tokenizer_name)
                    print(f"[加载] token-channel LM tokenizer: {lm_tokenizer_name}")
        except Exception as e:
            print(f"[警告] 无法加载 token-channel LM tokenizer: {e}", file=sys.stderr)


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
        min_blocks=min_blocks,
        adaptive_detection=adaptive_detection_config,
        adaptive_gamma=adaptive_gamma_config,
        token_channel=token_channel_config,
    )
    detector = WatermarkDetector(extract_config, encoder, tokenizer, device=device, lm_tokenizer=lm_tokenizer)

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


def run_generate_negative(args: argparse.Namespace, state: RunStateManager) -> int:
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


def resolve_token_channel_train_config(args: argparse.Namespace) -> dict[str, object]:
    """Merge token-channel-train config with CLI overrides."""

    default_cfg = load_config(DEFAULT_CONFIG_FILE)
    default_section = default_cfg.get("token_channel_train", {})
    if default_section is None:
        train_cfg: dict[str, object] = {}
    elif isinstance(default_section, dict):
        train_cfg = dict(default_section)
    else:
        raise ValueError("token_channel_train must be a JSON object")

    cfg = get_config(args)
    raw_section = cfg.get("token_channel_train", {})
    if raw_section is None:
        configured_train_cfg: dict[str, object] = {}
    elif isinstance(raw_section, dict):
        configured_train_cfg = dict(raw_section)
    else:
        raise ValueError("token_channel_train must be a JSON object")
    train_cfg.update(configured_train_cfg)

    overrides = {
        "dataset": getattr(args, "dataset", None),
        "dataset_path": getattr(args, "dataset_path", None),
        "lm_model_path": getattr(args, "lm_model_path", None),
        "model_path": getattr(args, "token_channel_model_path", None),
        "cache_path": getattr(args, "token_channel_cache_path", None),
        "context_width": getattr(args, "token_channel_context_width", None),
        "hidden_size": getattr(args, "token_channel_hidden_size", None),
        "batch_size": getattr(args, "token_channel_batch_size", None),
        "epochs": getattr(args, "token_channel_epochs", None),
        "lr": getattr(args, "token_channel_lr", None),
        "entropy_threshold": getattr(args, "token_channel_entropy_threshold", None),
        "diversity_threshold": getattr(args, "token_channel_diversity_threshold", None),
        "split_ratio": getattr(args, "token_channel_split_ratio", None),
        "seed": getattr(args, "token_channel_seed", None),
        "teacher_batch_size": getattr(args, "token_channel_teacher_batch_size", None),
        "max_variants": getattr(args, "token_channel_max_variants", None),
        "top_k_logits": getattr(args, "token_channel_top_k_logits", None),
        "resume_training": getattr(args, "token_channel_resume_training", None),
    }
    for key, value in overrides.items():
        if value is not None:
            train_cfg[key] = value

    dataset = train_cfg.get("dataset")
    if dataset is not None and dataset not in {"humaneval", "mbpp"}:
        raise ValueError("dataset must be one of: humaneval, mbpp")

    positive_int_fields = (
        "context_width",
        "hidden_size",
        "batch_size",
        "epochs",
    )
    for field_name in positive_int_fields:
        field_value = train_cfg.get(field_name)
        if field_value is not None and int(field_value) <= 0:
            raise ValueError(f"{field_name} must be > 0")

    lr = train_cfg.get("lr")
    if lr is not None and float(lr) <= 0:
        raise ValueError("lr must be > 0")

    split_ratio = train_cfg.get("split_ratio")
    if split_ratio is not None and not 0 < float(split_ratio) < 1:
        raise ValueError("split_ratio must be within (0, 1)")

    diversity_threshold = train_cfg.get("diversity_threshold")
    if diversity_threshold is not None and int(diversity_threshold) < 1:
        raise ValueError("diversity_threshold must be >= 1")

    entropy_threshold = train_cfg.get("entropy_threshold")
    if entropy_threshold is not None and float(entropy_threshold) < 0:
        raise ValueError("entropy_threshold must be >= 0")

    return train_cfg


def validate_token_channel_train_config(train_cfg: dict[str, object]) -> str | None:
    """Validate required user-facing inputs before workflow construction."""

    if not train_cfg.get("dataset"):
        return "[错误] token-channel-train 需要提供 dataset（可通过配置文件或 --dataset 指定）"
    if not train_cfg.get("lm_model_path"):
        return "[错误] token-channel-train 需要提供 lm_model_path（可通过配置文件或 --lm-model-path 指定）"
    return None


def run_token_channel_train(args: argparse.Namespace, state: RunStateManager) -> int:
    """Run the token-channel training workflow."""

    from wfcllm.watermark.token_channel.train_workflow import (
        TokenChannelTrainWorkflowConfig,
    )
    from wfcllm.watermark.token_channel.train_workflow import (
        format_token_channel_train_workflow_summary,
    )
    from wfcllm.watermark.token_channel.train_workflow import (
        run_token_channel_train_workflow,
    )

    print("=== 可选阶段：Token Channel 训练 ===")

    try:
        train_cfg = resolve_token_channel_train_config(args)
    except ValueError as exc:
        print(f"[错误] token-channel-train 配置无效：{exc}", file=sys.stderr)
        return 1

    validation_error = validate_token_channel_train_config(train_cfg)
    if validation_error is not None:
        print(validation_error, file=sys.stderr)
        return 1

    try:
        workflow_config = TokenChannelTrainWorkflowConfig(**train_cfg)
    except (TypeError, ValueError) as exc:
        print(f"[错误] token-channel-train 配置无效：{exc}", file=sys.stderr)
        return 1

    if workflow_config.cache_path.exists():
        print(f"[提示] overwrite existing cache: {workflow_config.cache_path}")
    if workflow_config.model_path.exists():
        print(f"[提示] overwrite existing model artifacts: {workflow_config.model_path}")

    try:
        summary = run_token_channel_train_workflow(workflow_config)
    except Exception as exc:
        print(f"[错误] token-channel-train 运行失败：{exc}", file=sys.stderr)
        return 1

    for line in format_token_channel_train_workflow_summary(summary):
        print(line)

    state.mark_done(
        "token-channel-train",
        dataset=summary.dataset,
        cache_path=str(summary.cache_path),
        artifact_dir=str(summary.artifact_dir),
    )
    return 0
