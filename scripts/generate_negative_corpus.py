#!/usr/bin/env python
"""生成负样本语料：用 LLM 直接生成代码（不加水印），用于 ThresholdCalibrator 校准。

输出 JSONL 格式与阶段二水印数据集相同（含 generated_code 字段），
可直接作为 --calibration-corpus 传给 run.py --phase extract，
或作为 scripts/calibrate.py 的 --input。

用法：
    python scripts/generate_negative_corpus.py \\
        --lm-model-path data/models/deepseek-coder-7b \\
        --dataset humaneval \\
        --dataset-path data/datasets \\
        --output data/negative_corpus.jsonl \\
        [--max-new-tokens 512] \\
        [--temperature 0.8] \\
        [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成负样本语料（LLM 直接生成，无水印）"
    )
    parser.add_argument(
        "--lm-model-path", required=True,
        help="代码生成 LLM 路径（同 run.py --lm-model-path）"
    )
    parser.add_argument(
        "--dataset", default="humaneval", choices=["humaneval", "mbpp"],
        help="数据集（默认: humaneval）"
    )
    parser.add_argument(
        "--dataset-path", default="data/datasets",
        help="本地数据集根目录（默认: data/datasets）"
    )
    parser.add_argument(
        "--output", required=True,
        help="输出 JSONL 文件路径（如 data/negative_corpus.jsonl）"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512,
        help="最大生成 token 数（默认: 512）"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="采样温度（默认: 0.8）"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.95,
        help="nucleus sampling top-p（默认: 0.95）"
    )
    parser.add_argument(
        "--top-k", type=int, default=50,
        help="top-k sampling（默认: 50）"
    )
    parser.add_argument(
        "--device", default="cuda",
        help="推理设备（默认: cuda）"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="只处理前 N 条 prompt（调试用，默认: 全量）"
    )
    return parser.parse_args()


def _load_prompts(dataset: str, dataset_path: str) -> list[dict]:
    """加载数据集 prompt，返回 [{"id": ..., "prompt": ...}]。"""
    from datasets import load_dataset

    path = str(Path(dataset_path) / dataset)

    if dataset == "humaneval":
        ds = load_dataset(
            "openai/openai_humaneval",
            cache_dir=path,
            download_mode="reuse_cache_if_exists",
        )
        prompts = []
        for split in ds:
            for item in ds[split]:
                prompts.append({"id": item["task_id"], "prompt": item["prompt"]})
        return prompts

    # mbpp
    ds = load_dataset(
        "google-research-datasets/mbpp", "full",
        cache_dir=path,
        download_mode="reuse_cache_if_exists",
    )
    prompts = []
    for split in ds:
        for item in ds[split]:
            prompts.append({"id": f"mbpp/{item['task_id']}", "prompt": item["text"]})
    return prompts


def _generate(model, tokenizer, prompt: str, args: argparse.Namespace, device: str) -> str:
    """用 LLM 直接生成代码，不加水印。"""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    # 去掉 prompt 部分，只保留生成内容
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main() -> None:
    args = _parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[警告] CUDA 不可用，回退到 CPU", file=sys.stderr)
        device = "cpu"

    print(f"加载模型 {args.lm_model_path} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.lm_model_path)

    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.lm_model_path,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.lm_model_path,
            torch_dtype=torch.float32,
        ).to(device)
    model.eval()

    print(f"加载数据集 {args.dataset} from {args.dataset_path} ...", file=sys.stderr)
    prompts = _load_prompts(args.dataset, args.dataset_path)
    if args.limit:
        prompts = prompts[: args.limit]
    print(f"  共 {len(prompts)} 条 prompt", file=sys.stderr)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(prompts):
            try:
                code = _generate(model, tokenizer, item["prompt"], args, device)
            except Exception as e:
                print(f"[警告] {item['id']} 生成失败：{e}", file=sys.stderr)
                code = ""

            record = {
                "id": item["id"],
                "dataset": args.dataset,
                "prompt": item["prompt"],
                "generated_code": code,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(
                f"  [{i + 1}/{len(prompts)}] {item['id']} | "
                f"tokens: {len(code.split())}",
                file=sys.stderr,
            )

    print(f"\n完成，输出至: {out_path}（{len(prompts)} 条）", file=sys.stderr)


if __name__ == "__main__":
    main()
