#!/usr/bin/env python
"""生成负样本语料：用 LLM 直接生成代码（不加水印），用于 ThresholdCalibrator 校准。

输出 JSONL 格式与阶段二水印数据集相同（含 generated_code 字段），
可直接作为 --calibration-corpus 传给 run.py --phase extract。

推荐使用 run.py 入口：
    python run.py --phase generate-negative \\
        --lm-model-path data/models/deepseek-coder-7b-base \\
        --dataset humaneval \\
        --negative-output data/negative_corpus.jsonl

也可直接调用本脚本（仅保留作向后兼容）：
    python scripts/generate_negative_corpus.py \\
        --lm-model-path data/models/deepseek-coder-7b-base \\
        --dataset humaneval \\
        --dataset-path data/datasets \\
        --output data/negative_corpus.jsonl
"""
from __future__ import annotations

import argparse


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成负样本语料（LLM 直接生成，无水印）"
    )
    parser.add_argument("--lm-model-path", required=True, help="代码生成 LLM 路径")
    parser.add_argument(
        "--dataset", default="humaneval", choices=["humaneval", "mbpp"],
        help="数据集（默认: humaneval）",
    )
    parser.add_argument(
        "--dataset-path", default="data/datasets",
        help="本地数据集根目录（默认: data/datasets）",
    )
    parser.add_argument("--output", required=True, help="输出 JSONL 文件路径")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    from wfcllm.extract.negative_corpus import NegativeCorpusConfig, NegativeCorpusGenerator

    config = NegativeCorpusConfig(
        lm_model_path=args.lm_model_path,
        output_path=args.output,
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        device=args.device,
        limit=args.limit,
    )
    generator = NegativeCorpusGenerator(config)
    generator.run()


if __name__ == "__main__":
    main()
