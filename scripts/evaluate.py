#!/usr/bin/env python
"""Unified offline evaluation entry point.

Three subcommands:
  exec        compute pass@k (and friends) over JSONL candidate rows
  detection   build the offline regression report from saved summary + details
  dual        run the dual-channel end-to-end harness (semantic / lexical / dual)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.evaluation.code_execution import (  # noqa: E402
    annotate_correctness_from_references,
    compute_pass_at_k,
    load_jsonl_records,
)
from wfcllm.evaluation.detection_report import (  # noqa: E402
    build_offline_regression_report,
    load_detail_artifact,
    load_summary_artifact,
    load_watermarked_artifact,
    write_offline_regression_report,
)
from wfcllm.evaluation import dual_channel as dual_channel_module  # noqa: E402


def _cmd_exec(args: argparse.Namespace) -> int:
    records: list[dict] = []
    for path in args.inputs:
        records.extend(load_jsonl_records(path))

    if args.reference is not None:
        reference_records = load_jsonl_records(args.reference)
        records = annotate_correctness_from_references(records, reference_records)

    if args.metric == "pass_at_k":
        k = args.k
    elif args.metric == "pass_at_1":
        k = 1
    elif args.metric == "pass_at_10":
        k = 10
    else:
        raise ValueError(f"unsupported metric: {args.metric}")

    value = compute_pass_at_k(records, k=k)
    print(json.dumps({
        "metric": args.metric,
        "k": k,
        "value": value,
        "sample_count": len(records),
    }, ensure_ascii=False, indent=2))
    return 0


def _cmd_detection(args: argparse.Namespace) -> int:
    left_watermarked = (
        load_watermarked_artifact(args.left_watermarked) if args.left_watermarked else None
    )
    right_watermarked = (
        load_watermarked_artifact(args.right_watermarked) if args.right_watermarked else None
    )
    report = build_offline_regression_report(
        left_summary=load_summary_artifact(args.left_summary),
        left_details=load_detail_artifact(args.left_details),
        left_watermarked=left_watermarked,
        right_summary=load_summary_artifact(args.right_summary),
        right_details=load_detail_artifact(args.right_details),
        right_watermarked=right_watermarked,
    )
    output_path = write_offline_regression_report(args.output, report)
    print(f"[完成] 离线回归报告已保存至 {output_path}")
    return 0


def _cmd_dual(args: argparse.Namespace) -> int:
    result = dual_channel_module.run_evaluation(
        dataset=args.dataset,
        config_path=args.config,
        output_dir=args.output_dir,
        candidate_count=args.num_candidates,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from wfcllm.evaluation.benchmark import BenchmarkConfig, BenchmarkRunner

    config = BenchmarkConfig(
        dataset=args.dataset,
        config_path=args.config,
        dataset_path=args.dataset_path,
        num_candidates=args.num_candidates,
        timeout_per_test=args.timeout,
        watermarked_dirs=args.watermarked_dirs,
        positive_details=args.positive_details,
        negative_details=args.negative_details,
        auto_generate=args.auto_generate,
        negative_corpus=args.negative_corpus,
        output_dir=args.output_dir,
        min_blocks=args.min_blocks,
    )
    runner = BenchmarkRunner(config)
    report = runner.run()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WFCLLM unified offline evaluation entry point.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    exec_parser = subparsers.add_parser("exec", help="pass@k from JSONL candidate rows")
    exec_parser.add_argument("inputs", nargs="+", help="one or more JSONL files of candidates")
    exec_parser.add_argument(
        "--metric",
        choices=["pass_at_1", "pass_at_10", "pass_at_k"],
        default="pass_at_1",
    )
    exec_parser.add_argument("--k", type=int, default=1, help="k for --metric pass_at_k")
    exec_parser.add_argument(
        "--reference",
        default=None,
        help="optional reference JSONL; if given, candidate rows are re-annotated for correctness",
    )
    exec_parser.set_defaults(func=_cmd_exec)

    det_parser = subparsers.add_parser(
        "detection",
        help="offline regression report from saved summary + details artifacts",
    )
    det_parser.add_argument("--left-summary", required=True)
    det_parser.add_argument("--left-details", required=True)
    det_parser.add_argument("--right-summary", required=True)
    det_parser.add_argument("--right-details", required=True)
    det_parser.add_argument("--left-watermarked", default=None)
    det_parser.add_argument("--right-watermarked", default=None)
    det_parser.add_argument("--output", required=True, help="report JSON output path")
    det_parser.set_defaults(func=_cmd_detection)

    dual_parser = subparsers.add_parser(
        "dual",
        help="end-to-end dual-channel evaluation harness",
    )
    dual_parser.add_argument("--dataset", default="humaneval", choices=["humaneval", "mbpp"])
    dual_parser.add_argument("--config", default="configs/base_config.json")
    dual_parser.add_argument("--output-dir", default="data/eval/dual_channel")
    dual_parser.add_argument("--num-candidates", type=int, default=10)
    dual_parser.set_defaults(func=_cmd_dual)

    bench_parser = subparsers.add_parser(
        "bench",
        help="compute Pass@1, Pass@10, AUROC from watermarked candidates + negative corpus",
    )
    bench_parser.add_argument(
        "--dataset", required=True, choices=["humaneval", "mbpp"],
    )
    bench_parser.add_argument("--config", default="configs/base_config.json")
    bench_parser.add_argument("--dataset-path", default="data/datasets")
    bench_parser.add_argument(
        "--watermarked-dirs", nargs="+", default=None,
        help="directories containing watermarked candidate JSONL files",
    )
    bench_parser.add_argument("--positive-details", default=None)
    bench_parser.add_argument("--negative-details", default=None)
    bench_parser.add_argument("--negative-corpus", default=None)
    bench_parser.add_argument("--auto-generate", action="store_true")
    bench_parser.add_argument("--num-candidates", type=int, default=10)
    bench_parser.add_argument("--timeout", type=float, default=5.0)
    bench_parser.add_argument("--output-dir", default="data/eval/benchmark")
    bench_parser.add_argument(
        "--min-blocks", type=int, default=0,
        help="skip records with total_blocks < MIN_BLOCKS",
    )
    bench_parser.set_defaults(func=_cmd_bench)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
