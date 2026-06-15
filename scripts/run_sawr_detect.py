#!/usr/bin/env python
"""Run SAWR final-code-only black-box detection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.sawr.detect.calibration import (  # noqa: E402
    load_calibration_artifact,
    write_calibration_artifact,
)
from wfcllm.sawr.detect.config import SawrDetectionConfig  # noqa: E402
from wfcllm.sawr.detect.metrics import (  # noqa: E402
    build_detection_report,
    split_records_by_task,
    write_detection_report,
)
from wfcllm.sawr.detect.pipeline import (  # noqa: E402
    SawrDetectionPipeline,
    load_jsonl_records,
)
from wfcllm.sawr.detect.scoring import load_sawr_window_scorer  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SAWR final-code-only black-box detection.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Build a SAWR detector calibration artifact from negative final-code rows.",
    )
    _add_detector_args(calibrate_parser)
    calibrate_parser.add_argument("--input", required=True)
    calibrate_parser.add_argument("--output", required=True)
    calibrate_parser.set_defaults(handler=_cmd_calibrate)

    detect_parser = subparsers.add_parser(
        "detect",
        help="Score final-code rows with a calibrated SAWR detector.",
    )
    _add_detector_args(detect_parser)
    detect_parser.add_argument("--input", required=True)
    detect_parser.add_argument("--calibration", required=True)
    detect_parser.add_argument("--output", required=True)
    detect_parser.set_defaults(handler=_cmd_detect)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate positive and negative SAWR detector detail rows.",
    )
    evaluate_parser.add_argument("--positive-details", required=True)
    evaluate_parser.add_argument("--negative-details", required=True)
    evaluate_parser.add_argument("--output", required=True)
    evaluate_parser.set_defaults(handler=_cmd_evaluate)

    split_parser = subparsers.add_parser(
        "split",
        help="Split JSONL rows into task-disjoint dev, calibration, and test sets.",
    )
    split_parser.add_argument("--input", required=True)
    split_parser.add_argument("--output-dir", required=True)
    split_parser.add_argument("--dev-ratio", type=float, required=True)
    split_parser.add_argument("--calibration-ratio", type=float, required=True)
    split_parser.add_argument("--seed", type=int, required=True)
    split_parser.set_defaults(handler=_cmd_split)

    return parser


def _add_detector_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--secret-key", required=True)
    parser.add_argument("--encoder-model-path", required=True)
    parser.add_argument("--encoder-checkpoint-path", default=None)
    parser.add_argument("--encoder-embed-dim", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoder-use-lora", action="store_true")
    parser.add_argument("--encoder-use-bf16", action="store_true")
    parser.add_argument("--lsh-whitening-path", default=None)
    parser.add_argument("--lsh-d", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.75)
    parser.add_argument("--semantic-margin", type=float, default=0.0)
    parser.add_argument("--max-group-statements", type=int, default=2)
    parser.add_argument("--min-scoreable-contexts", type=int, default=1)
    parser.add_argument("--min-proxy-windows", type=int, default=2)
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--use-ordinal-keying", action="store_true")
    parser.add_argument(
        "--evidence-mode",
        choices=["hit_plus_margin", "hit_only", "margin_only"],
        default="hit_plus_margin",
    )
    parser.add_argument(
        "--statistic",
        choices=[
            "calibrated_context_max",
            "raw_context_max",
            "context_mean_window_evidence",
        ],
        default="calibrated_context_max",
    )
    parser.add_argument("--no-structure-aware", action="store_true")


def _build_pipeline(args: argparse.Namespace) -> SawrDetectionPipeline:
    config = SawrDetectionConfig(
        secret_key=args.secret_key,
        lsh_d=args.lsh_d,
        gamma=args.gamma,
        semantic_margin=args.semantic_margin,
        max_group_statements=args.max_group_statements,
        min_scoreable_contexts=args.min_scoreable_contexts,
        min_proxy_windows=args.min_proxy_windows,
        target_fpr=args.target_fpr,
        use_ordinal_keying=args.use_ordinal_keying,
        evidence_mode=args.evidence_mode,
        statistic=args.statistic,
        structure_aware=not args.no_structure_aware,
    )
    scorer = load_sawr_window_scorer(
        config=config,
        encoder_model_path=args.encoder_model_path,
        encoder_checkpoint_path=args.encoder_checkpoint_path,
        embed_dim=args.encoder_embed_dim,
        device=args.device,
        use_lora=args.encoder_use_lora,
        use_bf16=args.encoder_use_bf16,
        whitening_path=args.lsh_whitening_path,
    )
    return SawrDetectionPipeline(config=config, scorer=scorer)


def _cmd_calibrate(args: argparse.Namespace) -> int:
    records = load_jsonl_records(args.input)
    pipeline = _build_pipeline(args)
    artifact = pipeline.calibrate(records)
    output_path = Path(args.output)
    write_calibration_artifact(output_path, artifact)
    print(f"[完成] SAWR calibration artifact saved to {output_path}", file=sys.stderr)
    return 0


def _cmd_detect(args: argparse.Namespace) -> int:
    artifact = load_calibration_artifact(args.calibration)
    records = load_jsonl_records(args.input)
    pipeline = _build_pipeline(args)
    output_path = Path(args.output)
    pipeline.detect_to_jsonl(records, artifact=artifact, output_path=output_path)
    print(f"[完成] SAWR detection details saved to {output_path}", file=sys.stderr)
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    positive_rows = load_jsonl_records(args.positive_details)
    negative_rows = load_jsonl_records(args.negative_details)
    report = build_detection_report(positive_rows, negative_rows)
    output_path = Path(args.output)
    write_detection_report(output_path, report)
    print(f"[完成] SAWR detection report saved to {output_path}", file=sys.stderr)
    return 0


def _cmd_split(args: argparse.Namespace) -> int:
    records = load_jsonl_records(args.input)
    splits = split_records_by_task(
        records,
        dev_ratio=args.dev_ratio,
        calibration_ratio=args.calibration_ratio,
        seed=args.seed,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("dev", "calibration", "test"):
        _write_jsonl_records(output_dir / f"{split_name}.jsonl", splits[split_name])
    print(f"[完成] SAWR task splits saved to {output_dir}", file=sys.stderr)
    return 0


def _write_jsonl_records(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, allow_nan=False, ensure_ascii=False) + "\n"
            )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
