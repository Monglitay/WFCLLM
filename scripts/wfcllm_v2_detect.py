#!/usr/bin/env python
"""Run strict final-code-only WFCLLM V2 calibration and detection."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.detection.metrics import (  # noqa: E402
    build_detection_report,
    write_detection_report,
)
from wfcllm.detection.pipeline import load_jsonl_records  # noqa: E402
from wfcllm.detection.pipeline_v2 import (  # noqa: E402
    WFCLLMV2DetectionConfig,
    WFCLLMV2DetectionPipeline,
    load_v2_calibration_artifact,
    write_v2_calibration_artifact,
)
from wfcllm.detection.signature_v2 import load_v2_signature_scorer  # noqa: E402

SECRET_KEY_ENV_VAR = "WFCLLM_SECRET_KEY"


def _resolve_secret_key(cli_secret_key: str | None) -> str:
    if cli_secret_key:
        return cli_secret_key
    env_secret_key = os.environ.get(SECRET_KEY_ENV_VAR)
    if env_secret_key:
        return env_secret_key
    raise ValueError(
        f"secret key must be provided via --secret-key or {SECRET_KEY_ENV_VAR}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run strict final-code-only WFCLLM V2 detection.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    calibrate = subparsers.add_parser("calibrate")
    _add_detector_args(calibrate)
    calibrate.add_argument("--input", required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.set_defaults(handler=_cmd_calibrate)

    detect = subparsers.add_parser("detect")
    _add_detector_args(detect)
    detect.add_argument("--input", required=True)
    detect.add_argument("--calibration", required=True)
    detect.add_argument("--output", required=True)
    detect.set_defaults(handler=_cmd_detect)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--positive-details", required=True)
    evaluate.add_argument("--negative-details", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.set_defaults(handler=_cmd_evaluate)
    return parser


def _add_detector_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--secret-key", default=None)
    parser.add_argument("--encoder-model-path", required=True)
    parser.add_argument("--encoder-checkpoint-path", default=None)
    parser.add_argument("--encoder-embed-dim", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoder-use-lora", action="store_true")
    parser.add_argument("--encoder-use-bf16", action="store_true")
    parser.add_argument("--lsh-whitening-path", default=None)
    parser.add_argument("--signature-bits", type=int, default=16)
    parser.add_argument("--min-canonical-units", type=int, default=1)
    parser.add_argument("--target-fpr", type=float, default=0.05)


def _build_pipeline(args: argparse.Namespace) -> WFCLLMV2DetectionPipeline:
    secret_key = _resolve_secret_key(args.secret_key)
    config = WFCLLMV2DetectionConfig(
        secret_key=secret_key,
        signature_bits=args.signature_bits,
        min_canonical_units=args.min_canonical_units,
        target_fpr=args.target_fpr,
    )
    scorer = load_v2_signature_scorer(
        encoder_model_path=args.encoder_model_path,
        encoder_checkpoint_path=args.encoder_checkpoint_path,
        embed_dim=args.encoder_embed_dim,
        device=args.device,
        use_lora=args.encoder_use_lora,
        use_bf16=args.encoder_use_bf16,
        secret_key=secret_key,
        signature_bits=args.signature_bits,
        whitening_path=args.lsh_whitening_path,
    )
    return WFCLLMV2DetectionPipeline(config=config, scorer=scorer)


def _cmd_calibrate(args: argparse.Namespace) -> int:
    records = load_jsonl_records(args.input)
    pipeline = _build_pipeline(args)
    artifact = pipeline.calibrate(records)
    output_path = Path(args.output)
    write_v2_calibration_artifact(output_path, artifact)
    print(f"[完成] WFCLLM V2 calibration saved to {output_path}", file=sys.stderr)
    return 0


def _cmd_detect(args: argparse.Namespace) -> int:
    records = load_jsonl_records(args.input)
    artifact = load_v2_calibration_artifact(args.calibration)
    pipeline = _build_pipeline(args)
    output_path = Path(args.output)
    pipeline.detect_to_jsonl(records, artifact=artifact, output_path=output_path)
    print(f"[完成] WFCLLM V2 details saved to {output_path}", file=sys.stderr)
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    positive_rows = _load_generic_jsonl(args.positive_details)
    negative_rows = _load_generic_jsonl(args.negative_details)
    report = build_detection_report(positive_rows, negative_rows)
    output_path = Path(args.output)
    write_detection_report(output_path, report)
    print(f"[完成] WFCLLM V2 report saved to {output_path}", file=sys.stderr)
    return 0


def _load_generic_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL row {line_number} must be an object")
                rows.append(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSONL at line {exc.lineno}") from exc
    return rows


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
