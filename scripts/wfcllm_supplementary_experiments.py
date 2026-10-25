"""Run current-method supplementary experiment utilities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wfcllm.evaluation.supplementary_experiments import (
    DEFAULT_RENAME_CONDITIONS,
    aggregate_detection_conditions,
    aggregate_passk,
    build_all_experiment_plan,
    prepare_rename_intensity_inputs,
    write_all_experiment_plan,
    write_current_study_specs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Current-method WFCLLM supplementary experiment utilities",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser(
        "plan",
        help="Emit the local plan covering the two servers' experiment families.",
    )
    plan.add_argument("--output", type=Path)

    specs = subparsers.add_parser(
        "write-specs",
        help="Write formal ablation/current-study specs and run_commands.sh.",
    )
    specs.add_argument("--config", type=Path, required=True)
    specs.add_argument("--output-root", type=Path, required=True)
    specs.add_argument("--family-id", default="current-study")

    rename = subparsers.add_parser(
        "prepare-rename",
        help="Create deterministic renamed strict detector inputs.",
    )
    rename.add_argument("--positive-input", type=Path, required=True)
    rename.add_argument("--negative-input", type=Path, required=True)
    rename.add_argument("--entry-points", type=Path, required=True)
    rename.add_argument("--output-root", type=Path, required=True)
    rename.add_argument("--name-seed", type=int, default=7)
    rename.add_argument("--selection-seed", type=int, default=20260722)
    rename.add_argument(
        "--condition",
        action="append",
        default=[],
        metavar="NAME=FRACTION",
        help="Rename fraction condition; defaults to rename25/50/75/100.",
    )

    detect = subparsers.add_parser(
        "aggregate-detection",
        help="Aggregate detector detail JSONL files by robustness condition.",
    )
    detect.add_argument("--root", type=Path, required=True)
    detect.add_argument("--condition", action="append", required=True)
    detect.add_argument("--output-name", default="robustness_metrics")
    detect.add_argument("--minimum-reliable-windows", type=int, default=3)

    passk = subparsers.add_parser(
        "aggregate-passk",
        help="Aggregate posthoc sampled-pool Pass@k results.",
    )
    passk.add_argument("--root", type=Path, required=True)
    passk.add_argument("--num-samples", type=int, default=10)
    passk.add_argument("--k", action="append", type=int, default=[])
    passk.add_argument("--expected-task-count", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "plan":
        payload = (
            write_all_experiment_plan(args.output)
            if args.output is not None
            else build_all_experiment_plan()
        )
    elif args.command == "write-specs":
        payload = write_current_study_specs(
            canonical_config_path=args.config,
            output_root=args.output_root,
            family_id=args.family_id,
        )
    elif args.command == "prepare-rename":
        payload = prepare_rename_intensity_inputs(
            positive_input=args.positive_input,
            negative_input=args.negative_input,
            entry_points=args.entry_points,
            output_root=args.output_root,
            conditions=_parse_conditions(args.condition),
            name_seed=args.name_seed,
            selection_seed=args.selection_seed,
        )
    elif args.command == "aggregate-detection":
        payload = aggregate_detection_conditions(
            root=args.root,
            conditions=args.condition,
            output_name=args.output_name,
            minimum_reliable_windows=args.minimum_reliable_windows,
        )
    elif args.command == "aggregate-passk":
        payload = aggregate_passk(
            root=args.root,
            num_samples=args.num_samples,
            k_values=tuple(args.k) if args.k else (1, 10),
            expected_task_count=args.expected_task_count,
        )
    else:
        parser.error(f"unknown command: {args.command}")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def _parse_conditions(items: list[str]) -> dict[str, float]:
    if not items:
        return dict(DEFAULT_RENAME_CONDITIONS)
    result: dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError("--condition must use NAME=FRACTION")
        name, raw = item.split("=", 1)
        result[name] = float(raw)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
