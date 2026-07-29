#!/usr/bin/env python3
"""Create a source manifest and fresh private key banks for one experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wfcllm.gate.prepare import (
    prepare_gated_experiment,
    prepare_gated_source_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-catalog", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--training-key-bank", type=Path, required=True)
    parser.add_argument("--holdout-key-bank", type=Path, required=True)
    parser.add_argument("--deployment-key", type=Path, required=True)
    parser.add_argument(
        "--reuse-private-resources",
        action="store_true",
        help=(
            "write only the source manifest; the supplied family key paths "
            "must already exist and are never modified"
        ),
    )
    args = parser.parse_args()
    if args.reuse_private_resources:
        for path in (
            args.training_key_bank,
            args.holdout_key_bank,
            args.deployment_key,
        ):
            if not path.is_file() or path.is_symlink():
                raise ValueError(
                    "shared family private resources must be existing regular files"
                )
        manifest = prepare_gated_source_manifest(
            source_catalog=args.source_catalog,
            source_manifest=args.source_manifest,
        )
    else:
        manifest = prepare_gated_experiment(
            source_catalog=args.source_catalog,
            source_manifest=args.source_manifest,
            training_key_bank=args.training_key_bank,
            holdout_key_bank=args.holdout_key_bank,
            deployment_key=args.deployment_key,
        )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
