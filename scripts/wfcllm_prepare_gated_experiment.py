#!/usr/bin/env python3
"""Create a source manifest and fresh private key banks for one experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wfcllm.gate.prepare import prepare_gated_experiment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-catalog", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--training-key-bank", type=Path, required=True)
    parser.add_argument("--holdout-key-bank", type=Path, required=True)
    parser.add_argument("--deployment-key", type=Path, required=True)
    args = parser.parse_args()
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
