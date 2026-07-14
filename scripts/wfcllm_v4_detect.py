#!/usr/bin/env python3
"""Run the frozen final-code-only Watermark Mechanism V4 detector."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.batch_invariant_v4.calibration import load_calibration  # noqa: E402
from wfcllm.batch_invariant_v4.channel import StructuralChannel  # noqa: E402
from wfcllm.batch_invariant_v4.config import load_public_config  # noqa: E402
from wfcllm.batch_invariant_v4.context import (  # noqa: E402
    ContextConfig,
    StructuralContextExtractor,
)
from wfcllm.batch_invariant_v4.detector import (  # noqa: E402
    DetectorPayload,
    V4Detector,
)
from wfcllm.batch_invariant_v4.keying import load_secret_key  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-config", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_public_config(args.public_config)
        calibration = load_calibration(args.calibration)
        key = load_secret_key(args.key_file)
        payload = DetectorPayload.from_dict(
            json.loads(args.input.read_text(encoding="utf-8"))
        )
        detector = V4Detector(
            extractor=StructuralContextExtractor(
                ContextConfig(
                    schema_version=config.canonical_context.schema_version,
                    max_unit_bytes=config.canonical_context.max_unit_bytes,
                    max_context_bytes=config.canonical_context.max_context_bytes,
                    global_ordinal_keying=config.canonical_context.global_ordinal_keying,
                )
            ),
            channel=StructuralChannel(
                key,
                bit_count=config.channel.bit_count_per_unit,
                minimum_independent_units=config.channel.minimum_independent_units,
            ),
            calibration=calibration,
        )
        evidence = detector.detect(payload)
        result = {
            "artifact_type": "wfcllm_v4_r3_detection",
            "schema_version": "wfcllm-v4-exact-evidence/v1",
            "secret_metadata_included": False,
            "public_config_sha256": hashlib.sha256(
                args.public_config.read_bytes()
            ).hexdigest(),
            "evidence": asdict(evidence),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
