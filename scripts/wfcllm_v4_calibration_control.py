#!/usr/bin/env python3
"""Run frozen V4 primary- and wrong-key controls on calibration negatives only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
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
from wfcllm.batch_invariant_v4.detector import DetectorPayload, V4Detector  # noqa: E402
from wfcllm.batch_invariant_v4.keying import (  # noqa: E402
    derive_wrong_control_key,
    load_secret_key,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-config", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--calibration-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_public_config(args.public_config)
        calibration = load_calibration(args.calibration)
        input_sha256 = _sha256(args.calibration_input)
        if input_sha256 != calibration.source_sha256:
            raise ValueError("calibration input SHA-256 does not match frozen artifact")
        primary = load_secret_key(args.key_file)
        keys = {
            "primary": primary,
            "wrong_key_domain_separated": derive_wrong_control_key(primary),
        }
        extractor = StructuralContextExtractor(
            ContextConfig(
                schema_version=config.canonical_context.schema_version,
                max_unit_bytes=config.canonical_context.max_unit_bytes,
                max_context_bytes=config.canonical_context.max_context_bytes,
                global_ordinal_keying=config.canonical_context.global_ordinal_keying,
            )
        )
        rows = [
            json.loads(raw)
            for raw in args.calibration_input.read_text(encoding="utf-8").splitlines()
            if raw
        ]
        controls = {}
        for label, key in keys.items():
            detector = V4Detector(
                extractor=extractor,
                channel=StructuralChannel(
                    key,
                    bit_count=config.channel.bit_count_per_unit,
                    minimum_independent_units=config.channel.minimum_independent_units,
                ),
                calibration=calibration,
            )
            details = []
            for row in rows:
                evidence = detector.detect(
                    DetectorPayload(final_code=str(row["final_code"]))
                )
                details.append(
                    {
                        "id": str(row["id"]),
                        "score": evidence.score,
                        "independent_units": evidence.independent_units,
                        "eligible": evidence.eligible,
                        "p_value": evidence.p_value,
                        "decision": evidence.decision,
                    }
                )
            positives = sum(item["decision"] for item in details)
            controls[label] = {
                "row_count": len(details),
                "eligible_count": sum(item["eligible"] for item in details),
                "positive_count": positives,
                "positive_rate": positives / len(details),
                "details": details,
            }
        result = {
            "artifact_type": "wfcllm_v4_calibration_key_control",
            "schema_version": "wfcllm-v4-calibration-key-control/v1",
            "source_role": "calibration_negative",
            "heldout_accessed": False,
            "secret_metadata_included": False,
            "calibration_input_sha256": input_sha256,
            "public_config_sha256": _sha256(args.public_config),
            "calibration_artifact_sha256": _sha256(args.calibration),
            "target_fpr": calibration.target_fpr,
            "controls": controls,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
