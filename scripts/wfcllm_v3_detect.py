#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.dynamic_semantic.calibration import load_calibration
from wfcllm.dynamic_semantic.channel import SemanticChannel
from wfcllm.dynamic_semantic.config import load_public_config
from wfcllm.dynamic_semantic.context import DynamicContextExtractor
from wfcllm.dynamic_semantic.detector import R3Detector
from wfcllm.dynamic_semantic.encoder import SemanticEncoderRuntime
from wfcllm.dynamic_semantic.keying import load_secret_key
from wfcllm.dynamic_semantic.whitening import load_whitening


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run final-code-only Dynamic Semantic V3 R3 detection.")
    parser.add_argument("--public-config", required=True)
    parser.add_argument("--whitening", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--key-file", required=True)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--code-field", choices=["final_code", "generated_code"], required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_public_config(args.public_config)
    runtime = SemanticEncoderRuntime.load(config.encoder, max_tokens=config.context.max_context_tokens, device=args.device)
    extractor = DynamicContextExtractor(config.context, token_counter=runtime.token_count)
    detector = R3Detector(
        extractor=extractor,
        encoder=runtime,
        whitening=load_whitening(args.whitening),
        channel=SemanticChannel(load_secret_key(args.key_file), config.channel),
        calibration=load_calibration(args.calibration),
        minimum_independent_units=config.channel.minimum_independent_units,
    )
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Path(args.input_jsonl).open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as output:
        for raw in source:
            if not raw.strip():
                continue
            row = json.loads(raw)
            result = detector.detect_payload({"final_code": str(row[args.code_field])})
            public = result.public_dict()
            public["id"] = row.get("id")
            output.write(json.dumps(public, ensure_ascii=False, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
