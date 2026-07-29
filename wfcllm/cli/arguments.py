from __future__ import annotations

import argparse
from pathlib import Path

from wfcllm.orchestration.state import PHASES

DEFAULT_CONFIG_FILE = Path("configs/base_config.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WFCLLM Gate-only Full Reproduction Core",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_FILE)
    parser.add_argument("--phase", choices=PHASES)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--run-dir")
    parser.add_argument("--state-file")

    parser.add_argument("--language", choices=["python", "cpp", "java", "js"])
    parser.add_argument(
        "--dataset", choices=["humaneval", "mbpp", "humanevalpack"]
    )
    parser.add_argument("--dataset-path")

    parser.add_argument("--negative-input")
    parser.add_argument("--test-negative-input")
    parser.add_argument("--ablation-spec", type=Path)

    parser.add_argument("--secret-key-file")
    parser.add_argument("--secret-key-env")
    parser.add_argument("--training-key-bank-file")
    parser.add_argument("--training-key-bank-env")
    parser.add_argument("--holdout-key-bank-file")
    parser.add_argument("--holdout-key-bank-env")

    parser.add_argument("--gate-source-manifest")
    parser.add_argument("--gate-source-catalog")
    parser.add_argument("--pilot-feasibility")
    parser.add_argument("--generation-model-path")
    parser.add_argument("--rewrite-model-path")
    parser.add_argument("--semantic-encoder-model-path")
    parser.add_argument("--gate-base-model-path")
    parser.add_argument("--model-device", default="cuda")
    parser.add_argument("--gate-device", default="cuda")
    parser.add_argument("--gate-cache-dir", default="data/gate-cache")
    parser.add_argument("--gate-batch-size", type=int, default=9)
    parser.add_argument("--gate-data-scale", choices=["pilot", "full"])
    return parser
