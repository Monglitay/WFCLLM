"""Configuration for node entropy experiment."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "node_entropy_results.json"

INPUT_FILE = (
    PROJECT_ROOT
    / "experiment"
    / "statement_block_transform"
    / "results"
    / "mbpp_blocks_transformed.json"
)
