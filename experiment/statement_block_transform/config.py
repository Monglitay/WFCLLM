"""Configuration for statement block transform experiment."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "mbpp_blocks_transformed.json"
OUTPUT_FILE_NEGATIVE = RESULTS_DIR / "mbpp_blocks_negative_transformed.json"

# Input from Phase 1
INPUT_FILE = PROJECT_ROOT / "experiment" / "statement_block_split" / "results" / "mbpp_blocks.json"

# Permutation limits
MAX_PERMUTATION_LENGTH = 5
MAX_VARIANTS_PER_BLOCK = 1000
