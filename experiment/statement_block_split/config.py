"""Configuration for statement block split experiment."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "mbpp_blocks.json"

# Dataset
DATASET_NAME = "google-research-datasets/mbpp"
DATASET_CONFIG = "full"
DATA_CACHE_DIR = PROJECT_ROOT / "data" / "mbpp"

# Statement block node types
SIMPLE_STATEMENT_TYPES = frozenset({
    "expression_statement",
    "return_statement",
    "assert_statement",
    "import_statement",
    "import_from_statement",
    "pass_statement",
    "break_statement",
    "continue_statement",
    "raise_statement",
    "delete_statement",
    "global_statement",
    "nonlocal_statement",
})

COMPOUND_STATEMENT_TYPES = frozenset({
    "function_definition",
    "class_definition",
    "if_statement",
    "for_statement",
    "while_statement",
    "try_statement",
    "with_statement",
    "match_statement",
})

STATEMENT_TYPES = SIMPLE_STATEMENT_TYPES | COMPOUND_STATEMENT_TYPES
