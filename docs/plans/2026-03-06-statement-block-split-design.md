# Statement Block Split Design

## Context

WFCLLM (Watermark for Code LLM) project needs to split code into independent statement blocks as preprocessing for LLM code generation watermark experiments.

- Dataset: MBPP (google-research-datasets/mbpp, "full" split, ~974 samples)
- Tool: tree-sitter with Python grammar
- Environment: conda WFCLLM, Python

## Directory Structure

```
WFCLLM/
├── data/mbpp/                         # MBPP dataset (auto-downloaded)
├── experiment/
│   ├── statement_block_split/         # Experiment directory
│   │   ├── split.py                   # Main: splitting logic
│   │   ├── config.py                  # Config (paths, params)
│   │   └── results/
│   │       └── mbpp_blocks.json       # Output
│   └── 语句块分割实验.md
```

## Dependencies

- `tree-sitter` — AST parsing engine
- `tree-sitter-python` — Python grammar
- `datasets` — Hugging Face, load MBPP

## Statement Block Classification

### Simple Statements

`expression_statement`, `return_statement`, `assert_statement`, `import_statement`, `import_from_statement`, `pass_statement`, `break_statement`, `continue_statement`, `raise_statement`, `delete_statement`, `global_statement`, `nonlocal_statement`

### Compound Statements

`function_definition`, `class_definition`, `if_statement`, `for_statement`, `while_statement`, `try_statement`, `with_statement`, `match_statement`

## Block Data Structure

```json
{
  "id": 0,
  "type": "compound",
  "node_type": "function_definition",
  "source": "def foo(x):\n    return x + 1",
  "start_line": 1,
  "end_line": 2,
  "depth": 0,
  "parent_id": null,
  "children_ids": [1]
}
```

Fields:
- `id`: unique block index within a code sample
- `type`: `"simple"` or `"compound"`
- `node_type`: tree-sitter node type name
- `source`: source code text
- `start_line` / `end_line`: line range
- `depth`: nesting depth (0 = top-level)
- `parent_id`: parent block id (null for top-level)
- `children_ids`: child block id list

## Output JSON Structure

```json
{
  "metadata": {
    "dataset": "mbpp",
    "split": "full",
    "total_samples": 974,
    "processed": 970,
    "failed": 4,
    "timestamp": "2026-03-06T..."
  },
  "samples": [
    {
      "task_id": 1,
      "prompt": "Write a function to find ...",
      "original_code": "def min_cost(...):\n  ...",
      "blocks": [ ... ],
      "stats": {
        "total_blocks": 5,
        "simple_blocks": 3,
        "compound_blocks": 2,
        "max_depth": 2
      }
    }
  ]
}
```

## Processing Flow

1. Load dataset via `datasets.load_dataset("google-research-datasets/mbpp", "full")`
2. Iterate each sample, take `code` field
3. Parse with tree-sitter into AST
4. Recursively extract statement blocks from `module` children:
   - Simple statement node -> create simple block
   - Compound statement node -> create compound block, recurse into body
5. Record per-sample statistics
6. Output JSON to `results/mbpp_blocks.json`

## Validation Strategy

Process first 10 samples, verify correctness, then run full dataset.

## Nesting Handling

Recursive expansion of all levels. Compound blocks contain their full source, and their children are also extracted as separate blocks with parent-child references.
