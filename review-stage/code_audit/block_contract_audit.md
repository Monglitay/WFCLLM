# Block Parser, Contract, and Alignment Audit

Generated: 2026-06-08T11:12:06.020503+00:00
Remote root: `/root/autodl-tmp/WFCLLM`
Scope: WFCLLM code/config/artifact/debug audit only. No production modules, configs, data artifacts, tests, artifact schema, or block-contract schema were modified.

## Verdict

The Python parser and block contract are deterministic for normal code in the probe, and alignment compares structural plus numeric contract fields. The main risk is that tree-sitter can produce blocks for syntactically invalid code, while some token-channel/training tests expect syntax-invalid samples to fail closed or be skipped. That is a confirmed test failure and a suspected data-processing bug for lexical training/evaluation.

## Source Review

- Python parser recursively extracts tree-sitter nodes in `wfcllm/lang/python/parser.py`; block IDs are assigned in traversal order, and source spans are decoded from tree-sitter node text.
- Canonical block contract preserves `ordinal`, `block_id`, `node_type`, `parent_node_type`, `block_text_hash`, `start_line`, `end_line`, `entropy_units`, `gamma_target`, `k`, and `gamma_effective`.
- `wfcllm/extract/alignment.py` compares rebuilt contracts to embedded contracts. Structure mismatches include block count, node type, and text hash; numeric mismatches include entropy/gamma/k/effective gamma. `contract_valid` focuses on structure, while `is_aligned` also requires no numeric mismatches.
- `BlockScorer` prefers hash-keyed contract lookup over block ID (`wfcllm/extract/scorer.py:34-45`, `scorer.py:63-86`), which is a sensible guard against block-ID shifts when source text is stable.

## Probe Evidence

- Simple function: 3 blocks, 2 simple contracts; stable source hashes and parent types.
- Compound loop/if: 3 blocks, 1 simple contract; parent type correctly set to `if_statement`.
- Bad syntax sample still produced a function definition and one return-statement contract rather than failing. This is the key risk.
- Generated `HumanEval/55` produced 43 parser blocks and 39 contracts. The embedded artifact also contains 39 block contracts, but duplicate repeated print statements share identical `block_text_hash` across many ordinals. Hash-only joins can therefore become ambiguous when duplicate source spans are repeated; ordinal is still present, but scorer's hash index overwrites duplicate hashes when building `contracts_by_hash`.

## Test Evidence

- `tests/lang/` passed: 9 passed.
- Token-channel training/extract tests failed around syntax-invalid handling:
  - `tests/watermark/token_channel/test_train_corpus.py::test_build_training_rows_skips_syntax_invalid_positive_variants` failed with `TypeError: object of type 'generator' has no len()`.
  - `tests/extract/test_token_channel.py::test_replay_detector_fails_closed_when_ast_feature_prep_fails` expected 0 scored positions but got 2.

## Risk Assessment

Block extraction itself is not ruled in as the primary semantic poor-performance cause, but bad-syntax and duplicate-hash behavior are not safely ruled out for token-channel or data-processing metrics. For semantic-only regenerated under eval mode, block parser determinism looked stable on probed cases.
