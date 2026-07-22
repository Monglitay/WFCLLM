# C++ / Java Language Support Design

## Goal

Extend the existing pluggable language and dataset layers so local WFCLLM tooling can consume C++ and Java HumanEvalPack samples and extract statement-level code blocks from both languages.

## Chosen design

- Register `cpp` and `java` implementations of `LanguageAdapter`.
- Use the official Tree-sitter grammar wheels for deterministic local parsing, matching the existing Python adapter.
- Share only the language-neutral AST traversal and block record; keep each language's statement taxonomy and parser construction in its own module.
- Implement the existing `HumanEvalPackAdapter` against the `bigcode/humanevalpack` dataset contract used by the referenced `code-watermark` repository. Restrict this increment to C++ and Java.
- Load with `download_mode="reuse_cache_if_exists"` and a caller-provided Hugging Face cache root. Fully offline runs require this cache to be pre-populated; the adapter does not treat the root as a standalone Parquet/JSONL dataset.

## Contracts

- `wfcllm.lang.names()` contains `python`, `cpp`, and `java`.
- C++ and Java adapters return flat statement blocks with stable IDs, source spans, nesting depth, and parent/child references.
- Transform rule lists are empty until language-specific semantics-preserving rules are implemented; parsing and dataset support do not pretend Python rules are portable.
- `HumanEvalPackAdapter.iter_samples(language)` yields normalized `CodeSample` rows from the dataset's `test` split.
- `language` must be `cpp` or `java`; omitting it iterates both configurations.
- `get_sample` accepts an optional language to disambiguate task IDs shared across configurations.

## Testing

- Registry tests cover both new language names.
- Adapter tests parse representative nested C++ and Java functions.
- Dataset tests mock the Hugging Face boundary and verify configuration, cache location, normalization, iteration, lookup, and invalid-language errors without network access.
