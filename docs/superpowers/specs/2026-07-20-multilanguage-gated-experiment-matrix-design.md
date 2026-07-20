# Multi-language Gated Experiment Matrix Design

Date: 2026-07-20

## Goal

Provide server-ready experiment entry points and explicit configurations for
every currently valid language/dataset pair, using the gated semantic-window
method rather than the historical carrier-style keyed-text-region method.

The supported matrix is:

| Language | Dataset | Full profile | Fast profile |
| --- | --- | --- | --- |
| Python | HumanEval | yes | yes |
| Python | MBPP | yes | yes |
| C++ | HumanEvalPack | yes | yes |
| Java | HumanEvalPack | yes | yes |

Invalid pairs such as C++/MBPP are not exposed.

## Current Constraints

The repository already contains language adapters for Python, C++, and Java
and dataset adapters for HumanEval, MBPP, and HumanEvalPack. However, the
official gated runtime still contains Python-only assumptions:

- the CLI dataset choices omit HumanEvalPack and have no generation language;
- gated generation loads prompts through the legacy HumanEval/MBPP loader;
- the formal window contract and detector extractor are Python-specific;
- the current AST-equivalent rewriter certifies only Python rewrites;
- C++ and Java transformation-rule lists are intentionally empty.

Consequently, copying shell scripts alone would create C++ and Java entry
points that fail during generation or detection. The implementation must add
the smallest explicit language-aware runtime seam needed by the scripts. It
must not silently fall back to a carrier method.

## Chosen Structure

Use one shared shell runner plus eight thin public entry points.

The shared runner owns environment validation, private-key preparation, phase
ordering, offline environment variables, artifact paths, and full/fast profile
behavior. Each public entry point fixes only:

- language;
- dataset;
- profile;
- configuration path;
- default sample limit;
- default experiment root suffix.

This avoids eight drifting copies of the server orchestration while preserving
one directly executable script per experiment.

The public scripts are:

```text
scripts/experiments/run_python_humaneval_full.sh
scripts/experiments/run_python_humaneval_fast.sh
scripts/experiments/run_python_mbpp_full.sh
scripts/experiments/run_python_mbpp_fast.sh
scripts/experiments/run_cpp_humanevalpack_full.sh
scripts/experiments/run_cpp_humanevalpack_fast.sh
scripts/experiments/run_java_humanevalpack_full.sh
scripts/experiments/run_java_humanevalpack_fast.sh
```

The shared implementation is:

```text
scripts/experiments/run_gated_experiment.sh
```

## Profiles

### Full

The full profile follows the server's formal sequence:

1. prepare disjoint pilot and full source manifests and private key material;
2. build pilot gate data;
3. build full gate data using the pilot feasibility artifact;
4. train the gate;
5. validate and publish the gate bundle;
6. generate final code with gated semantic-window embedding;
7. calibrate on strict four-field negative final-code input;
8. detect positive final-code input;
9. report;
10. audit.

### Fast

The fast profile keeps the same gated semantic LSH method but reduces resource
cost through sample limits, smaller gate-data scale, fewer gate epochs, and a
bounded calibration subset. It may use the repository's explicitly
unvalidated runtime path, and its artifacts must remain marked as fast or
unvalidated. It must not use `keyed_text_region` or the carrier-style config.

## Configuration Contract

There is one explicit JSON configuration per matrix row and profile. Each
configuration declares at least:

- `generation.language`;
- `generation.dataset`;
- the language-specific window contract;
- the program finalizer, if any;
- semantic LSH parameters;
- rewrite strategy;
- gate validation requirements;
- a profile/method marker that distinguishes full and fast artifacts.

HumanEval uses the existing target-function finalizer. MBPP and
HumanEvalPack use no Python-specific finalizer unless a dataset-specific
finalizer is implemented and tested.

The fast configurations are derived semantically from the no-carrier method,
not from `gated_semantic_window_v1.json`, because that existing experimental
configuration selects `keyed_text_region`.

## Runtime Language Seam

The CLI gains an explicit generation language option and accepts
`humanevalpack` as a dataset. Gated generation resolves the configured dataset
adapter, validates that it supports the configured language, and normalizes
samples to the existing `id`/`prompt` generation contract.

Window extraction is selected explicitly by language and must be identical in
generation and final-code detection. The bundle/config checks include language
and window-contract compatibility so that a Python gate cannot accidentally be
used as a C++ or Java gate.

Python retains the certified AST-equivalent rewrite path. C++ and Java must
never call the Python AST validator. Until certified language-specific rewrite
rules exist, they use only an explicitly configured model-rewrite path with
parser-structure and semantic-preservation checks. If that path is unavailable,
preflight fails with a direct error instead of substituting whitespace or a
carrier channel.

## Server Inputs And Artifacts

The shared runner keeps the current environment-variable interface:

```text
PILOT_SOURCE_CATALOG
FULL_SOURCE_CATALOG
GENERATION_MODEL_PATH
SEMANTIC_ENCODER_MODEL_PATH
GATE_BASE_MODEL_PATH
NEGATIVE_INPUT
EXPERIMENT_ROOT
```

Optional rewrite-model, semantic-checkpoint, whitening, gate-epoch, and device
overrides remain supported. Pilot and full catalogs must resolve to different
files. Private key material remains under the experiment root and is never
written to public configuration or committed.

Each experiment receives a distinct default root containing its language,
dataset, and profile, preventing state-file and artifact collisions.

## Fail-fast Checks

Before a costly phase starts, the runner or Python preflight validates:

- language/dataset support;
- configuration language, dataset, profile, and semantic rule;
- absence of `keyed_text_region` for these main-method scripts;
- required model, source-catalog, negative-input, and dataset paths;
- distinct pilot/full source catalogs for full runs;
- required C++/Java parser packages;
- availability of the configured rewrite strategy;
- consistency between generation, gate bundle, and detection window contracts.

## Verification

Implementation verification includes:

- shell syntax checks for all experiment scripts;
- JSON parsing and config-contract tests for all eight configurations;
- tests that each thin script selects the intended matrix entry;
- tests that invalid language/dataset pairs fail before model loading;
- tests that no matrix config selects `keyed_text_region`;
- dataset and language adapter tests;
- generation/detection tests for language selection and mismatch rejection;
- integration tests for full/fast phase construction using fake runtimes;
- offline compile and relevant pytest suites.

No GPU experiment is required for the local implementation check. Server-side
execution remains responsible for validating actual model artifacts, datasets,
and full-run metrics.
