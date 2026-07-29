#!/usr/bin/env bash
set -euo pipefail

: "${WFCLLM_LANGUAGE:?wrapper must set WFCLLM_LANGUAGE}"
: "${WFCLLM_DATASET:?wrapper must set WFCLLM_DATASET}"
: "${WFCLLM_PROFILE:?wrapper must set WFCLLM_PROFILE}"
: "${WFCLLM_CONFIG:?wrapper must set WFCLLM_CONFIG}"
if [[ "${WFCLLM_PROFILE}" != "full" ]]; then
  echo "only the full reproduction profile is supported" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${PYTHONPATH:-.}"

CONDA_EXE="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "${CONDA_EXE}" ]]; then
  echo "conda is required to run the WFCLLM environment" >&2
  exit 2
fi
RUN_PY=("${CONDA_EXE}" run -n WFCLLM python)

: "${GENERATION_MODEL_PATH:?set GENERATION_MODEL_PATH to a local HF code model}"
REWRITE_MODEL_PATH="${REWRITE_MODEL_PATH:-}"
if [[ "${WFCLLM_LANGUAGE}" != "python" && -z "${REWRITE_MODEL_PATH}" ]]; then
  echo "set REWRITE_MODEL_PATH for non-Python model rewriting" >&2
  exit 2
fi
: "${SEMANTIC_ENCODER_MODEL_PATH:?set SEMANTIC_ENCODER_MODEL_PATH to a local semantic encoder}"
: "${GATE_BASE_MODEL_PATH:?set GATE_BASE_MODEL_PATH to a local CodeT5 gate base model}"
: "${NEGATIVE_INPUT:?set NEGATIVE_INPUT to strict four-field negative final_code JSONL}"
ABLATION_SPEC="${ABLATION_SPEC:-}"
TEST_NEGATIVE_INPUT="${TEST_NEGATIVE_INPUT:-}"
ABLATION_TRAINING_KEY_BANK="${ABLATION_TRAINING_KEY_BANK:-}"
ABLATION_HOLDOUT_KEY_BANK="${ABLATION_HOLDOUT_KEY_BANK:-}"
ABLATION_DEPLOYMENT_KEY="${ABLATION_DEPLOYMENT_KEY:-}"
if [[ -n "${ABLATION_SPEC}" && -z "${TEST_NEGATIVE_INPUT}" ]]; then
  echo "set TEST_NEGATIVE_INPUT for Supplementary Ablation actual FPR" >&2
  exit 2
fi
if [[ -n "${ABLATION_SPEC}" ]]; then
  : "${ABLATION_TRAINING_KEY_BANK:?set ABLATION_TRAINING_KEY_BANK to the family-shared private bank}"
  : "${ABLATION_HOLDOUT_KEY_BANK:?set ABLATION_HOLDOUT_KEY_BANK to the family-shared private bank}"
  : "${ABLATION_DEPLOYMENT_KEY:?set ABLATION_DEPLOYMENT_KEY to the family-shared private key}"
fi
: "${PILOT_SOURCE_CATALOG:?set PILOT_SOURCE_CATALOG for a full run}"
: "${FULL_SOURCE_CATALOG:?set FULL_SOURCE_CATALOG for a full run}"
if [[ "${PILOT_SOURCE_CATALOG}" == "${FULL_SOURCE_CATALOG}" ]]; then
  echo "PILOT_SOURCE_CATALOG and FULL_SOURCE_CATALOG must be different" >&2
  exit 2
fi

# Run public-contract validation before creating any private experiment state.
DATASET_PATH="${DATASET_PATH:-data/datasets}"
PREFLIGHT_RESOURCES=(
  --check-runtime-resources
  --generation-model-path "${GENERATION_MODEL_PATH}"
  --semantic-encoder-model-path "${SEMANTIC_ENCODER_MODEL_PATH}"
  --gate-base-model-path "${GATE_BASE_MODEL_PATH}"
  --dataset-path "${DATASET_PATH}"
  --pilot-source-catalog "${PILOT_SOURCE_CATALOG}"
  --full-source-catalog "${FULL_SOURCE_CATALOG}"
  --negative-input "${NEGATIVE_INPUT}"
)
if [[ -n "${REWRITE_MODEL_PATH}" ]]; then
  PREFLIGHT_RESOURCES+=(--rewrite-model-path "${REWRITE_MODEL_PATH}")
fi
if [[ -n "${ABLATION_SPEC}" ]]; then
  PREFLIGHT_RESOURCES+=(
    --ablation-spec "${ABLATION_SPEC}"
    --test-negative-input "${TEST_NEGATIVE_INPUT}"
    --training-key-bank "${ABLATION_TRAINING_KEY_BANK}"
    --holdout-key-bank "${ABLATION_HOLDOUT_KEY_BANK}"
    --deployment-key "${ABLATION_DEPLOYMENT_KEY}"
  )
fi
env PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  "${RUN_PY[@]}" -m wfcllm.cli.experiment_preflight \
  --config "${WFCLLM_CONFIG}" \
  --language "${WFCLLM_LANGUAGE}" \
  --dataset "${WFCLLM_DATASET}" \
  --profile "${WFCLLM_PROFILE}" \
  --check-runtime-capabilities \
  "${PREFLIGHT_RESOURCES[@]}"

ROOT="${EXPERIMENT_ROOT:-data/experiments/${WFCLLM_LANGUAGE}-${WFCLLM_DATASET}-${WFCLLM_PROFILE}}"
if [[ -e "${ROOT}" ]]; then
  echo "Fresh Reproduction Run requires a new EXPERIMENT_ROOT: ${ROOT}" >&2
  exit 2
fi
PRIVATE_DIR="${ROOT}/private"
CACHE_DIR="${ROOT}/gate-cache"
RUN_DIR="${ROOT}/run"
STATE_FILE="${ROOT}/run_state.json"
PILOT_PRIVATE_DIR="${ROOT}/pilot-private"
PILOT_RUN_DIR="${ROOT}/pilot"
PILOT_STATE_FILE="${ROOT}/pilot_state.json"
PILOT_SOURCE_MANIFEST="${ROOT}/pilot_source_manifest.json"
PILOT_TRAINING_KEYS="${PILOT_PRIVATE_DIR}/training_keys.json"
PILOT_HOLDOUT_KEYS="${PILOT_PRIVATE_DIR}/holdout_keys.json"
PILOT_DEPLOYMENT_KEY="${PILOT_PRIVATE_DIR}/deployment.key"
SOURCE_MANIFEST="${ROOT}/source_manifest.json"
TRAINING_KEYS="${PRIVATE_DIR}/training_keys.json"
HOLDOUT_KEYS="${PRIVATE_DIR}/holdout_keys.json"
DEPLOYMENT_KEY="${PRIVATE_DIR}/deployment.key"
PREPARE_PRIVATE_MODE=()
if [[ -n "${ABLATION_SPEC}" ]]; then
  TRAINING_KEYS="${ABLATION_TRAINING_KEY_BANK}"
  HOLDOUT_KEYS="${ABLATION_HOLDOUT_KEY_BANK}"
  DEPLOYMENT_KEY="${ABLATION_DEPLOYMENT_KEY}"
  PILOT_TRAINING_KEYS="${ABLATION_TRAINING_KEY_BANK}"
  PILOT_HOLDOUT_KEYS="${ABLATION_HOLDOUT_KEY_BANK}"
  PILOT_DEPLOYMENT_KEY="${ABLATION_DEPLOYMENT_KEY}"
  PREPARE_PRIVATE_MODE=(--reuse-private-resources)
fi

mkdir -p "${ROOT}"
env PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 "${RUN_PY[@]}" scripts/wfcllm_prepare_gated_experiment.py \
  --source-catalog "${PILOT_SOURCE_CATALOG}" \
  --source-manifest "${PILOT_SOURCE_MANIFEST}" \
  --training-key-bank "${PILOT_TRAINING_KEYS}" \
  --holdout-key-bank "${PILOT_HOLDOUT_KEYS}" \
  --deployment-key "${PILOT_DEPLOYMENT_KEY}" \
  "${PREPARE_PRIVATE_MODE[@]}"
env PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 "${RUN_PY[@]}" scripts/wfcllm_prepare_gated_experiment.py \
  --source-catalog "${FULL_SOURCE_CATALOG}" \
  --source-manifest "${SOURCE_MANIFEST}" \
  --training-key-bank "${TRAINING_KEYS}" \
  --holdout-key-bank "${HOLDOUT_KEYS}" \
  --deployment-key "${DEPLOYMENT_KEY}" \
  "${PREPARE_PRIVATE_MODE[@]}"

COMMON=(
  --config "${WFCLLM_CONFIG}"
  --run-dir "${RUN_DIR}"
  --state-file "${STATE_FILE}"
  --language "${WFCLLM_LANGUAGE}"
  --dataset "${WFCLLM_DATASET}"
  --dataset-path "${DATASET_PATH}"
  --gate-source-manifest "${SOURCE_MANIFEST}"
  --gate-source-catalog "${FULL_SOURCE_CATALOG}"
  --training-key-bank-file "${TRAINING_KEYS}"
  --holdout-key-bank-file "${HOLDOUT_KEYS}"
  --secret-key-file "${DEPLOYMENT_KEY}"
  --generation-model-path "${GENERATION_MODEL_PATH}"
  --semantic-encoder-model-path "${SEMANTIC_ENCODER_MODEL_PATH}"
  --gate-base-model-path "${GATE_BASE_MODEL_PATH}"
  --gate-cache-dir "${CACHE_DIR}"
)
if [[ -n "${REWRITE_MODEL_PATH}" ]]; then
  COMMON+=(--rewrite-model-path "${REWRITE_MODEL_PATH}")
fi
if [[ -n "${ABLATION_SPEC}" ]]; then
  COMMON+=(
    --ablation-spec "${ABLATION_SPEC}"
    --test-negative-input "${TEST_NEGATIVE_INPUT}"
  )
fi

OFFLINE=(PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1)
run_phase() {
  local phase="$1"
  shift
  env "${OFFLINE[@]}" "${RUN_PY[@]}" run.py --phase "${phase}" "${COMMON[@]}" "$@"
}

PILOT_COMMON=(
    --config "${WFCLLM_CONFIG}"
    --run-dir "${PILOT_RUN_DIR}"
    --state-file "${PILOT_STATE_FILE}"
    --language "${WFCLLM_LANGUAGE}"
    --dataset "${WFCLLM_DATASET}"
    --dataset-path "${DATASET_PATH}"
    --gate-source-manifest "${PILOT_SOURCE_MANIFEST}"
    --gate-source-catalog "${PILOT_SOURCE_CATALOG}"
    --training-key-bank-file "${PILOT_TRAINING_KEYS}"
    --holdout-key-bank-file "${PILOT_HOLDOUT_KEYS}"
    --secret-key-file "${PILOT_DEPLOYMENT_KEY}"
    --generation-model-path "${GENERATION_MODEL_PATH}"
    --semantic-encoder-model-path "${SEMANTIC_ENCODER_MODEL_PATH}"
    --gate-base-model-path "${GATE_BASE_MODEL_PATH}"
    --gate-cache-dir "${CACHE_DIR}"
)
if [[ -n "${REWRITE_MODEL_PATH}" ]]; then
  PILOT_COMMON+=(--rewrite-model-path "${REWRITE_MODEL_PATH}")
fi
if [[ -n "${ABLATION_SPEC}" ]]; then
  PILOT_COMMON+=(
    --ablation-spec "${ABLATION_SPEC}"
    --test-negative-input "${TEST_NEGATIVE_INPUT}"
  )
fi
env "${OFFLINE[@]}" "${RUN_PY[@]}" run.py --phase encoder "${PILOT_COMMON[@]}"
env "${OFFLINE[@]}" "${RUN_PY[@]}" run.py --phase gate-data \
  "${PILOT_COMMON[@]}" --gate-data-scale pilot \
  --model-device cuda --gate-device cuda
PILOT_FEASIBILITY="${PILOT_RUN_DIR}/gate-data/feasibility_summary.json"
run_phase encoder
run_phase gate-data --gate-data-scale full \
  --pilot-feasibility "${PILOT_FEASIBILITY}" \
  --model-device cuda --gate-device cuda
GATE_TRAIN_ARGS=(
  --model-device cuda
  --gate-device cuda
  --pilot-feasibility "${PILOT_FEASIBILITY}"
)
run_phase gate-train "${GATE_TRAIN_ARGS[@]}"

run_phase generate --model-device cuda --gate-device cpu
run_phase calibrate --negative-input "${NEGATIVE_INPUT}" --model-device cuda --gate-device cpu
run_phase detect --model-device cuda --gate-device cpu
run_phase report
run_phase audit

env "${OFFLINE[@]}" "${RUN_PY[@]}" scripts/wfcllm_metric_contract.py \
  --run-dir "${RUN_DIR}" \
  --output "${RUN_DIR}/reports/metric_contract.jsonl"

echo "completed: ${RUN_DIR}"
