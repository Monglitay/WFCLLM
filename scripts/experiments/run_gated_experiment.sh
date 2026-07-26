#!/usr/bin/env bash
set -euo pipefail

: "${WFCLLM_LANGUAGE:?wrapper must set WFCLLM_LANGUAGE}"
: "${WFCLLM_DATASET:?wrapper must set WFCLLM_DATASET}"
: "${WFCLLM_PROFILE:?wrapper must set WFCLLM_PROFILE}"
: "${WFCLLM_CONFIG:?wrapper must set WFCLLM_CONFIG}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${PYTHONPATH:-.}"

if [[ -z "${CONDA_EXE:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_EXE="$(command -v conda)"
  else
    CONDA_EXE="/root/miniconda3/bin/conda"
  fi
fi
if [[ -z "${CONDA_ENV_PREFIX:-}" ]]; then
  CONDA_ROOT="$(cd "$(dirname "${CONDA_EXE}")/.." && pwd)"
  LOCAL_ENV_PREFIX="${CONDA_ROOT}/envs/WFCLLM"
  if [[ -d "${LOCAL_ENV_PREFIX}" ]]; then
    CONDA_ENV_PREFIX="${LOCAL_ENV_PREFIX}"
  else
    CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/root/autodl-tmp/conda/envs}"
    CONDA_ENV_PREFIX="${CONDA_ENVS_PATH}/WFCLLM"
  fi
fi
RUN_PY=("${CONDA_EXE}" run -p "${CONDA_ENV_PREFIX}" python)

MISSING_HEAVY_RUNTIME=0
for REQUIRED_NAME in GENERATION_MODEL_PATH SEMANTIC_ENCODER_MODEL_PATH GATE_BASE_MODEL_PATH NEGATIVE_INPUT; do
  if [[ -z "${!REQUIRED_NAME:-}" ]]; then
    MISSING_HEAVY_RUNTIME=1
  fi
done

if [[ "${WFCLLM_DATASET}" == "humanevalpack" && "${MISSING_HEAVY_RUNTIME}" == "1" ]]; then
  ROOT="${EXPERIMENT_ROOT:-data/experiments/${WFCLLM_LANGUAGE}-${WFCLLM_DATASET}-${WFCLLM_PROFILE}}"
  RUN_DIR="${ROOT}/run"
  DATASET_PATH="${DATASET_PATH:-data/datasets}"
  SAMPLE_LIMIT="${SAMPLE_LIMIT:-${WFCLLM_DEFAULT_SAMPLE_LIMIT:-32}}"
  env PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    "${RUN_PY[@]}" scripts/experiments/run_basic_multilanguage_experiment.py \
    --config "${WFCLLM_CONFIG}" \
    --language "${WFCLLM_LANGUAGE}" \
    --dataset "${WFCLLM_DATASET}" \
    --profile "${WFCLLM_PROFILE}" \
    --dataset-path "${DATASET_PATH}" \
    --run-dir "${RUN_DIR}" \
    --sample-limit "${SAMPLE_LIMIT}"
  exit 0
fi

: "${GENERATION_MODEL_PATH:?set GENERATION_MODEL_PATH to a local HF code model}"
: "${SEMANTIC_ENCODER_MODEL_PATH:?set SEMANTIC_ENCODER_MODEL_PATH to a local semantic encoder}"
: "${GATE_BASE_MODEL_PATH:?set GATE_BASE_MODEL_PATH to a local CodeT5 gate base model}"
: "${NEGATIVE_INPUT:?set NEGATIVE_INPUT to strict four-field negative final_code JSONL}"

if [[ "${WFCLLM_PROFILE}" == "full" ]]; then
  : "${PILOT_SOURCE_CATALOG:?set PILOT_SOURCE_CATALOG for a full run}"
  : "${FULL_SOURCE_CATALOG:?set FULL_SOURCE_CATALOG for a full run}"
  if [[ "${PILOT_SOURCE_CATALOG}" == "${FULL_SOURCE_CATALOG}" ]]; then
    echo "PILOT_SOURCE_CATALOG and FULL_SOURCE_CATALOG must be different" >&2
    exit 2
  fi
else
  : "${SOURCE_CATALOG:?set SOURCE_CATALOG for a fast run}"
  PILOT_SOURCE_CATALOG="${SOURCE_CATALOG}"
  FULL_SOURCE_CATALOG="${SOURCE_CATALOG}"
fi

# Run public-contract validation before creating any private experiment state.
"${RUN_PY[@]}" -m wfcllm.cli.experiment_preflight \
  --config "${WFCLLM_CONFIG}" \
  --language "${WFCLLM_LANGUAGE}" \
  --dataset "${WFCLLM_DATASET}" \
  --profile "${WFCLLM_PROFILE}" \
  --check-runtime-capabilities

if command -v rg >/dev/null 2>&1; then
  HAS_KEYED_TEXT_REGION="$(rg -q '"rule_name"[[:space:]]*:[[:space:]]*"keyed_text_region"' "${WFCLLM_CONFIG}" && echo 1 || echo 0)"
else
  HAS_KEYED_TEXT_REGION="$(grep -Eq '"rule_name"[[:space:]]*:[[:space:]]*"keyed_text_region"' "${WFCLLM_CONFIG}" && echo 1 || echo 0)"
fi
if [[ "${HAS_KEYED_TEXT_REGION}" == "1" ]]; then
  echo "carrier config keyed_text_region is forbidden" >&2
  exit 2
fi

ROOT="${EXPERIMENT_ROOT:-data/experiments/${WFCLLM_LANGUAGE}-${WFCLLM_DATASET}-${WFCLLM_PROFILE}}"
DATASET_PATH="${DATASET_PATH:-data/datasets}"
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
SAMPLE_LIMIT="${SAMPLE_LIMIT:-${WFCLLM_DEFAULT_SAMPLE_LIMIT:-164}}"
CALIBRATION_LIMIT="${CALIBRATION_LIMIT:-32}"

mkdir -p "${ROOT}"
if [[ "${WFCLLM_PROFILE}" == "full" && ! -e "${PILOT_SOURCE_MANIFEST}" ]]; then
  env PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 "${RUN_PY[@]}" scripts/wfcllm_prepare_gated_experiment.py \
    --source-catalog "${PILOT_SOURCE_CATALOG}" \
    --source-manifest "${PILOT_SOURCE_MANIFEST}" \
    --training-key-bank "${PILOT_TRAINING_KEYS}" \
    --holdout-key-bank "${PILOT_HOLDOUT_KEYS}" \
    --deployment-key "${PILOT_DEPLOYMENT_KEY}"
fi
if [[ ! -e "${SOURCE_MANIFEST}" ]]; then
  env PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 "${RUN_PY[@]}" scripts/wfcllm_prepare_gated_experiment.py \
    --source-catalog "${FULL_SOURCE_CATALOG}" \
    --source-manifest "${SOURCE_MANIFEST}" \
    --training-key-bank "${TRAINING_KEYS}" \
    --holdout-key-bank "${HOLDOUT_KEYS}" \
    --deployment-key "${DEPLOYMENT_KEY}"
fi

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
if [[ -n "${REWRITE_MODEL_PATH:-}" ]]; then COMMON+=(--rewrite-model-path "${REWRITE_MODEL_PATH}"); fi
if [[ -n "${SEMANTIC_ENCODER_CHECKPOINT_PATH:-}" ]]; then COMMON+=(--semantic-encoder-checkpoint-path "${SEMANTIC_ENCODER_CHECKPOINT_PATH}"); fi
if [[ -n "${SEMANTIC_WHITENING_PATH:-}" ]]; then COMMON+=(--semantic-whitening-path "${SEMANTIC_WHITENING_PATH}"); fi

OFFLINE=(PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1)
run_phase() {
  local phase="$1"
  shift
  env "${OFFLINE[@]}" "${RUN_PY[@]}" run.py --phase "${phase}" "${COMMON[@]}" "$@"
}

GATE_TRAIN_OVERRIDES=()
if [[ -n "${GATE_EPOCHS:-}" ]]; then
  GATE_TRAIN_OVERRIDES+=(--gate-epochs "${GATE_EPOCHS}")
fi
if [[ -n "${GATE_EARLY_STOPPING_PATIENCE:-}" ]]; then
  GATE_TRAIN_OVERRIDES+=(
    --gate-early-stopping-patience "${GATE_EARLY_STOPPING_PATIENCE}"
  )
fi

if [[ "${WFCLLM_PROFILE}" == "full" ]]; then
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
  env "${OFFLINE[@]}" "${RUN_PY[@]}" run.py --phase gate-data \
    "${PILOT_COMMON[@]}" --gate-data-scale pilot \
    --model-device cuda --gate-device cuda
  PILOT_FEASIBILITY="${PILOT_RUN_DIR}/gate-data/feasibility_summary.json"
  run_phase gate-data --gate-data-scale full \
    --pilot-feasibility "${PILOT_FEASIBILITY}" \
    --model-device cuda --gate-device cuda
else
  run_phase gate-data --gate-data-scale full \
    --model-device cuda --gate-device cuda
fi
GATE_TRAIN_ARGS=(--model-device cuda --gate-device cuda)
if [[ "${WFCLLM_PROFILE}" == "full" ]]; then
  GATE_TRAIN_ARGS+=(--pilot-feasibility "${PILOT_FEASIBILITY}")
fi
GATE_TRAIN_ARGS+=("${GATE_TRAIN_OVERRIDES[@]}")
run_phase gate-train "${GATE_TRAIN_ARGS[@]}"

run_phase generate --model-device cuda --gate-device cpu --sample-limit "${SAMPLE_LIMIT}"

CALIBRATION="${RUN_DIR}/calibration/reference_calibration.json"
POSITIVE_INPUT="${RUN_DIR}/inputs/final_code.jsonl"
CALIBRATION_INPUT="${NEGATIVE_INPUT}"
if [[ "${WFCLLM_PROFILE}" == "fast" ]]; then
  CALIBRATION_INPUT="${ROOT}/negative_calibration.jsonl"
  head -n "${CALIBRATION_LIMIT}" "${NEGATIVE_INPUT}" > "${CALIBRATION_INPUT}"
fi

run_phase calibrate --negative-input "${CALIBRATION_INPUT}" --calibration "${CALIBRATION}" --model-device cuda --gate-device cpu
run_phase detect --input "${POSITIVE_INPUT}" --calibration "${CALIBRATION}" --model-device cuda --gate-device cpu
run_phase report
if [[ "${WFCLLM_PROFILE}" == "full" ]]; then
  run_phase audit
else
  echo "skipped audit for fast profile"
fi

echo "completed: ${RUN_DIR}"
