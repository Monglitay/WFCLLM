#!/usr/bin/env bash
set -euo pipefail

: "${SOURCE_CATALOG:?set SOURCE_CATALOG to the strict gate source JSONL}"
: "${GENERATION_MODEL_PATH:?set GENERATION_MODEL_PATH to a local HF code model}"
: "${SEMANTIC_ENCODER_MODEL_PATH:?set SEMANTIC_ENCODER_MODEL_PATH to the local semantic encoder}"
: "${GATE_BASE_MODEL_PATH:?set GATE_BASE_MODEL_PATH to the local CodeT5 gate base model}"
: "${NEGATIVE_INPUT:?set NEGATIVE_INPUT to strict four-field negative final_code JSONL}"

ROOT="${EXPERIMENT_ROOT:-data/experiments/gated-single-gpu}"
DATASET_PATH="${DATASET_PATH:-data/datasets}"
FULL_CONFIG="configs/wfcllm/gated_semantic_window_v1.json"
PRIVATE_DIR="${ROOT}/private"
SOURCE_MANIFEST="${ROOT}/source_manifest.json"
TRAINING_KEYS="${PRIVATE_DIR}/training_keys.json"
HOLDOUT_KEYS="${PRIVATE_DIR}/holdout_keys.json"
DEPLOYMENT_KEY="${PRIVATE_DIR}/deployment.key"
CACHE_DIR="${ROOT}/gate-cache"
FULL_RUN="${ROOT}/full"
FULL_STATE="${ROOT}/full_state.json"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-32}"
CALIBRATION_LIMIT="${CALIBRATION_LIMIT:-32}"
GATE_EPOCHS="${GATE_EPOCHS:-}"
GATE_EARLY_STOPPING_PATIENCE="${GATE_EARLY_STOPPING_PATIENCE:-}"
CALIBRATION_INPUT="${ROOT}/negative_calibration.jsonl"

mkdir -p "${ROOT}"
if [[ ! -e "${SOURCE_MANIFEST}" && ! -e "${TRAINING_KEYS}" && ! -e "${HOLDOUT_KEYS}" && ! -e "${DEPLOYMENT_KEY}" ]]; then
  conda run -n WFCLLM python scripts/wfcllm_prepare_gated_experiment.py \
    --source-catalog "${SOURCE_CATALOG}" \
    --source-manifest "${SOURCE_MANIFEST}" \
    --training-key-bank "${TRAINING_KEYS}" \
    --holdout-key-bank "${HOLDOUT_KEYS}" \
    --deployment-key "${DEPLOYMENT_KEY}"
fi

COMMON=(
  --gate-source-manifest "${SOURCE_MANIFEST}"
  --gate-source-catalog "${SOURCE_CATALOG}"
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
GATE_TRAIN_OVERRIDES=()
if [[ -n "${GATE_EPOCHS}" ]]; then GATE_TRAIN_OVERRIDES+=(--gate-epochs "${GATE_EPOCHS}"); fi
if [[ -n "${GATE_EARLY_STOPPING_PATIENCE}" ]]; then
  GATE_TRAIN_OVERRIDES+=(--gate-early-stopping-patience "${GATE_EARLY_STOPPING_PATIENCE}")
fi

env "${OFFLINE[@]}" conda run -n WFCLLM python run.py \
  --phase gate-data --config "${FULL_CONFIG}" --run-dir "${FULL_RUN}" \
  --state-file "${FULL_STATE}" \
  --model-device cuda --gate-device cuda "${COMMON[@]}"

env "${OFFLINE[@]}" conda run -n WFCLLM python run.py \
  --phase gate-train --config "${FULL_CONFIG}" --run-dir "${FULL_RUN}" \
  --state-file "${FULL_STATE}" \
  --model-device cuda --gate-device cuda "${COMMON[@]}" "${GATE_TRAIN_OVERRIDES[@]}"

env "${OFFLINE[@]}" conda run -n WFCLLM python run.py \
  --phase generate --config "${FULL_CONFIG}" --run-dir "${FULL_RUN}" \
  --state-file "${FULL_STATE}" --dataset-path "${DATASET_PATH}" \
  --model-device cuda --gate-device cpu --sample-limit "${SAMPLE_LIMIT}" "${COMMON[@]}"

CALIBRATION="${FULL_RUN}/calibration/reference_calibration.json"
head -n "${CALIBRATION_LIMIT}" "${NEGATIVE_INPUT}" > "${CALIBRATION_INPUT}"
POSITIVE_INPUT="${FULL_RUN}/inputs/final_code.jsonl"
env "${OFFLINE[@]}" conda run -n WFCLLM python scripts/wfcllm_fast_postprocess.py \
  --config "${FULL_CONFIG}" --run-dir "${FULL_RUN}" \
  --state-file "${FULL_STATE}" --negative-input "${CALIBRATION_INPUT}" \
  --input "${POSITIVE_INPUT}" --calibration "${CALIBRATION}" \
  --model-device cuda --gate-device cpu "${COMMON[@]}"

echo "completed: ${FULL_RUN}"

