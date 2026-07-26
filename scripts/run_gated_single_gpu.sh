#!/usr/bin/env bash
set -euo pipefail

: "${PILOT_SOURCE_CATALOG:?set PILOT_SOURCE_CATALOG to the pilot gate source JSONL}"
: "${FULL_SOURCE_CATALOG:?set FULL_SOURCE_CATALOG to the disjoint full gate source JSONL}"
: "${GENERATION_MODEL_PATH:?set GENERATION_MODEL_PATH to a local HF code model}"
: "${SEMANTIC_ENCODER_MODEL_PATH:?set SEMANTIC_ENCODER_MODEL_PATH to the local semantic encoder}"
: "${GATE_BASE_MODEL_PATH:?set GATE_BASE_MODEL_PATH to the local CodeT5 gate base model}"
: "${NEGATIVE_INPUT:?set NEGATIVE_INPUT to strict four-field negative final_code JSONL}"

ROOT="${EXPERIMENT_ROOT:-data/experiments/gated-single-gpu}"
DATASET_PATH="${DATASET_PATH:-data/datasets}"
CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/root/autodl-tmp/conda/envs}"
CONDA_EXE="${CONDA_EXE:-/root/miniconda3/bin/conda}"
PILOT_CONFIG="configs/wfcllm/gated_semantic_window_pilot.json"
FULL_CONFIG="configs/wfcllm/gated_semantic_window_nocarrier_v1.json"
PILOT_PRIVATE_DIR="${ROOT}/pilot-private"
FULL_PRIVATE_DIR="${ROOT}/full-private"
PILOT_SOURCE_MANIFEST="${ROOT}/pilot_source_manifest.json"
FULL_SOURCE_MANIFEST="${ROOT}/full_source_manifest.json"
PILOT_TRAINING_KEYS="${PILOT_PRIVATE_DIR}/training_keys.json"
PILOT_HOLDOUT_KEYS="${PILOT_PRIVATE_DIR}/holdout_keys.json"
PILOT_DEPLOYMENT_KEY="${PILOT_PRIVATE_DIR}/deployment.key"
FULL_TRAINING_KEYS="${FULL_PRIVATE_DIR}/training_keys.json"
FULL_HOLDOUT_KEYS="${FULL_PRIVATE_DIR}/holdout_keys.json"
FULL_DEPLOYMENT_KEY="${FULL_PRIVATE_DIR}/deployment.key"
CACHE_DIR="${ROOT}/gate-cache"
PILOT_RUN="${ROOT}/pilot"
FULL_RUN="${ROOT}/full"
PILOT_STATE="${ROOT}/pilot_state.json"
FULL_STATE="${ROOT}/full_state.json"
SAMPLE_LIMIT="${SAMPLE_LIMIT:-164}"
GATE_EPOCHS="${GATE_EPOCHS:-}"
GATE_EARLY_STOPPING_PATIENCE="${GATE_EARLY_STOPPING_PATIENCE:-}"

mkdir -p "${ROOT}"
if [[ ! -e "${PILOT_SOURCE_MANIFEST}" && ! -e "${PILOT_TRAINING_KEYS}" && ! -e "${PILOT_HOLDOUT_KEYS}" && ! -e "${PILOT_DEPLOYMENT_KEY}" ]]; then
  "${CONDA_EXE}" run -p "${CONDA_ENVS_PATH}/WFCLLM" python scripts/wfcllm_prepare_gated_experiment.py \
    --source-catalog "${PILOT_SOURCE_CATALOG}" \
    --source-manifest "${PILOT_SOURCE_MANIFEST}" \
    --training-key-bank "${PILOT_TRAINING_KEYS}" \
    --holdout-key-bank "${PILOT_HOLDOUT_KEYS}" \
    --deployment-key "${PILOT_DEPLOYMENT_KEY}"
fi

if [[ ! -e "${FULL_SOURCE_MANIFEST}" && ! -e "${FULL_TRAINING_KEYS}" && ! -e "${FULL_HOLDOUT_KEYS}" && ! -e "${FULL_DEPLOYMENT_KEY}" ]]; then
  "${CONDA_EXE}" run -p "${CONDA_ENVS_PATH}/WFCLLM" python scripts/wfcllm_prepare_gated_experiment.py \
    --source-catalog "${FULL_SOURCE_CATALOG}" \
    --source-manifest "${FULL_SOURCE_MANIFEST}" \
    --training-key-bank "${FULL_TRAINING_KEYS}" \
    --holdout-key-bank "${FULL_HOLDOUT_KEYS}" \
    --deployment-key "${FULL_DEPLOYMENT_KEY}"
fi

MODEL_ARGS=(
  --generation-model-path "${GENERATION_MODEL_PATH}"
  --semantic-encoder-model-path "${SEMANTIC_ENCODER_MODEL_PATH}"
  --gate-base-model-path "${GATE_BASE_MODEL_PATH}"
  --gate-cache-dir "${CACHE_DIR}"
)
if [[ -n "${REWRITE_MODEL_PATH:-}" ]]; then MODEL_ARGS+=(--rewrite-model-path "${REWRITE_MODEL_PATH}"); fi
if [[ -n "${SEMANTIC_ENCODER_CHECKPOINT_PATH:-}" ]]; then MODEL_ARGS+=(--semantic-encoder-checkpoint-path "${SEMANTIC_ENCODER_CHECKPOINT_PATH}"); fi
if [[ -n "${SEMANTIC_WHITENING_PATH:-}" ]]; then MODEL_ARGS+=(--semantic-whitening-path "${SEMANTIC_WHITENING_PATH}"); fi

PILOT_COMMON=(
  --gate-source-manifest "${PILOT_SOURCE_MANIFEST}"
  --gate-source-catalog "${PILOT_SOURCE_CATALOG}"
  --training-key-bank-file "${PILOT_TRAINING_KEYS}"
  --holdout-key-bank-file "${PILOT_HOLDOUT_KEYS}"
  --secret-key-file "${PILOT_DEPLOYMENT_KEY}"
  "${MODEL_ARGS[@]}"
)
FULL_COMMON=(
  --gate-source-manifest "${FULL_SOURCE_MANIFEST}"
  --gate-source-catalog "${FULL_SOURCE_CATALOG}"
  --training-key-bank-file "${FULL_TRAINING_KEYS}"
  --holdout-key-bank-file "${FULL_HOLDOUT_KEYS}"
  --secret-key-file "${FULL_DEPLOYMENT_KEY}"
  "${MODEL_ARGS[@]}"
)

OFFLINE=(PYTHONPATH=. HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1)
GATE_TRAIN_OVERRIDES=()
if [[ -n "${GATE_EPOCHS}" ]]; then GATE_TRAIN_OVERRIDES+=(--gate-epochs "${GATE_EPOCHS}"); fi
if [[ -n "${GATE_EARLY_STOPPING_PATIENCE}" ]]; then
  GATE_TRAIN_OVERRIDES+=(--gate-early-stopping-patience "${GATE_EARLY_STOPPING_PATIENCE}")
fi

env "${OFFLINE[@]}" "${CONDA_EXE}" run -p "${CONDA_ENVS_PATH}/WFCLLM" python run.py \
  --phase encoder --config "${PILOT_CONFIG}" --run-dir "${PILOT_RUN}" \
  --state-file "${PILOT_STATE}" "${PILOT_COMMON[@]}"

env "${OFFLINE[@]}" "${CONDA_EXE}" run -p "${CONDA_ENVS_PATH}/WFCLLM" python run.py \
  --phase gate-data --config "${PILOT_CONFIG}" --run-dir "${PILOT_RUN}" \
  --state-file "${PILOT_STATE}" \
  --model-device cuda --gate-device cuda "${PILOT_COMMON[@]}"

PILOT_FEASIBILITY="${PILOT_RUN}/gate-data/feasibility_summary.json"
env "${OFFLINE[@]}" "${CONDA_EXE}" run -p "${CONDA_ENVS_PATH}/WFCLLM" python run.py \
  --phase encoder --config "${FULL_CONFIG}" --run-dir "${FULL_RUN}" \
  --state-file "${FULL_STATE}" "${FULL_COMMON[@]}"

env "${OFFLINE[@]}" "${CONDA_EXE}" run -p "${CONDA_ENVS_PATH}/WFCLLM" python run.py \
  --phase gate-data --config "${FULL_CONFIG}" --run-dir "${FULL_RUN}" \
  --state-file "${FULL_STATE}" --pilot-feasibility "${PILOT_FEASIBILITY}" \
  --model-device cuda --gate-device cuda "${FULL_COMMON[@]}"

env "${OFFLINE[@]}" "${CONDA_EXE}" run -p "${CONDA_ENVS_PATH}/WFCLLM" python run.py \
  --phase gate-train --config "${FULL_CONFIG}" --run-dir "${FULL_RUN}" \
  --state-file "${FULL_STATE}" --pilot-feasibility "${PILOT_FEASIBILITY}" \
  --model-device cuda --gate-device cuda "${FULL_COMMON[@]}" "${GATE_TRAIN_OVERRIDES[@]}"

env "${OFFLINE[@]}" "${CONDA_EXE}" run -p "${CONDA_ENVS_PATH}/WFCLLM" python run.py \
  --phase generate --config "${FULL_CONFIG}" --run-dir "${FULL_RUN}" \
  --state-file "${FULL_STATE}" --dataset-path "${DATASET_PATH}" \
  --model-device cuda --gate-device cpu --sample-limit "${SAMPLE_LIMIT}" "${FULL_COMMON[@]}"

CALIBRATION="${FULL_RUN}/calibration/reference_calibration.json"
POSITIVE_INPUT="${FULL_RUN}/inputs/final_code.jsonl"
env "${OFFLINE[@]}" "${CONDA_EXE}" run -p "${CONDA_ENVS_PATH}/WFCLLM" python run.py \
  --phase calibrate --config "${FULL_CONFIG}" --run-dir "${FULL_RUN}" \
  --state-file "${FULL_STATE}" --negative-input "${NEGATIVE_INPUT}" \
  --calibration "${CALIBRATION}" --model-device cuda --gate-device cpu "${FULL_COMMON[@]}"

env "${OFFLINE[@]}" "${CONDA_EXE}" run -p "${CONDA_ENVS_PATH}/WFCLLM" python run.py \
  --phase detect --config "${FULL_CONFIG}" --run-dir "${FULL_RUN}" \
  --state-file "${FULL_STATE}" --input "${POSITIVE_INPUT}" \
  --calibration "${CALIBRATION}" --model-device cuda --gate-device cpu "${FULL_COMMON[@]}"

env "${OFFLINE[@]}" "${CONDA_EXE}" run -p "${CONDA_ENVS_PATH}/WFCLLM" python run.py \
  --phase report --config "${FULL_CONFIG}" --run-dir "${FULL_RUN}" --state-file "${FULL_STATE}"
env "${OFFLINE[@]}" "${CONDA_EXE}" run -p "${CONDA_ENVS_PATH}/WFCLLM" python run.py \
  --phase audit --config "${FULL_CONFIG}" --run-dir "${FULL_RUN}" --state-file "${FULL_STATE}"

echo "completed: ${FULL_RUN}"
