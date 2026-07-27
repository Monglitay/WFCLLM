#!/usr/bin/env bash
set -euo pipefail
WFCLLM_LANGUAGE="python"
WFCLLM_DATASET="mbpp"
WFCLLM_PROFILE="full"
WFCLLM_CONFIG="configs/wfcllm/experiments/python_mbpp_full.json"
export WFCLLM_LANGUAGE WFCLLM_DATASET WFCLLM_PROFILE WFCLLM_CONFIG
exec "$(dirname "${BASH_SOURCE[0]}")/run_gated_experiment.sh" "$@"
