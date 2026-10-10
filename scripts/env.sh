#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export DATA_ROOT="${DATA_ROOT:-/data/vllm_project}"
export SYSTEM_ROOT="${SYSTEM_ROOT:-${HOME}/vllm_project}"
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
export MODEL_REVISION="${MODEL_REVISION:-b968826d9c46dd6066d109eabc6255188de91218}"
export MODEL_DIR="${MODEL_DIR:-${DATA_ROOT}/models/Qwen3-8B}"
export RESULTS_ROOT="${RESULTS_ROOT:-${SYSTEM_ROOT}/results}"
export LOG_ROOT="${LOG_ROOT:-${SYSTEM_ROOT}/logs}"
export PROFILE_ROOT="${PROFILE_ROOT:-${SYSTEM_ROOT}/profiles}"
export CACHE_ROOT="${CACHE_ROOT:-${SYSTEM_ROOT}/cache}"
export VLLM_ENV_ROOT="${VLLM_ENV_ROOT:-/data/vllm-ascend-env}"
export VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
export VLLM_PORT="${VLLM_PORT:-8010}"
export CASES_FILE="${CASES_FILE:-${PROJECT_ROOT}/workloads/text_cases.json}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${CACHE_ROOT}/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${CACHE_ROOT}/hub}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
mkdir -p "$RESULTS_ROOT/raw" "$RESULTS_ROOT/summary" "$LOG_ROOT" "$PROFILE_ROOT" "$CACHE_ROOT"
if [[ -f "${VLLM_ENV_ROOT}/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VLLM_ENV_ROOT}/bin/activate"
fi

