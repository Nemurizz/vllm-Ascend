#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"
PORT="${1:-${VLLM_PORT}}"
ARGS=(--model "$MODEL_DIR" --served-model-name Qwen3-8B --dtype bfloat16 --max-model-len 4096 --max-num-seqs 8 --max-num-batched-tokens 1024 --gpu-memory-utilization 0.80 --host "$VLLM_HOST" --port "$PORT" --no-enable-log-requests --cudagraph-metrics)
if [[ -n "${GRAPH_CAPTURE_CONFIG:-}" ]]; then ARGS+=(--compilation-config "$GRAPH_CAPTURE_CONFIG"); fi
exec python -m vllm.entrypoints.openai.api_server "${ARGS[@]}"

