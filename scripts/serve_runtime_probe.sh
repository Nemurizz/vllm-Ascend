#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"
MODE="${1:-graph}"
PORT="${2:-8020}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export RUNTIME_PROBE_STATS_DIR="${RESULTS_ROOT}/raw/runtime_${MODE}_${RUN_ID}"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
ARGS=(--model "$MODEL_DIR" --served-model-name Qwen3-8B --dtype bfloat16
      --max-model-len 4096 --max-num-seqs 8 --max-num-batched-tokens 1024
      --gpu-memory-utilization 0.80 --host 127.0.0.1 --port "$PORT"
      --no-enable-log-requests --no-enable-prefix-caching
      --worker-cls runtime_probe_worker.RuntimeProbeWorker
      --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${PROFILE_ROOT}/runtime_${MODE}_${RUN_ID}\",\"torch_profiler_with_stack\":false,\"ignore_frontend\":true}")
case "$MODE" in
  graph) ;;
  eager) ARGS+=(--enforce-eager) ;;
  pa) ARGS+=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --additional-config '{"pa_shape_list":[1,2,4,8]}') ;;
  *) echo "Expected graph/eager/pa" >&2; exit 2 ;;
esac
exec python -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
