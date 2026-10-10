#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"
: "${MODEL_DIR:?set MODEL_DIR to a local Qwen3-8B checkpoint}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${RESULTS_ROOT}/raw/baseline_${RUN_ID}"
mkdir -p "$OUT_DIR"
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" --dtype bfloat16 --max-model-len 4096 \
  --max-num-seqs 8 --gpu-memory-utilization 0.90 \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
  2>&1 | tee "$OUT_DIR/server.log" &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
sleep "${SERVER_READY_WAIT:-60}"
python "${SCRIPT_DIR}/benchmark_requests.py" \
  --base-url "http://${VLLM_HOST}:${VLLM_PORT}" \
  --cases "$CASES_FILE" --repeats "${REPEATS:-5}" --seed 0 \
  --output "$OUT_DIR/metrics.json"
echo "results=${OUT_DIR}"

