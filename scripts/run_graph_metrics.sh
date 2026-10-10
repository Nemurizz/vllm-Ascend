#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"
PORT="${1:-${VLLM_PORT}}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_ROOT}/qwen3_8b_graph_${RUN_ID}.log"
OUT_FILE="${RESULTS_ROOT}/raw/qwen3_8b_graph_${RUN_ID}.json"
CASES="${GRAPH_CASES_FILE:-${PROJECT_ROOT}/workloads/graph_pad5.json}"
setsid bash "${SCRIPT_DIR}/serve_graph_metrics.sh" "$PORT" >"$LOG_FILE" 2>&1 < /dev/null &
pid=$!
cleanup() { kill -- -"$pid" 2>/dev/null || true; sleep 3; }
trap cleanup EXIT
ready=0
for _ in $(seq 1 180); do
  if python -c "import urllib.request; urllib.request.urlopen('http://${VLLM_HOST}:${PORT}/health', timeout=1)" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done
if [[ "$ready" -ne 1 ]]; then
  tail -80 "$LOG_FILE" >&2 || true
  exit 2
fi
python "${SCRIPT_DIR}/benchmark_concurrent.py" --base-url "http://${VLLM_HOST}:${PORT}" --cases "$CASES" --repeats "${REPEATS:-2}" --concurrency "${CONCURRENCY:-5}" --output "$OUT_FILE"
grep -E 'CUDAGraph|cudagraph|num_padd|Padded|padded|Replaying|metrics|error|Error' "$LOG_FILE" | tail -200 || true
echo "metrics=${OUT_FILE}"
echo "log=${LOG_FILE}"

