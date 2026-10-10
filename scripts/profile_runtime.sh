#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"
: "${PROFILE_COMMAND:?set PROFILE_COMMAND to the vLLM command to profile}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT="${PROFILE_ROOT}/runtime_${RUN_ID}"
mkdir -p "$OUT"
msprof --ascendcl=on --runtime-api=on --task-time=on --ai-core=on --sys-profiling=on --output="$OUT" bash -lc "$PROFILE_COMMAND"

