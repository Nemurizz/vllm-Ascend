#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"
VLLM_SPEC="${VLLM_SPEC:-vllm==0.22.1}"
VLLM_ASCEND_SPEC="${VLLM_ASCEND_SPEC:-vllm-ascend==0.22.1rc1}"
TRITON_ASCEND_SPEC="${TRITON_ASCEND_SPEC:-triton-ascend==3.2.1}"
echo "Installing Python packages with ${PIP_INDEX_URL}. CANN and torch-npu must come from the image."
python -m pip install --index-url "$PIP_INDEX_URL" "$VLLM_SPEC" "$VLLM_ASCEND_SPEC" "$TRITON_ASCEND_SPEC"
"${SCRIPT_DIR}/check_stack.sh"

