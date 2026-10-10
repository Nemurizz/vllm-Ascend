#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"
command -v huggingface-cli >/dev/null || { echo "huggingface-cli is required" >&2; exit 1; }
mkdir -p "${MODEL_DIR}"
huggingface-cli download "$MODEL_ID" --revision "$MODEL_REVISION" --local-dir "$MODEL_DIR" --local-dir-use-symlinks False
echo "model_dir=${MODEL_DIR}"

