#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"
echo "project_root=${PROJECT_ROOT}"
echo "model_dir=${MODEL_DIR}"
echo "python=$(python --version 2>&1)"
command -v npu-smi >/dev/null && npu-smi info || true
python - <<'PY'
import importlib.metadata as m
for name in ["vllm", "vllm-ascend", "torch", "torch-npu", "transformers", "triton-ascend"]:
    try: print(f"{name}=={m.version(name)}")
    except m.PackageNotFoundError: print(f"{name}==NOT_INSTALLED")
try:
    import torch
    print("torch.version=", torch.__version__)
    print("torch.npu.available=", torch.npu.is_available())
except Exception as exc:
    print("torch_npu_check_error=", repr(exc))
PY

