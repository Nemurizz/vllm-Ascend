#!/usr/bin/env bash
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"
OUT="${1:-${RESULTS_ROOT}/env_$(date +%Y%m%d_%H%M%S).txt}"
mkdir -p "$(dirname "$OUT")"
{
  echo "# time"; date -Is
  echo "# host"; uname -a; uname -m
  echo "# cpu"; lscpu || true
  echo "# disk"; df -h
  echo "# npu"; npu-smi info || true
  echo "# ascend install"; cat /etc/ascend_install.info 2>/dev/null || true
  echo "# python"; python --version
  python - <<'PY'
import importlib.metadata as m
names = ["vllm", "vllm-ascend", "torch", "torch-npu", "transformers", "triton-ascend"]
for n in names:
    try: print(f"{n}=={m.version(n)}")
    except m.PackageNotFoundError: print(f"{n}==NOT_INSTALLED")
try:
    import torch
    print("torch.version=", torch.__version__)
    print("torch.npu.available=", torch.npu.is_available())
except Exception as e: print("torch_npu_check_error=", repr(e))
PY
} 2>&1 | tee "$OUT"
echo "saved: $OUT"
