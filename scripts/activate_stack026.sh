#!/bin/bash
# Source in a new shell; this does not alter the old environment or shell profiles.
PROJECT_ROOT=${PROJECT_ROOT:-/root/vllm_project}
SDK_ROOT=${SDK_ROOT:-/opt/vllm-project-sdk}
# Vendor set_env scripts read unset shell variables. Preserve the caller's nounset
# setting while sourcing them; using set -u inside those scripts aborts activation.
case $- in *u*) _stack026_restore_nounset=1; set +u ;; *) _stack026_restore_nounset=0 ;; esac
unset PYTHONPATH ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME ASCEND_OPP_PATH
unset ASCEND_AICPU_PATH ASCEND_RT_VISIBLE_DEVICES VLLM_TARGET_DEVICE
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/lib
source "$SDK_ROOT/cann/set_env.sh"
source "$SDK_ROOT/nnal/atb/set_env.sh"
if [ "$_stack026_restore_nounset" = 1 ]; then set -u; fi
unset _stack026_restore_nounset
export PATH="$PROJECT_ROOT/envs/vllm026/bin:$PATH"
# Explicit source priority avoids an older package shadowing an editable install
# when the baseline's general Python dependencies are reused read-only.
export PYTHONPATH="$PROJECT_ROOT/src/stack026/vllm:$PROJECT_ROOT/scripts:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 TASK_QUEUE_ENABLE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
