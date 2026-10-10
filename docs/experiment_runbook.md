# 实验运行约定

所有请求 benchmark 都固定 `temperature=0`，并在结果 JSON 中保存 workload、重复次数、并发和错误。E2E、TTFT、TPOT 和 NPU task 时间分开报告；启动、编译和 graph capture 时间不混入稳态请求延迟。

## Graph bucket A/B

```bash
source configs/remote.env
export GRAPH_CASES_FILE="$PROJECT_ROOT/workloads/graph_pad5.json"
export VLLM_PORT=8010
bash scripts/run_graph_metrics.sh
```

需要精确 bucket 时，另起干净进程并设置 `GRAPH_CAPTURE_CONFIG='{"cudagraph_capture_sizes":[1,2,4,5,8]}'`。两组必须使用相同模型、模型 revision、输入、输出长度、预热和重复次数。

## Runtime profiling

先使用 `torch_npu.profiler` 或 `msprof` 采集已有路径，再通过小范围 marker 将 `ACLGraphWrapper` replay 前同步与 Runtime API 时间对应。只有在 eager 对照、Graph 对照和正确性检查都完成后，才考虑修改同步依赖。

## ATB

ATB warm-up 的内存统计属于资源正确性问题。记录 `mem_get_info`、PyTorch allocated/reserved、`npu-smi` HBM 和 KV cache 容量，分别覆盖自动 KV cache 与显式 `--kv-cache-memory` 路径。已有上游 Issue/PR 要在报告中注明，不能把重复实现当作新贡献。

