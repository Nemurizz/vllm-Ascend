# ACLGraph 复查与 Runtime/ATB 转向

调查日期：2026-10-11（Asia/Shanghai）  
远端环境：Ascend 910B2 64GB 单卡、CANN 9.0.0、torch 2.10.0+cpu、torch-npu 2.10.0、vLLM 0.22.1、vLLM-Ascend 0.22.1rc1、Triton Ascend 3.2.1。  
模型：`/data/vllm_project/models/Qwen3-8B`。  
状态：完成一次真实 padding 观测和精确 bucket A/B；没有提交源码修改。

## 1. ACLGraph 调用链和当前证据

旧版固定环境的执行路径是：

1. `vllm_ascend/worker/model_runner_v1.py:_determine_batch_execution_and_padding()` 先计算 logical token 数，再调用 `cudagraph_dispatcher.dispatch()` 选择 `CUDAGraphMode` 和 `BatchDescriptor`；约第 2893 行开始写入 `CUDAGraphStat`，记录 `num_unpadded_tokens`、`num_padded_tokens` 和 `num_paddings`。
2. `set_ascend_forward_context()` 把 `batch_descriptor` 和 runtime mode 传入模型执行上下文。
3. `vllm_ascend/compilation/acl_graph.py:ACLGraphWrapper.__call__()` 以 `batch_descriptor` 为 key 缓存 `torch.npu.NPUGraph`；第一次命中时 capture，后续命中时 replay。
4. FULL graph replay 前，旧版在 `enable_enpu=False` 且不是 EAGLE draft graph 时调用 `torch.npu.current_stream().synchronize()`。源码注释说明这是为了保证异步调度或多线程场景中 `update_attn_params` 与前一轮 graph replay 的顺序。

最后一个同步点是当前最值得用 Runtime trace 验证的候选。它不能直接删除：源码把它当成参数更新的顺序屏障，且只在 ENPU 或 FULL+EAGLE draft 路径跳过。可控的下一步是测量该同步占用的 host/API 时间和它是否实际等待 device；先做观测，再决定是否需要更细粒度的事件依赖。

## 2. Graph bucket/padding 复现

工作负载是 5 个相同短请求、并发 5、每请求生成 32 tokens、重复 2 次。服务只改变 capture size 列表：

| A/B | capture sizes | FULL 统计 | wall time | 吞吐 | 每 case E2E P50 平均 | 每 case TTFT P50 平均 | 结果 |
|---|---|---:|---:|---:|---:|---:|---|
| 默认 | `[1,2,4,8]` | `5 -> 8`，3 个 padding，60 次 | 1.3843 s | 7.224 req/s | 688.58 ms | 217.01 ms | 10/10 成功 |
| 精确 bucket | `[1,2,4,5,8]` | `5 -> 5`，0 个 padding，60 次 | 1.3757 s | 7.269 req/s | 684.27 ms | 226.21 ms | 10/10 成功 |

精确 bucket 的吞吐变化约 +0.62%，E2E P50 平均变化约 -0.63%；TTFT 没有改善。样本只有 2 次重复，结果只能作为 screening，不能作为统计显著性结论。原始 JSON 和服务日志分别为：

- `outputs/phase1/evidence/aclgraph_followup/qwen3_8b_graph_pad5_metrics_valid.json`
- `outputs/phase1/evidence/aclgraph_followup/qwen3_8b_graph_metrics_valid.log`
- `outputs/phase1/evidence/aclgraph_followup/qwen3_8b_graph_pad5_exact5_metrics.json`
- `outputs/phase1/evidence/aclgraph_followup/qwen3_8b_graph_metrics_exact5.log`

这证明 padding 现象真实存在，但在当前 Qwen3-8B、5 请求短批次上没有达到项目的 3% 性能门槛。增加 bucket 还会增加 capture 数量和启动阶段编译/显存成本，因此停止“为每个常见 batch 增加精确 bucket”作为当前优化方案。只有在目标业务显示更高 padding 比例，或在更长 decode 的 profile 中出现明确的 device time 差异时，才重新开启该实验。

## 3. 上游问题排除

- 原先关联的 [Issue #16049](https://github.com/vllm-project/vllm-ascend/issues/16049) 已由 [PR #16970](https://github.com/vllm-project/vllm-ascend/pull/16970) 合并；本地 `[1,2,4,8,12]` patch 只保留为旧版回移/回归证据，不作为新的上游选题。
- [Issue #16637](https://github.com/vllm-project/vllm-ascend/issues/16637) 仍是有价值的投机解码 padding 问题：动态 speculative token schedule 会让 FULL graph 失去统一 decode padding，报告中给出约 3 倍 FULL graph 慢化。但当前单卡没有 draft model/权重，不能把它伪装成 Qwen3-8B 的现成复现；它保留为条件性后续实验。
- [Issue #13058](https://github.com/vllm-project/vllm-ascend/issues/13058) 的 host-side graph 参数更新是设计/RFC 级背景，暂不改动。
- ATB warm-up 的约 98 MiB 未计入 PyTorch allocator 已在前一阶段实测，对应 [Issue #14300](https://github.com/vllm-project/vllm-ascend/issues/14300) 和开放 [PR #14302](https://github.com/vllm-project/vllm-ascend/pull/14302)，不能重复包装成原创修复。

## 4. 下一步 Runtime/ATB profiling 计划

先不改 `synchronize()`，按同一 workload 做三组 trace：

1. 默认异步调度、FULL graph replay；
2. 相同配置但强制 eager，用于扣除 graph replay 之外的 Runtime API 基线；
3. 在本地实验副本中仅增加可观测标记，记录 replay 前同步的 host 区间，不改变同步语义。

使用 `torch_npu.profiler` 记录 Python/forward/replay 区间，使用 `msprof --ascendcl=on --runtime-api=on --task-time=on --ai-core=on --sys-profiling=on` 导出 Runtime API、ACL task、AI Core 和 host/device gap。每组固定预热次数，至少重复 5 次，记录 TTFT、TPOT、E2E P50/P95/P99、graph replay 次数、padding token 数、HBM 峰值和同步区间时长。

若同步区间在 device idle 时仍占据可观比例，再检查是否能用已有事件依赖替换全流同步；正确性验收必须覆盖连续短请求、长短混合、参数更新和重复进程。若同步只占总 decode 时间很小的比例，停止 ACLGraph replay 改动，转向 ATB warm-up 的剩余覆盖路径或 scheduler-HOL。

## 5. 现有 msprof 轨迹的 Runtime 初筛

为避免重新启动服务，先导出了前一轮 Qwen3-8B graph 服务的已有轨迹。该轨迹是并发 1 的三 case workload，不能和上面的 5 并发精确 bucket A/B 混为同一实验；它只用于确认 Runtime API 是否值得继续测量。

`api_statistic` 和 `ApiData` 的结果一致指向同步调用占用：

| ACL Runtime API | 次数 | 累计时间 | P50 | 最大 |
|---|---:|---:|---:|---:|
| `aclrtSynchronizeStreamWithTimeout` | 317 | 544.16 ms | 1.792 ms | 2.561 ms |
| `aclrtSynchronizeStream` | 658 | 16.13 ms | 0.009 ms | 10.100 ms |
| `aclrtSynchronizeDeviceWithTimeout` | 294 | 14.71 ms | 0.036 ms | 3.616 ms |
| `aclrtSynchronizeEvent` | 651 | 15.38 ms | 0.007 ms | 4.569 ms |

`aclrtSynchronizeStreamWithTimeout` 是最明显的长同步项；317 次与该轨迹中的连续 decode 迭代数处于同一量级。这个结果支持“replay 前同步可能是 Runtime 瓶颈”的假设，但不能把所有 544 ms 都归给 `ACLGraphWrapper.__call__()`，因为服务启动、模型执行和其他同步也在同一进程中。原始数据保存在：

- `outputs/phase1/evidence/aclgraph_followup/api_statistic_graph_msprof.csv`
- `outputs/phase1/evidence/aclgraph_followup/api_event_graph_msprof.db`
- `outputs/phase1/evidence/aclgraph_followup/runtime_sync_profile_summary.json`

下一次只需加一个 replay 前后的 host marker，重新跑相同请求，再把 marker 区间与 `aclrtSynchronizeStreamWithTimeout` 做时间重叠匹配，就能区分“ACLGraph 屏障本身”与其他 Runtime 同步。当前不对 `synchronize()` 做删除或替换。

## 6. 当前决策

“精确 graph bucket 消除 padding”已经完成实证，但当前 workload 的收益低于 3%，不进入源码修改阶段。下一阶段的主假设是 **FULL graph replay 前的 Runtime 同步是否造成可测的 host/device 空洞**；ATB 资源统计作为第二条线，沿用前一阶段 98 MiB 现象做回归和边界验证，不重复已有 PR 的目标。
