# Runtime/ATB Profiling：Qwen3-8B ACL Graph

调查日期：2026-10-11（Asia/Shanghai）  
实例：Ascend 910B2 64GB 单卡  
软件：CANN 9.0.0、torch 2.10.0+cpu、torch-npu 2.10.0、vLLM 0.22.1、vLLM-Ascend 0.22.1rc1  
模型：`/data/vllm_project/models/Qwen3-8B`  
状态：完成 Runtime/ATB 实测；没有把未验证的改动合入基线。

## 结论

当前主要可行动的性能假设是 **FULL graph 每个 decode token 的 FIA host-side parameter update**，而不是简单删除 `ACLGraphWrapper` 的 replay 前同步。5 个串行请求、每个生成 128 token 的图模式 trace 中，`update_full_graph_params` 标记覆盖 635 次，累计 6050.7 ms，中位数 9.02 ms；同一 trace 中 replay 前同步累计约 100.0 ms，中位数 0.152 ms。去掉同步的实验会启动并通过健康检查，但首个普通请求 30.02 s 内没有返回，随后已恢复基线。因此同步是正确性/顺序屏障，不能作为未经证明的优化点。

同一 workload 下，图模式输出吞吐为 56.52 token/s，eager 为 11.08 token/s；图模式有效，但剩余成本集中在 attention 参数更新。切换到现有 `FULL_DECODE_ONLY` + paged-attention 配置没有绕开问题：吞吐 56.61 token/s（+0.17%，处于单次 screening 噪声范围），`attention_params_update` 反而升至 7867.5 ms，其中 paged-attention workspace 查询标记累计 2982.5 ms。该配置不作为优化方案。

不过，PA 路径暴露的这个 Runtime/ATB 冗余已经被社区明确提出并解决：上游 Issue [#15575](https://github.com/vllm-project/vllm-ascend/issues/15575) 描述的就是旧版 `attention_v1.py:update_graph_params()` 在每个 attention layer 调用 `_npu_paged_attention_get_workspace(...)`；对应的 [PR #15579](https://github.com/vllm-project/vllm-ascend/pull/15579) 已于 2026-09-04 合并。该 PR 的改法与本实验相同，都是在一次 update pass 内按 workspace 相关签名复用结果，而不是跨 decode step 盲目复用。仓库中的 `patches/runtime_pa_workspace_cache.patch` 因此只能被视为针对 v0.22.1rc1 的回移/复现补丁，不是社区未覆盖的新优化。

这个回移 patch 在相同的 5×128 固定负载上通过了 5/5 请求和 deterministic 文本/usage 对齐：输出吞吐从 56.61 增到 67.80 token/s（+19.77%），E2E P50 从 2264.47 ms 降到 1887.95 ms，TPOT P50 从 17.14 ms 降到 14.24 ms；`attention_params_update` 从 7867.5 ms 降到 4395.7 ms（−44.13%），workspace 查询从 22860 次降到 635 次（−97.22%）。这组数据是对社区 PR 在旧版本、910B2 和本项目 workload 上的独立复现，不应作为原创收益或默认 FIA 的收益声明；远端已恢复原文件，仓库只保存补丁和证据。

补充的 4×256、并发 2 对照也保持方向一致：未修改 PA 为 109.14 token/s、E2E P50 4689.16 ms；workspace cache 为 129.40 token/s（+18.57%）、E2E P50 3953.65 ms（−15.69%）、TPOT P50 14.95 ms（−16.01%）。两组均为 4/4 成功，按 request index 比较生成文本和 usage 完全一致；更新区间从 6515.1 ms 降到 3533.4 ms，workspace 查询从 18360 次降到 510 次。

## 实际调用链

固定 checkout 中的关键入口如下：

1. `vllm_ascend/worker/model_runner_v1.py:_update_full_graph_params_if_needed()` 在 FULL graph 运行时调用 `update_full_graph_params(...)`。
2. `vllm_ascend/compilation/acl_graph.py:update_full_graph_params()` 转到 `attn_backend.get_impl_cls().update_graph_params(...)`。
3. `vllm_ascend/attention/attention_v1.py:AscendAttentionBackendImpl.update_graph_params()` 依照 `graph_params.attn_params[num_tokens]`、`handles` 和 `events` 的并行列表遍历 attention 层；每层都调用 `torch.npu.graph_task_update_begin(update_stream, handle)`、`torch_npu.npu_fused_infer_attention_score_v2.out(...)`、`graph_task_update_end(...)`，然后记录 event。
4. `vllm_ascend/compilation/acl_graph.py:ACLGraphWrapper.__call__()` 在普通 FULL 路径调用 `torch.npu.current_stream().synchronize()`，再执行 `entry.aclgraph.replay()`。源码注释明确该屏障用于防止前一轮 replay 与当前轮参数更新交错。

这个结构解释了 trace：`attention_params_update` 每个 replay 都重新发起一组按层的 FIA task update，而 replay submit 自身只有约 59.4 ms。`msprof` 导出的 `op_statistic.csv` 进一步显示：

| 项目 | 次数 | host 累计 | device 累计 |
|---|---:|---:|---:|
| `aclnnInnerFusedInferAttentionScore` | 23040 | 1117.75 ms | — |
| `aclnnInnerFusedInferAttentionScoreGetWorkspaceSize` | 23040 | 456.36 ms | — |
| `FusedInferAttentionScore` | 23040 | — | 526.95 ms |
| `aclmdlRICaptureTaskUpdateBegin/End` | 22860/22860 | 60.37/64.87 ms | — |
| `aclrtSynchronizeStreamWithTimeout` | 635 | 13.26 ms | — |

host 标记比这些 API 累计更大，说明 Python 参数重建、弱引用解包、按层循环和 runtime task update 组合也占有明显成本；不能把整个 6.05 s 归因给某一个 ACL API。

## A/B 与止损

| 配置 | 成功 | 输出吞吐 | TPOT P50 | 结论 |
|---|---:|---:|---:|---|
| 默认 FULL/FIA | 5/5 | 56.52 token/s | 16.32 ms | 基线图路径 |
| eager | 5/5 | 11.08 token/s | 88.82 ms | 图路径仍有约 5.1× 吞吐优势 |
| FULL_DECODE_ONLY/PA | 5/5 | 56.61 token/s | 17.14 ms | 没有可见收益，更新区间更大 |
| FULL + 去 replay 前同步 | 健康检查通过；请求未完成 | — | — | 30 s 超时，立即回滚 |

每个成功请求都返回 128 completion tokens、无错误；生成文本被保存在原始 JSON 中。实验命令和轻量证据文件位于：

- `scripts/serve_runtime_probe.sh`
- `scripts/runtime_probe_worker.py`
- `scripts/benchmark_runtime.py`
- `evidence/runtime_atb_summary.json`
- `evidence/runtime_atb_worker_graph_probe1.json`
- `evidence/runtime_atb_worker_pa_probe.json`
- `evidence/runtime_atb_benchmark_graph_probe1.json`
- `evidence/runtime_atb_benchmark_eager_probe.json`
- `evidence/runtime_atb_benchmark_pa_probe.json`
- `evidence/runtime_atb_benchmark_pa_cache_probe.json`
- `evidence/runtime_atb_nosync_failure.json`
- `evidence/runtime_atb_worker_pa_cache_probe.json`
- `evidence/runtime_atb_benchmark_pa_mix_base.json`
- `evidence/runtime_atb_benchmark_pa_mix_cache.json`
- `evidence/runtime_atb_worker_pa_mix_base.json`
- `evidence/runtime_atb_worker_pa_mix_cache.json`

远端保留完整 `torch_npu.profiler` trace：`/root/vllm_project/profiles/runtime_graph_probe1`、`/root/vllm_project/profiles/runtime_eager_probe_eager` 和 `/root/vllm_project/profiles/runtime_pa_probe_pa`；仓库只保存可审计的摘要，避免提交百 MB 级数据库。实验结束时已停止服务，NPU HBM 回到约 3429 MB 的空闲占用，安装目录的 `acl_graph.py` 已恢复原始副本。

## 上游状态与可行改动边界

上游 [ACL Graph 设计文档](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/developer_guide/Design_Documents/ACL_Graph.md) 把 host-side attention update、`graph_task_update_*` 和 replay ordering 列为 Ascend full graph 的核心机制。[Issue #13058](https://github.com/vllm-project/vllm-ascend/issues/13058) 正在讨论把任务所有权从“按 layer 顺序的平行列表”移到 graph-owned registry；其描述的主要问题正好对应本 checkout 的 `attn_params/handles/events` 重建路径。当前主分支已经出现 `UpdatableGraph` 原型，但该设计经历过多次合并/回滚，不能直接当成 0.22.1rc1 的小补丁，也不能跳过本机正确性验证。

因此下一步不应再把 PA workspace cache 当作新的上游选题。默认 FIA 的 graph-owned task registry 也已有上游 RFC [#13058](https://github.com/vllm-project/vllm-ascend/issues/13058) 和当前主线的 `UpdatableGraph` 实现；它可以作为旧版本兼容性研究，但不能声称社区没有类似方案。若继续做旧版本回移，范围应限定为：在 v0.22.1rc1 上复现 [PR #15579](https://github.com/vllm-project/vllm-ascend/pull/15579) 的行为，补齐长输出、混合 batch、异构 attention shape 和多 capture size 回归，并记录与主线的差异。若目标是形成新的上游贡献，则必须另找尚未有等价 PR/主线实现的 Runtime/ATB 问题。删除 replay 同步的实验仍只是本项目的负向安全证据：首个请求 30 s 超时，不能作为优化建议。

这条路线的收益上限来自 6.05 s/11.32 s 的局部区间，理论上比删除 100 ms 的 replay sync 更有空间；但 graph-owned registry 仍属于中等偏难的 Runtime 改动，必须先做最小 FIA-only patch、短请求正确性、长 decode、长短混合和多进程回归，不能承诺收益。
