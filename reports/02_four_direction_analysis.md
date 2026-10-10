# 四个方向的源码、实机与可复现性分析

调查日期：2026-10-10（Asia/Shanghai）  
调查范围：远端 Ascend 910B2 64GB 单卡、CANN 9.0.0、torch 2.10.0+cpu、torch-npu 2.10.0、vLLM 0.22.1、vLLM-Ascend 0.22.1rc1、Triton Ascend 3.2.1。  
状态：已完成源码和兼容性调查，完成小模型启动、`torch_npu.profiler`、`msprof` 和三个最小实证；尚未修改上游源码。

## 结论边界

当前远端版本对应官方兼容矩阵的 `v0.22.1rc1 / v0.22.1 / CANN 9.0.0 / torch_npu 2.10.0 / Triton Ascend 3.2.1` 行；项目审计的固定参考提交是 v0.23.0.post1，但它要求 CANN 9.1.0 和 torch-npu 2.10.0.post4，因此没有把 v0.23 运行时强行装到 9.0.0 实例上。[官方兼容矩阵](https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html#release-compatibility-matrix)

`Qwen3-0.6B` 已下载到 `/data/vllm_project/models/Qwen3-0.6B`，权重约 1.51GB。使用 vLLM V1、单卡、BF16、`max_model_len=512`、eager 模式完成真实推理：模型加载约 70.6s，首个短请求生成约 21.4s，输出正常。日志还确认本机使用 `device_config=npu`，并且 `enforce_eager` 会关闭编译和 ACLGraph；所以这次只证明模型和基础运行链可用，不把它当成 graph 性能基线。

## 四个方向的判断

### 1. 动态 Batch / ACLGraph

**调用链。** `vllm_ascend/worker/model_runner_v1.py` 的 `execute_model()` 先计算未填充 token 数，再由 `_determine_batch_execution_and_padding()` 选择 `cudagraph_mode` 和 `batch_descriptor`，必要时把 token 数补到有限的 capture bucket。随后通过 `set_ascend_forward_context(..., aclgraph_runtime_mode=..., batch_descriptor=...)` 进入 `vllm_ascend/compilation/acl_graph.py` 的 `ACLGraphWrapper.__call__()`。wrapper 按 `batch_descriptor` 缓存 `torch.npu.NPUGraph`，命中后 replay，未命中时 capture；代码只对已识别的 stream-resource 错误给出指导，没有通用的 capture-failed-to-eager 回退。

**可疑点。** bucket 选择依赖有限的 capture size，但当前执行决策没有一个按 `num_padded_tokens / num_unpadded_tokens` 的门槛。小批量 decode 或长短混合 prefill 可能为了命中图而执行大量 padding，造成图 replay 的 kernel 时间下降、总 token 数和端到端延迟上升。这个是性能假设，不把“有 padding”直接称为 bug；需要从 `CUDAGraphStat`、`msprof` task time 和请求延迟同时证明。

默认 graph 路径已经在 Qwen3-0.6B 上实际运行：编译约 24.7s，FULL/PIECEWISE capture 成功，日志出现 `Replaying aclgraph`；ACL graph pool 实际约 0.30GiB，运行时估算约 0.41GiB，估算偏差约 36.6%。同一脚本已由 `msprof` 成功包住并导出 host/device 数据到 `/root/vllm_project/profiles/qwen3_06b_graph_msprof`。这证明调用链可用，也说明 graph memory 统计本身值得纳入后续实验，但还不足以证明 padding 是主要延迟来源。

**可控方案。** 在 `_determine_batch_execution_and_padding()` 增加可关闭的 `aclgraph_padding_threshold` 和计数器：padding 比例低于门槛时保持现有 FULL/PIECEWISE；超过门槛时走已有 eager/PIECEWISE 路径，不改 NPU 算子和图捕获实现。先用 0.25、0.50 两个阈值做 A/B，只有在代表性负载的 P95/P99 或 TPOT 改善至少 3%、吞吐不下降时才保留。若 padding 比例低或 graph 始终快于 eager，则停止该方向。

**复现矩阵。** Qwen3-0.6B 先跑 128/512/2048 输入、16/64 输出，并发 1/4/8/16；分别跑 `enforce_eager=True`、默认 graph、两组 capture size。记录每步 logical/padded token、选中的 bucket、ACLGraph capture/replay 次数、NPU task time、TTFT、TPOT、P99。使用 `torch_npu.profiler` 看 Python/算子区间，使用 `msprof --ascendcl=on --runtime-api=on --task-time=on --ai-core=on --sys-profiling=on` 看 ACL/Runtime/AI Core 时间。

### 2. Qwen3-VL 多模态

**调用链。** upstream `vllm/multimodal/processing/processor.py` 负责图像处理、placeholder 和 token 插入；`vllm/multimodal/encoder_budget.py` 计算 encoder compute budget、cache size 和每 item token 上限；Ascend `vllm_ascend/worker/encoder_acl_graph.py` 以固定 `token_budget` 捕获并重放 encoder graph。

**已定位的 buffer 生命周期问题。** `EncoderAclGraphManager._run_budget_graph()` 读取当前输入 `src` 和 `n = src.shape[0]`，仅执行 `graph_meta.input_buffer[:n].copy_(src)`（当前远端文件约第 346–350 行）。同一函数对 metadata buffer 则先 `buf.zero_()` 再复制 active slice（约第 351–361 行）。因此固定 budget graph 被较长请求填充后，下一次较短请求不会清理 input buffer 的尾部。这个结论在源码层是确定的；它是否影响 Qwen3-VL 输出取决于 encoder forward 是否会读取尾部，必须做模型级验证。

远端 NPU 最小实证先把 8 行 buffer 写成 7，再只写前 3 行为 2；尾部 5 行仍为 7。脚本是 `/data/vllm_project/src/encoder_tail_repro.py`，结果保存在本次 SSH 输出中。它证明了当前 copy 语义会留下旧数据，不声称该脚本已经等价于 vision encoder 的最终错误输出。

**可控方案。** 在 `n < input_buffer.shape[0]` 时只清理 inactive tail，然后复制 active slice，例如 `input_buffer[n:].zero_()`；`n == capacity` 时不增加清理。新增一个 CPU/NPU buffer 生命周期单测，再用 Qwen3-VL 的两种图像数量和不同 patch token 数做 correctness 对比。若模型输出完全不受尾部影响，可把 zero 操作保留为可选 debug guard 并停止性能改动；若出现错误输出，这是一项中等工作量、可回归验证的正确性修复。

**多模态 Profiling。** 端到端拆为 host decode/resize、processor cache、encoder graph capture/replay、decoder prefill、decode 五段；每段记录墙钟时间、encoder budget、graph miss/hit、active item 数和 token 数。先使用可装入单卡的 Qwen3-VL 小尺寸权重；没有权重时不以文本模型结果代替 vision 结论。

### 3. Runtime / 算子资源统计

**源码证据。** 远端 `vllm_ascend/worker/worker.py` 在 `determine_available_memory()` 的 `memory_profiling` 上下文中先调用 `profile_run()`，然后在约第 504 行读取 `torch.npu.memory_stats()` 的 peak；`_warm_up_atb()` 则在约第 776–779 行、内存 profiling 结束以后才执行。`_warm_up_atb()` 调用 `torch_npu._npu_matmul_add_fp32()`，其 ATB/NNAL `TilingBufferPool` 通过 ACL 分配，可能不进入 PyTorch allocator。

**910B2 实证。** `/data/vllm_project/src/measure_atb_warmup.py` 在清空缓存后测量 `torch.npu.mem_get_info()` 和 `torch.npu.memory_stats()`：

| 阶段 | device used | allocated | reserved |
|---|---:|---:|---:|
| before | 380,297,216 B | 0 B | 0 B |
| 输入 tensor 后 | 386,588,672 B | 1,536 B | 2,097,152 B |
| ATB warm-up 后 | 489,349,120 B | 1,536 B | 2,097,152 B |

因此本机 CANN 9.0.0/910B2 的一次 ATB warm-up 额外占用 102,760,448 B（约 98MiB），而 PyTorch allocated/reserved 计数不变。这与仍未关闭的 vLLM-Ascend #14300 描述的“warm-up 在 profiling 后、TilingBufferPool 不计入 PyTorch memory accounting”一致。[Issue #14300](https://github.com/vllm-project/vllm-ascend/issues/14300)

**排除重复选题。** 上游已有开放 PR [#14302](https://github.com/vllm-project/vllm-ascend/pull/14302)，其目标正是把 `_warm_up_atb()` 移入 `memory_profiling` 并覆盖显式 `--kv-cache-memory` 快路径；当前 PR 仍未合并且存在冲突/评审讨论。因此本项目不能把“移动 warm-up”作为原创修复，只能把 910B2 上的 98MiB 现象作为回归验证基线，或者寻找 PR 未覆盖的剩余路径。

**可控方案。** 把 warm-up 移到 `memory_profiling` 内、`profile_run()` 之后、`profile_torch_peak` 记录之前；显式 `--kv-cache-memory` 快路径也在完成 profile_run 后执行一次 warm-up。随后在同一进程比较 available KV cache、`npu-smi` HBM 和 OOM 边界。这个方案修复的是 KV cache 容量正确性，不承诺直接提高 tokens/s；若实际 workload 没有内存压力，收益表现为消除错误容量和偶发 OOM。

### 4. 业务负载感知 Scheduler

**源码不一致。** 远端基础 scheduler 在 `vllm/v1/core/sched/scheduler.py` 约第 661–667 行，chunked prefill 关闭且 `num_new_tokens > token_budget` 时直接 `break`。Ascend 的 `scheduler_profiling_chunk.py` 约第 538–539 行和 `recompute_scheduler.py` 约第 509–512 行也使用 `break`。但 Ascend `scheduler_dynamic_batch.py` 约第 396–399 行已经采用另一种语义：弹出当前长请求、放入 `step_skipped_waiting`，然后 `continue`，让后面的短请求使用剩余 token budget。

**确定的可复现调度缺口。** 设 `token_budget=512`，waiting 队首长请求需要 4096 token，队列第二个短请求需要 128 token，且关闭 chunked prefill。`break` 会使本轮一个请求都不进入 running；`continue` 方案会跳过长请求并调度短请求。这不是人工极端边界，而是长短请求混合时的确定性 head-of-line blocking。需要确认目标业务的 SLA 是否采用这一策略；动态 batch 已经表明 Ascend 代码库接受“暂不严格 FCFS”的实现。

**可控方案。** 将 profiling/recompute 两个 scheduler 的该分支与 dynamic batch 对齐：`pop_request()`、`step_skipped_waiting.prepend_request(request)`、`continue`，保留本轮末尾的 requeue 顺序。补一条单元测试验证长请求仍在 waiting、短请求被调度、下一轮长请求仍可进入，另测吞吐和 preemption 次数。目标是混合 128/4096 输入、并发 4/8 下 P95/P99 TTFT 至少改善 3%，同时总吞吐不下降；若 workload 只有同长度请求，停止调度改动。

## 选入下一轮的三个假设

1. **Scheduler-HOL（优先级最高，源码缺口已确定）。** 触发条件是 `enable_chunked_prefill=False`、profiling/recompute scheduler、长请求位于短请求前；最小验证不需要模型权重，可先用 scheduler 单测和请求轨迹复现，再用 Qwen3-0.6B 实测 TTFT P99。
2. **Encoder-tail（正确性优先）。** 触发条件是同一 `token_budget` 图先后接收不同 item 数或不同 patch token 数；最小验证是 buffer 生命周期单测，再做 Qwen3-VL 输出一致性。修复范围是一处 inactive-tail 清理和回归测试，风险可控。
3. **ATB-accounting（资源正确性）。** 触发条件是使用 base `_warm_up_atb()` 路径；910B2 已实测约 98MiB 未被 PyTorch 计数。下一步验证小 KV 余量时是否能触发实际 over-allocation/OOM，并核对移动 warm-up 后 KV cache 下降约 98MiB。

ACLGraph padding 作为第四个测量候选保留，不先把它列入主选题：它需要先有 graph/eager 的实际 P99 差异。投机解码方面，Ascend RFC #17139 仍指出 FULL 使用 padded token、PIECEWISE/eager 使用 logical token 的语义差异，但该方向需要额外 draft 权重和 backend-specific correctness 测试，暂不抢占前三项。[RFC #17139](https://github.com/vllm-project/vllm-ascend/issues/17139)

## 已排除或不能直接复用的 Issue / PR

- NPU prefetch static-buffer slot ownership 问题（Ascend #10932）确有历史复现，但对应的上游 vLLM 修复 PR #54975 已于 2026-09-07 合并到 main；它不是新的优化题目。固定远端 v0.22.1 是否包含该回移修复需要单独检查，不能把“旧版本缺修复”包装成新根因。[PR #54975](https://github.com/vllm-project/vllm/pull/54975)
- RFC #17139 仍开放，适合作为投机解码的后续验证项；当前单卡没有 draft 权重，也没有把其 RFC 文字当成实机故障。
- 多卡 DP/EP hang、PP deep-copy、KV connector preemption 等问题与本项目单卡 Qwen3 目标不匹配，或已经有上游修复，不进入前三项。

## Profiling 与验收计划

按用户目录要求，模型权重放在数据盘 `/data/vllm_project/models`；缓存、日志、Profiler 和结果统一放在系统盘 `/root/vllm_project/{cache,profiles,logs,results}`，源码脚本副本在 `/root/vllm_project/src`。早期 smoke 产物仍保留在 `/data/vllm_project/{profiles,results,src}` 作为原始证据。后续运行将 `HF_HOME`、`XDG_CACHE_HOME`、`TRANSFORMERS_CACHE` 和 `TORCH_HOME` 指向 `/root/vllm_project/cache`。

1. **基线。** 对 Qwen3-0.6B 固定 prompt/output，预热后重复 5 次；并发 1/4/8/16，输入 128/512/2048，输出 16/64。记录 TTFT、TPOT、E2E、P50/P95/P99、NPU HBM、PyTorch memory、调度事件和 graph bucket。
2. **ACLGraph。** eager、默认 graph、两个 capture-size 列表三组 A/B；`torch_npu.profiler` 记录 forward、attention、copy 和 replay 区间，`msprof` 导出 `api_statistic`、`task_time`、`op_summary`、AI Core 利用率。
3. **Scheduler。** 构造 128/4096 混合请求和固定到达率，比较 base、dynamic、profiling、recompute 四条路径；重点是队列等待时间、短请求 TTFT P99、preemption、总吞吐。
4. **ATB。** 运行 profile 前、`profile_run` 后、warm-up 后、KV cache 创建后四次 `mem_get_info`；用 `msprof --sys-hardware-mem=on`（若当前 CANN 支持）核对 ACL 侧分配，比较显式和自动 KV cache 两条路径。
5. **多模态。** Qwen3-VL 权重到位后，将 host processor、encoder graph、decoder prefill 分开计时；同一图预算下先长图后短图、先短图后长图，并做 eager/graph correctness 对照。

保留规则：正确性修复必须在 greedy 输出、不同 shape 顺序和重复进程中无 mismatch；性能改动至少带来 3% 的目标指标改善，并且吞吐、显存峰值和长尾不能出现统计显著回退。三项主假设都未达到该门槛前不修改上游实现。

