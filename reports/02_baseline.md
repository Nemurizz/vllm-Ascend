# 阶段 2：Qwen3-8B 单卡基线

状态：已完成第一轮 Qwen3-8B Eager/默认 Graph 基线；尚未据此修改上游实现或决定最终选题。

## 固定环境

- NPU：Ascend 910B2，单卡，NPU ID 7
- Python：3.11.15
- CANN：9.0.0；驱动/npu-smi：26.1.1
- vLLM：0.22.1，commit identity 记录在远端 `/root/vllm_project/evidence/runtime_source_identity.json`
- vLLM-Ascend：0.22.1rc1，commit identity 记录在同一文件
- torch / torch-npu：2.10.0+cpu / 2.10.0
- 模型：`Qwen/Qwen3-8B`，revision `b968826d9c46dd6066d109eabc6255188de91218`
- 权重：`/data/vllm_project/models/Qwen3-8B`
- 系统盘结果：`/root/vllm_project/results/raw`、`/root/vllm_project/profiles`、`/root/vllm_project/logs`

服务启动实际使用 `/root/vllm_project/scripts/serve_qwen3_8b.sh`。vLLM 0.22.1 的实际参数是 `--no-enable-log-requests`；旧参数 `--disable-log-requests` 会在启动阶段直接退出，已在实验脚本中修正。

## 实验矩阵

本轮每个 case 重复 3 次；服务启动和编译时间单独记录，不混入请求延迟。原始结果保存在远端系统盘 `/root/vllm_project/results/raw/`，包括 `qwen3_8b_eager_c1.json`、`qwen3_8b_eager_c4.json`、`qwen3_8b_eager_c8.json`、`qwen3_8b_graph_c1.json`、`qwen3_8b_graph_c4.json` 和 `qwen3_8b_graph_c8.json`。

| 组 | Graph | Chunked prefill | Capture sizes | 并发 | 输入/输出 |
|---|---|---|---|---:|---|
| EAGER | `--enforce-eager` | on | none | 1/4/8 | 128/32、约 2k/32、128/256 |
| GRAPH-DEFAULT | 默认 | on | 默认 | 1/4/8 | 同上 |
| GRAPH-BUCKET | 默认 | on | `[1,2,4,8,16]` | 1/4/8 | 同上（待单独跑） |
| SCHED-HOL | 默认 | off | 默认 | 2/4/8 | 长请求先到、短请求后到（条件不满足时记录为不可复现） |

记录 TTFT P50/P95/P99、TPOT、E2E、输出 token/s、请求/s、NPU HBM、PyTorch memory、graph capture/replay、padding token、scheduler waiting/preemption 事件。Eager 和 Graph 只在同一权重 revision、同一 sampling、同一 workload 下比较。

## 已执行的准备检查

- Qwen3-0.6B eager smoke 成功；默认 FULL/PIECEWISE ACLGraph smoke 成功。
- `torch_npu.profiler` smoke 已生成 CANN trace。
- `msprof` 已包住 Qwen3-0.6B graph smoke，host/device 数据导出成功。
- 远端源码 identity、版本和关键文件 SHA256 已保存。

## 后续验收门槛

1. 为 Graph/Eager 增加一次固定 prompt 的 token 级 greedy 对齐；当前基线只保存了响应长度和无错误状态。
2. 在并发 8 及 `[1,2,4,8,16]` capture bucket 下补充同一 workload，使用 msprof/运行日志拆分 padding、ACLGraph replay、Runtime API 和 AI Core 时间。
3. 若转向 Qwen3-VL，先下载固定 revision 并分别测 vision encoder、host 预处理、prefill 和 decode；没有真实端到端差异前不改 encoder buffer 生命周期。
4. 只有当 Scheduler-HOL 或 ACLGraph padding 在目标混合负载上达到 ≥3% 目标指标改善，或正确性问题在真实模型上复现，才创建 `03_topic_decision.md`。ACLGraph off-stride 桶缺口已经在 910B2 上达到门槛，因此先完成了安装副本中的最小 patch 验证；扩大回归矩阵前不提交上游。

## Qwen3-8B 实测结果

所有请求均返回成功，Eager 和 Graph 的请求脚本、sampling 参数、模型 revision 相同；服务启动/编译没有计入下表。`c1/c4` 表示客户端并发数。

| 模式/并发 | long-in/short-out E2E P50 | short-in/long-out E2E P50 | short-in/short-out E2E P50 | 请求/s |
|---|---:|---:|---:|---:|
| Eager / c1 | 2453 ms | 19575 ms | 2401 ms | 0.104 |
| Graph / c1 | 515 ms | 3751 ms | 532 ms | 0.611 |
| Eager / c4 | 2670 ms | 20767 ms | 2726 ms | 0.346 |
| Graph / c4 | 695 ms | 4042 ms | 622 ms | 1.721 |
| Eager / c8 | 3206 ms | 21574 ms | 3204 ms | 0.417 |
| Graph / c8 | 1030 ms | 4426 ms | 1029 ms | 2.030 |

在本 workload 上，默认 Graph 的 E2E P50 相对 Eager 下降约 68%–81%，请求/s 提升约 5.0 倍（c1）、5.0 倍（c4）和 4.9 倍（c8）。这是已观测到的后端基线差异，不等于新的可提交优化；需要进一步拆分编译、ACLGraph replay、padding 和 kernel 时间，排除 workload/缓存因素。Graph c1 与 Eager c1 的输出字符数逐请求一致；c4/c8 因请求完成顺序不同，已保留原始响应长度，完整 token 级对齐仍是后续正确性门槛。

代表性 Graph 服务已由 `msprof` 包裹，轨迹目录为远端 `/root/vllm_project/profiles/qwen3_8b_graph_msprof/PROF_000001_20261010153545563__00022976HLCLFLCJ/`，日志中确认至少一次 `Replaying aclgraph`。采集包含 ACL、Runtime API、task time、AI Core 和系统采样。

默认配置日志明确记录 `cudagraph_capture_sizes=[1,2,4,8]`、`max_cudagraph_capture_size=8`，并提示 ACL graph memory profiling 会把 `--gpu-memory-utilization=0.80` 的有效值折算为约 0.7913。并发 8 的 Graph 请求/s 仍为 2.030，暂未看到默认桶在该负载下失效。

## Scheduler-HOL 条件检查

尝试以 `--no-enable-chunked-prefill` 启动 v0.22.1 时，SchedulerConfig 强制要求 `max_num_batched_tokens >= max_model_len`。因此“token budget 小于长请求、长请求触发源码中的 `break`”在标准 API 配置下不能直接构造；把长输入推到 4096 以上又会先被上下文长度校验拒绝。探针结果保留在远端 `/root/vllm_project/results/raw/qwen3_8b_no_chunk_hol.json`，长请求为 HTTP 400，短请求正常。这一方向目前不能称为已复现性能 Bug，后续只有在动态 batch / profiling chunk 等专用 scheduler 实际启用时再测。

## 阶段门槛结论

- Graph/Eager 的差异已经超过 3%，但它是现有实现的基线行为；当前没有证据表明简单调整 bucket 或 padding 能在目标负载上再取得收益。
- Runtime/ATB 统计偏差已经有上游 Issue #14300 和开放 PR #14302，不能作为新的独立题目；本项目可用本机实测做回归验证。
- Encoder ACLGraph 输入尾部复用的最小 NPU 复现已成立，但尚未在 Qwen3-VL 真实请求上观察到输出错误或性能损失。
- 因此暂不创建 `03_topic_decision.md`。Graph bucket/padding 已进入 patch 验证阶段；Qwen3-VL 和更大 bucket 矩阵仍需单独 profiling，不能由当前单点结果代替。

Graph bucket/padding profiling 的实测已记录在 [03_graph_bucket_profile.md](03_graph_bucket_profile.md)：`max_num_seqs=12` 的默认桶截断到 8，补齐 12 后 910B2 Qwen3-8B 的 12 路 E2E P50 从 2497 ms 降到 760 ms；安装副本中的最小 patch 自动补齐该桶后得到 763 ms，并通过确定性流式输出对齐。该缺陷已有上游 PR #16970 合并修复，因此本项目保留旧版本回归证据，不创建重复上游提交。

