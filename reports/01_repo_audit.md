# 阶段 1：仓库、版本与兼容性审计

审计日期：2026-10-10（Asia/Shanghai）  
状态：源码审计和兼容性检查已完成；用户授权的远端小模型与 Profiler 验证记录在 [02_four_direction_analysis.md](02_four_direction_analysis.md)，上游源码仍未修改。

## 结论边界

本工作区是 projectless Windows 目录，没有 vLLM/vLLM-Ascend checkout；源码结论来自官方 GitHub 固定提交。已通过用户提供的 SSH 访问实际 Linux 910B 实例，并将实机检查单独列出。项目文档要求把观察事实、推测和实机证据分开，本文遵守这一点。

## 版本与兼容性

| 组件 | 审计值 | 证据 |
|---|---|---|
| vLLM Ascend | v0.23.0.post1，commit `1cdb8c4db6e50f36f1fb3283b3e3dd618f6821e` | [tag](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.23.0.post1) |
| upstream vLLM | v0.23.0，commit `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` | [tag](https://github.com/vllm-project/vllm/releases/tag/v0.23.0) |
| Python | 官方安装要求 `>=3.10,<3.13`；发布验证表使用 3.12 | [installation](https://docs.vllm.ai/projects/ascend/en/latest/getting_started/installation.html#hardware-and-software-stack) |
| PyTorch / torch_npu | 2.10.0 / 2.10.0.post4 | 同上 |
| CANN / NNAL | CANN 9.1.0；NNAL 9.1.0 | 同上 |
| Triton Ascend | 3.2.2（A2/A3/950DT；310P/200I 不使用） | 同上 |
| 本地工作区 | Windows 11，Python 3.12.7；无上游 checkout | 本地命令输出 |
| 远端验证环境 | aarch64；Python 3.11.15；910B2 单卡；驱动/npu-smi 26.1.1；CANN 9.0.0；torch 2.10.0+cpu / torch-npu 2.10.0 | `outputs/phase1/evidence/remote_env_20261010.txt` |
| 远端安装目标 | vLLM 0.22.1、vLLM-Ascend 0.22.1rc1、Triton Ascend 3.2.1 | 官方矩阵与远端包版本；Qwen3-0.6B 小模型已放在数据盘 |

官方安装页列出的硬件是 Atlas A2/A3、310P/200I 和 950DT，没有把“910B 64GB”作为独立验证项。因此 910B 的准确产品/驱动分支必须由 `npu-smi info`、`/etc/ascend_install.info` 和镜像标签确认，不能仅凭显存容量推断它对应哪一行兼容矩阵。Qwen3 Dense 8B 和 Qwen-VL Dense 文档入口存在，但仍需要实机启动验证：权重可装入不等于模型、算子、图模式均兼容。

## 关键调用链（固定提交）

### Graph / Runtime

1. vLLM 的通用静态图配置和 batch descriptor 选择发生在 upstream worker；平台通过 `vllm_ascend.platform.NPUPlatform.get_static_graph_wrapper_cls()` 选择 `vllm_ascend.compilation.acl_graph.ACLGraphWrapper`（设计文档明确写出该边界）。
2. `vllm_ascend/worker/model_runner_v1.py:1950` 的 `execute_model()` 组织一次 worker 执行，`:2834` 的 `_model_forward()` 调模型；`:5101` 的 `capture_model()` 负责预热/捕获配置的图尺寸。该 runner 读取 `compilation_config.cudagraph_capture_sizes`，并在 `:678` 附近计算 DP/graph padding。
3. `vllm_ascend/compilation/acl_graph.py:65` 的 `ACLGraphWrapper` 按运行模式和 batch descriptor 建立缓存；`:138` 的 `__call__` 在 cache miss 中进入 capture，在命中时检查输入地址并 replay。`:185` 同步 offloader copy stream，`:209` 将已知 stream-resource capture 错误重新抛出；源码没有一个“一律捕获失败就 eager fallback”的承诺。
4. 图参数在 `acl_graph.py:270-371` 维护（full、draft、draft-prefill 的 workspace/attention 参数）。设计文档说明 graph 只能覆盖有限 capture buckets；运行 token 数大于最大 bucket 时会走图外路径/报错，具体分支需在目标版本和配置实机确认。当前设计文档还说明旧的本地 `update_aclgraph_sizes()` 收缩启发式已移除，捕获失败应优先调 `cudagraph_capture_sizes` / `max_cudagraph_capture_size`。

### 多模态

1. upstream `vllm/multimodal/processing/processor.py` 的 `BaseMultiModalProcessor`（约 `:972`）负责 prompt placeholder、token 插入和 processor cache；`:69-120` 还有 tokenizer encode/decode cache。图片 decode/resize 属于 processor/模型 processor 路径，需在服务端单独计时。
2. upstream `vllm/multimodal/encoder_budget.py:44` 的 `MultiModalBudget` 计算 encoder compute budget、cache size 和每 item 最大 token；v1 scheduler 在 `vllm/v1/core/sched/scheduler.py:189-205` 构造它，`schedule()` `:340` 之后同时扣 decoder token budget 和 encoder budget。
3. Ascend `vllm_ascend/worker/encoder_acl_graph.py:256` 的 `EncoderAclGraphManager` 按 token budget 捕获/重放 encoder ACL graph；`:135-177` 更新实际序列长度和 graph 参数。该文件的 TODO 明确指出 encoder graph 仍有 dual-path/`(path, token_budget)` 管理缺口，适合作为待测风险而非既定 bug。

### Scheduler

`vllm/v1/core/sched/scheduler.py:65` 的 `Scheduler` 保存 waiting/running 队列、KV/encoder cache 管理和优先级策略；`:340` 的 `schedule()` 先取 `max_num_scheduled_tokens`，再在 `:378-516` 遍历运行请求、处理 chunked prefill、encoder budget、KV block 对齐和抢占。优先级队列在约 `:164`、抢占选择在 `:473-477`。投机 token 在 `:369` 建立 `scheduled_spec_decode_tokens`，并在后续输出中回传。这里的可疑成本是“长短请求混合时 token/encoder budget 造成的尾延迟或不必要 preemption”，必须用 P99 和队列轨迹验证。

### 投机解码

upstream 的 draft/verification 接口位于 `vllm/v1/spec_decode/`；Ascend 当前 v0.23.0.post1 提供 `vllm_ascend/spec_decode/{dflash_proposer,draft_proposer,eagle_proposer,medusa_proposer,ngram_proposer,suffix_proposer}.py`，并有 V2 的 `worker/v2/spec_decode/eagle/{aclgraph,speculator}.py`。`dflash_proposer.py:28` 按 `max_batch_size * (1 + num_speculative_tokens)` 建立逻辑 query buffer，`:158-251` 处理 ACLGraph mode、metadata 和 full-graph 参数更新。

## Issue / PR 排除结果

| 条目 | 状态/时间 | 判断 |
|---|---|---|
| [#18121 DFlash/DFlash2 query buffer too small](https://github.com/vllm-project/vllm-ascend/issues/18121) | closed 2026-10-10；[PR #17379](https://github.com/vllm-project/vllm-ascend/pull/17379) merged | 已有修复：容量取 `max(logical, max_cudagraph_capture_size)`，不得作为本项目创新题目。 |
| [#16799 SFA MTP uneven-DP crash](https://github.com/vllm-project/vllm-ascend/issues/16799) | closed 2026-09-23 | 已关闭；需查对应合并提交后再决定是否有回归，不能直接复用为新 bug。 |
| [#17139 speculative attention metadata vs graph padding](https://github.com/vllm-project/vllm-ascend/issues/17139) | open，RFC 更新 2026-09-22 | 仍是候选调查项；它指出 FULL 使用 padded token count、PIECEWISE/eager 使用 logical count，多个 Ascend attention backend 仍需分别适配。 |
| [#18223 TASK_QUEUE_ENABLE=2 与 graph capture 冲突](https://github.com/vllm-project/vllm-ascend/issues/18223) | open，2026-10-10 更新 | 文档/兼容性问题，复现条件明确；优先作为环境门禁或小修复候选，不可假定有性能收益。 |
| [#14347 graph mode rtMemcpy 107030 plog](https://github.com/vllm-project/vllm-ascend/issues/14347) | open | 已报告“不影响推理”；除非实机证明同步/错误处理会拉长延迟，否则不作为主选题。 |
| [#12337 Qwen3-30B single-batch regression](https://github.com/vllm-project/vllm-ascend/issues/12337) | open | 模型/配置为 30B、报告版本跨度大，和 Qwen3-8B 单卡基线不直接等价；仅作背景，不当作已复现证据。 |

最新 v0.23.0 release notes 还列出 Qwen3-VL rotary copy race、spec-decode accuracy/OOB、graph 与 hybrid attention 限制等已合入修复/已知限制；因此多模态和投机路径必须以固定版本逐项查 commit，不能引用旧 issue 标题推断当前行为。

## 三个可验证假设（尚未宣称已发现瓶颈）

### H1：ACLGraph bucket padding 在小 batch decode 中抵消图收益

**入口：** `ACLGraphWrapper.__call__`、`model_runner_v1.py:capture_model/execute_model/_model_forward`、upstream scheduler 的 batch descriptor。  
**最小复现：** Qwen3-8B BF16，single NPU，固定 prompt/output（例如 128/128），并发 1/4/8/16/32；同一 capture-size 列表分别运行 eager、FULL/PIECEWISE（每个 case warmup 后重复），记录实际 token、选中的 bucket、capture/replay/host 时间。  
**支持证据：** bucket padding token 占比高，graph replay 的 NPU kernel 时间下降但 TTFT/TPOT 或 E2E P99 上升；捕获尺寸变化能稳定改变该差值。  
**否定证据/止损：** padding 占比低、graph 在所有代表负载均明显优于 eager，或瓶颈主要在 kernel；停止改 graph。  
**预计卡时：** 约 1.5–2.5 卡时（4 个并发 × 2 模式 × 3 重复，另加一次 profiler）。

### H2：V1 scheduler 的 encoder/token budget 在长短混合请求下产生可避免的 P99 尾延迟

**入口：** `vllm/v1/core/sched/scheduler.py:schedule()` 的 `token_budget`、`encoder_compute_budget`、chunked prefill 和抢占分支；多模态时再接 `MultiModalBudget` 和 `EncoderAclGraphManager`。  
**最小复现：** 文本先用 Qwen3-8B 构造 128/4k 输入混合，随后在可运行 Qwen-VL 时加入单图/多图；并发 4/8，固定到达率，采集每请求 enqueue/start/first-token/finish、scheduled token、preemption 次数和 encoder cache hit。  
**支持证据：** 长短混合时 P95/P99 TTFT 或 preemption 随预算边界显著增加，且总 NPU 利用率未饱和；改变 `max_num_batched_tokens` 或 chunk threshold 可改善尾延迟而不降低总吞吐。  
**否定证据/止损：** 尾延迟由模型 kernel/输入预处理主导，或调度参数改善一类负载却普遍损害吞吐；停止调度改动。  
**预计卡时：** 约 2–3 卡时（3 个负载 × 2 参数组 × 5 重复，暂不做 profiler）。

### H3：投机解码 metadata 的 logical/padded token 语义在 PIECEWISE/eager 仍存在 backend-specific shape 风险

**入口：** upstream `vllm/v1/spec_decode` metadata/speculator，Ascend `spec_decode/dflash_proposer.py` 和 attention backend；关联 open RFC [#17139](https://github.com/vllm-project/vllm-ascend/issues/17139)。  
**最小复现：** 只在 H1 基线稳定后，选有现成 draft 权重且官方支持的 DFlash/EAGLE/MTP 配置；构造 uneven DP batch、FULL/PIECEWISE/NONE 三种 graph mode，检查 query、slot mapping、`actual_seq_lengths_q`、draft/target 输出 shape 与 greedy correctness。  
**支持证据：** 某一 mode/backend 出现 shape mismatch、错误输出或只能通过过度 padding 避免，且能缩小到 metadata builder/attention boundary。  
**否定证据/止损：** 目标版本已有完整适配、没有可用 draft 权重、或仅在已关闭 issue 的旧提交复现；停止 speculative 方向。  
**预计卡时：** 约 2–4 卡时；需要额外 draft 权重空间，先确认显存后再租卡。

## 远端实例验收与依赖结果

远端只有一个 NPU（`npu-smi` ID 7，型号 910B2，HBM 3421/65536 MB，健康 OK，检查时无运行进程），`/data` 可用约 91 GB。CANN 的 `/usr/local/Ascend/cann-9.0.0/opp/version.info` 为 9.0.0，驱动版本文件为 26.1.1。该实机组合正好对应官方矩阵的 v0.22.1rc1 行：CANN 9.0.0、torch/torch-npu 2.10.0、Triton Ascend 3.2.1；v0.23.0/post1 要求 CANN 9.1.0 和 torch-npu 2.10.0.post4，因此没有强行安装 v0.23 运行栈。[官方兼容矩阵](https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html#release-compatibility-matrix)

在 `/data/vllm-ascend-env` 中按国内镜像安装了 vLLM、vLLM-Ascend、Triton Ascend 及其非 CUDA 运行依赖，并保留系统提供的 torch/torch-npu。实测 `torch.npu.is_available()` 为 True、设备数为 1，`torch.ones((2,2), device="npu") + 1` 能同步并返回正确结果；vLLM 平台插件识别为 `NPUPlatform`，设备名为 `Ascend910B2`。`python -m vllm.entrypoints.openai.api_server --help` 能正常解析参数。

`pip check` 仍会列出上游 vLLM 的 CUDA-only 包（flashinfer、nvidia-*、tilelang 等）和 Ascend 可选的 `memcache-hybrid`/`memfabric-hybrid`，这些没有安装；也会列出通用 vLLM 元数据要求 torch 2.11、torchvision/torchaudio 2.11/0.26 与 Ascend 矩阵的差异。不能把这些 resolver 文本当成 NPU 运行失败证据。CLI 日志还提示 vLLM 通用 Triton kernel 的 `triton.language.target_info` 不可用；Triton Ascend 扩展本身可导入，平台插件仍能加载。模型启动、Profiler 产物和四方向分析见 [02_four_direction_analysis.md](02_four_direction_analysis.md)。

## 技术缺口与风险

- 已确认 910B2、驱动/CANN、单卡基础 NPU 算子、`torch_npu.profiler`/`msprof` 权限和 Qwen3-0.6B 最小模型启动；模型级 graph 对比仍待执行。
- 官方 v0.23.0 文档的验证表是 A2/A3/950DT 产品族；“910B 64GB”需要供应商给出具体 Atlas 产品和镜像 tag。
- ACLGraph capture 失败可能直接抛错；不要在脚本中自动把失败解释为 eager fallback。
- Qwen3-VL 4B/8B 的 processor、vision encoder 和动态 token 预算只有在模型启动成功后才进入基线；第一轮不下载大权重。
- 当前主分支已出现 Model Runner V1 弃用 RFC；本项目固定 v0.23.0 时应记录 V1/V2 选择，不能把主分支行为回填到旧 tag。

## 阶段判断

远端小模型已经完成启动验证，Profiler 也已生成可读产物。四方向调查、三个优先假设和后续 Profiling 矩阵见 [02_four_direction_analysis.md](02_four_direction_analysis.md)；下一步等待确认后再修改运行时或 scheduler 源码。
