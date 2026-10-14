# Issue #16637：动态 speculative decoding 的统一 padding

调查日期：2026-10-11（Asia/Shanghai）  
状态（2026-10-15 更正）：历史 scheduler 单例失败→通过；新 0.26 NPU 对照发现配置层强制 PIECEWISE，原 padding patch 不能单独恢复 FULL。端到端效果正在验证，不能称为已完成的 Ascend 性能修复。

## 2026-10-15 新发现与结论更正

此前只检查 scheduler padding 门控，遗漏了 `VllmConfig._maybe_override_dynamic_sd_cudagraph_mode()` 和 `_maybe_disable_dynamic_sd_for_data_parallel()`。在固定官方 0.26 tag 和当前主线，这两处保护分别将 V1 动态 SD 的 FULL 模式改成 PIECEWISE，以及在 DP>1 时禁用动态表。因而 Issue 声明的 DP2+动态 FULL 配置不能直接等同于干净官方 tag 的实际生效配置；原作者可能还有其他环境差异，需要额外证据。

910B2×1、Qwen3.5-4B MTP×3、官方 0.26 的预跑中，c4/c8/c16 的 decode dispatch 分别有 86/49/77 次，全部为 PIECEWISE；成功 target replay 分别为 2,838/1,617/2,541 次（分段 replay 数不能当成 decode 命中率分母）。当前保留可靠性保护完成四组 A/B，不通过删保护伪造 FULL 加速。

[上游 PR #45953](https://github.com/vllm-project/vllm/pull/45953) 已合并 V2 动态 SD 的多 K Full Graph 支持，并明确 V1 保持 PIECEWISE；[PR #49652](https://github.com/vllm-project/vllm/pull/49652) 仍为 Open，提出 autoregressive drafter capture 修复。#45953 的已合并功能不能当作本项目原创，也不能直接推断 Ascend V2 + Qwen3.5 已兼容。另有[上游性能 Issue #49986](https://github.com/vllm-project/vllm/issues/49986) 讨论相同 PIECEWISE 成本。完整验收见报告 09。

## 先决核查

Issue [#16637](https://github.com/vllm-project/vllm-ascend/issues/16637) 仍是 Open，描述的是 `num_speculative_tokens_per_batch_size` 让 vLLM scheduler 跳过统一 speculative padding，导致 Ascend `FULL_DECODE_ONLY` 从约 333 tok/s 降至 107 tok/s；Issue 给出的旧栈为 vLLM/vllm-Ascend 0.26.0/0.26.0rc1，并附有建议补丁。Issue 页面没有关联 branch 或 PR。针对 `16637`、`dynamic_sd_lookup padding` 的 vllm-Ascend PR 搜索没有发现已合并的等价修复。

当前源码固定点：

- vLLM checkout：`1ff5edb02327583cd9c88f6bda3c485dbd504938`（Gitee `main` 镜像）。
- vLLM-Ascend checkout：`c8addbb24fe8fa820bb79685c116e43ba89b77e9`（Gitee `main` 镜像）。
- vLLM 主线 `vllm/v1/core/sched/scheduler.py` 仍以 `dynamic_sd_lookup is None` 作为 padding 条件；但同一文件稍后又按最终 `len(num_scheduled_tokens)` 选择动态 K，形成前后不一致。
- Ascend 的 `RecomputeScheduler` 和 `DyntraLBScheduler` 各自复制了这段 padding 逻辑，也仍使用同一门控。默认单节点 scheduler 仍走 vLLM scheduler；Ascend 特殊 P/D 和 Dyntra 路径需要同步修复。

这确认问题没有在当前主线自然消失；也没有把已解决的 PA workspace、off-stride capture 等旧问题重新包装成新贡献。

## 调用链和根因

每个调度 step 先建立 `num_scheduled_tokens`，然后由 `SchedulerOutput.num_spec_tokens_to_schedule` 按动态 batch-size 查表得到 K。对“新进入 decode 的请求”，padding 判断发生在 waiting request loop 内，此时必须提前决定 query 行数。原代码在配置了动态查表时直接跳过 padding，因此首个请求仍只有 1 个 query token，而同一步的其它 speculative 请求有 `1+K` 行；Ascend FULL graph 需要每个 request 的 query 行数一致，于是回退到慢路径。

本次修改在每个 step 开始时统计 `self.running` 中已经追平 `num_tokens` 的 decode 请求，使用 1-indexed `dynamic_sd_lookup` 得到本步 effective K。padding、Mamba alignment 的长度检查、placeholder draft 数量都使用同一个 effective K；K=0 时自然保持 1-token decode。静态 speculative decoding 仍使用原来的最大 K。

## 修改和提交

- vLLM：`codex/issue-16637-dynamic-padding`，提交 `f9de447`。
  - 为 scheduler 保存每步 effective K。
  - 将动态配置从“禁用 padding”改成“按当前 decode batch 查表后 padding”。
  - 新增 `test_spec_decode_padding_first_decode_step_with_dynamic_schedule`，覆盖一个已有 decode 请求带动新请求进入 decode 的场景。
- vLLM-Ascend：`codex/issue-16637-dynamic-padding`，提交 `b7a7f5e`。
  - 在 Recompute/Dyntra 两个复制的调度路径采用相同 effective K。
  - placeholder 数量也改为 effective K，避免 query 行数和 sampler draft 行数分离。

可迁移 patch：

- [`patches/issue_16637_vllm_scheduler.patch`](../patches/issue_16637_vllm_scheduler.patch)
- [`patches/issue_16637_vllm_ascend_scheduler.patch`](../patches/issue_16637_vllm_ascend_scheduler.patch)

## 验证

静态检查：三处 Python 文件 `compileall` 通过，两个 checkout 的 `git diff --check` 通过，提交后均 clean。

在远端缓存目录中用当前 vLLM 主线源码、CPU scheduler 配置执行回归测试。为避免 0.22.1 Ascend 插件与 CPU torch 冲突，测试设置了 `VLLM_TARGET_DEVICE=cpu`、`VLLM_PLUGINS=''`，没有触碰 NPU 安装目录：

| 对照 | 结果 |
|---|---|
| 原始 scheduler + 新增动态 padding 测试 | 失败，`out.num_scheduled_tokens[r2]` 不是 `1+K`，退出码 1 |
| 修复后动态 padding 测试 | 通过 |
| 修复后原有静态 padding 测试 | 通过 |
| 修复后同步 KV resumed padding 测试 | 通过 |

远端环境确实有一张 910B2 64GB（`npu-smi` 为空闲），但安装版本是 vLLM/vllm-Ascend 0.22.1/0.22.1rc1，缺少 Issue 的 0.26+ Qwen3.5 hybrid MTP 负载。当前不能把该旧栈的任何吞吐数字当作 #16637 的验证，也没有在旧栈上覆盖安装主线代码。

## 下一步边界

这个补丁已经证明了调度器层面的“动态 K 导致 padding 缺失”可以稳定复现并修复；尚未证明 910B2 上的最终收益。下一步只有在实例切换到与 Issue 相容的 vLLM/vllm-Ascend、CANN 和 Qwen3.5/MTP 负载后，才运行同一配置的 eager、静态 MTP、动态无补丁、动态补丁四组对照，并记录 TTFT、TPOT、吞吐、FULL/PIECEWISE 命中率和 `torch_npu.profiler`/`msprof` 结果。若兼容栈无法准备，应停止 #16637 的性能宣称并转 #15920，而不是用 0.22.1 结果替代。
