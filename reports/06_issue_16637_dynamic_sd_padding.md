# Issue #16637：动态 speculative decoding 的统一 padding

调查日期：2026-10-11（Asia/Shanghai）  
状态：源码修复和回归测试完成；910B2 NPU 吞吐 A/B 暂缓，等待与 Issue 相同的版本和模型负载。

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
