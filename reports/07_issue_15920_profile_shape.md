# Issue #15920：长上下文 profile shape 与密集 causal mask OOM

调查日期：2026-10-12（Asia/Shanghai）  
状态（2026-10-15 更正）：历史 profile shape 实验及 CPU 策略测试通过；它不是已验证的 #15920 修复。当前 V1 profile 绕过 attention，改变请求数量不能证明覆盖原始 mask OOM。

## 2026-10-15 适用性与结论更正

之前将下述策略改动称为“最小修复”，依据不充分。`GPUModelRunner.profile_run()` 的 `is_profile=True` dummy run 没有强制建立 attention metadata；Ascend attention 在 `attn_metadata is None` 时直接返回填零输出。因此 single-request profile shape 并不自动执行长序列 attention/mask，不能声称解决了 Issue 原始 OOM，也不能声称显存预算变得正确。补丁保留为历史实验，停止部署为修复。

在实际安装的 0.22.1/0.22.1rc1、CANN 9.0.0、单张 910B2 上，Qwen3 generation 路径调用 `AttentionMaskBuilder.get_splitfuse_attn_mask()`。对配置长度 2,048、32,768、98,304，NPU 实测均返回同一份 2,048×2,048 int8 mask（4,194,304 bytes），后两次没有新分配。这排除了此路径中的原始 L² mask 分配机制；这是 helper 实测，不能据此排除其他长上下文 OOM。完整命令与 JSON 将记录在当前环境适用性报告中。

## Issue、PR 和当前主线核查

[Issue #15920](https://github.com/vllm-project/vllm-ascend/issues/15920) 仍为 Open，页面没有关联 PR 或 branch。原始报告使用 vLLM/vLLM-Ascend 0.9.1、CANN 8.1 RC1、Qwen2.5-14B-Instruct-1M、910B2 多卡，并把 `max_model_len` 设到 81,920 或 98,304。

Issue 作者后来在同一页面修正了根因：实际是 V0，不是 V1；vllm-ascend 没有强制 chunked prefill；profile 的 `max_num_reqs=256` 将总 token 数拆成约 256 个短请求，因而没有覆盖单请求长序列产生的 dense causal mask。对单请求长度 L，相关临时张量规模按 `L² × 2 bytes` 增长，98,296 长度约为 18 GiB；把 98,304 拆为 256×384 时，对应 mask 只有约 288 KiB。作者的控制实验显示 `max_num_seqs=1, L=131072` 可以通过，而 `max_num_seqs=256, L=98304` OOM，说明关键变量是 profile shape，而非总 token 数。

我检索了 vLLM/vllm-Ascend 的 profile、`max_num_reqs`、attention mask 相关 Issue/PR，没有找到针对该 shape 缺口的已合并修复。[vLLM Issue #21652](https://github.com/vllm-project/vllm/issues/21652) 讨论了 profile_run 只覆盖 token budget 的相关现象，但已关闭且不是 #15920 的直接修复。

当前固定源码仍在 V1 `GPUModelRunner._dummy_run()` 中以 `scheduler_config.max_num_seqs` 计算 `num_reqs` 和每请求长度；`profile_run()` 只传 `self.max_num_tokens`。vLLM-Ascend V1 runner 复制了相同的 shape 构造，随后通过 `super().profile_run()` 进入该路径。当前主线默认启用 chunked prefill，因此旧 Issue 的 V0 复现不能直接宣称仍存在；但是显式关闭 chunked prefill 时，单请求长序列的 profile shape 仍未被覆盖。

## 历史 shape 策略实验（不能当作 Issue 修复）

本次只改变关闭 chunked prefill 的 profile 形状：

1. vLLM 增加 `_get_profile_max_num_reqs()`，chunked prefill 开启时返回 `None`，保持原有 profile；关闭时返回 `1`，让 profile 覆盖一次消费完整 token budget 的最长单请求。
2. `GPUModelRunner._dummy_run()` 增加可选 `profile_max_num_reqs`，只在 profile 调用处传入该策略。
3. vLLM-Ascend `NPUModelRunner._dummy_run()` 和 310P wrapper 透传同一个参数；V2 wrapper 使用 `*args, **kwargs`，无需另改。
4. 增加参数化单元测试，验证 chunked=True → `None`、chunked=False → `1`。

这个策略不改变默认 chunked prefill 的 profile。关闭 chunked 时，它改变 dummy 请求形状，但 attention 被跳过，尚未证明可覆盖真实 mask 工作区或改善显存预算。

## 固定提交和可迁移文件

- vLLM：`1ff5edb02327583cd9c88f6bda3c485dbd504938` → `b9f0001`，branch `codex/issue-15920-profile-shape`。
- vLLM-Ascend：`c8addbb24fe8fa820bb79685c116e43ba89b77e9` → `dad2ab9`，branch `codex/issue-15920-profile-shape`。
- [`patches/issue_15920_vllm_profile_shape.patch`](../patches/issue_15920_vllm_profile_shape.patch)
- [`patches/issue_15920_vllm_ascend_profile_shape.patch`](../patches/issue_15920_vllm_ascend_profile_shape.patch)

## 验证和限制

两个 checkout 的 Python `compileall` 与 `git diff --check` 均通过。远端 910B2 机器上使用当前源码文件、CPU backend 和现有测试依赖运行了策略 helper 及新增测试：chunked 两个分支均通过。远端安装栈是 vLLM/vllm-Ascend 0.22.1/0.22.1rc1、CPU torch 2.10.0 与 torch_npu 2.10.0，且没有 Qwen2.5-14B-Instruct-1M；为避免把版本错配结果误认为 Issue 复现，没有覆盖 NPU 安装目录，也没有运行长上下文 OOM 或吞吐 A/B。

下一次有兼容的 vLLM/V1、CANN 和长上下文模型后，应在 `enable_chunked_prefill=False` 下对比原始 256-request profile、修复后的 single-request profile 和实际单请求运行，记录显存峰值、profile 时间、KV cache 可用量、TTFT/TPOT，并用 `torch_npu.profiler`/`msprof` 采集 attention mask 和 workspace 的证据。
