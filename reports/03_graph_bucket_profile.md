# Graph bucket / padding profiling

状态：已完成配置逻辑复现、910B2 Qwen3-8B A/B 和旧版本回移 patch 验证；该缺陷已被上游 PR #16970 修复，不再作为新的上游选题。

## 1. 外部线索与版本核对

社区 Issue [vllm-ascend #16049](https://github.com/vllm-project/vllm-ascend/issues/16049) 描述了同一条路径：Ascend 将默认 `max_cudagraph_capture_size` 设为 `max_num_seqs * decode_query_len`，而 vLLM 的默认 ladder 仍以 `[1,2,4]` 和 8 的倍数生成；`max_num_seqs=12/20` 时最大实际桶分别停在 8/16，较大的 decode batch 没有图可 replay。Issue 给出的 Qwen3-30B 测量是 7.5×/6.7×，并建议把最大配置值追加到 capture list。该 Issue 已由 [PR #16970](https://github.com/vllm-project/vllm-ascend/pull/16970) 合并修复；因此本次工作验证的是旧版 `0.22.1rc1` 的回移效果。

上游设计文档说明：图只为有限的 capture size 捕获，运行时 batch 会映射到支持的桶；超过最大桶时跳过 graph 并走 eager。[vLLM-Ascend ACL Graph design](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/developer_guide/Design_Documents/ACL_Graph.md) 还强调 Ascend 平台可能进一步过滤 capture size。当前固定环境的官方兼容栈仍是 v0.22.1/v0.22.1rc1 + CANN 9.0.0，和最新 Issue 的版本不同，因此必须在本机复现，不能直接移植结论。

## 2. 固定版本源码证据

远端源码摘录保存于 `/root/vllm_project/evidence/aclgraph_bucket_source_excerpt.txt` 和 `/root/vllm_project/evidence/aclgraph_platform_helper_lines.txt`。

- `vllm_ascend/platform.py` 的 `_get_default_max_cudagraph_capture_size()` 明确去掉上游的尾部 `* 2`，返回 `min(max_num_seqs * decode_query_len, 512)`。
- `vllm/config/vllm.py:_set_cudagraph_sizes()` 随后用 `[1,2,4]` 和步长 8/16 的 ladder 生成列表；它只额外保证 `max_num_batched_tokens` 被捕获，没有保证 `max_cudagraph_capture_size` 自身被追加。
- 固定环境的纯配置输出：

```text
max_num_seqs=8   default=8   ladder=[1,2,4,8]
max_num_seqs=12  default=12  ladder=[1,2,4,8]
max_num_seqs=16  default=16  ladder=[1,2,4,8,16]
max_num_seqs=20  default=20  ladder=[1,2,4,8,16]
max_num_seqs=24  default=24  ladder=[1,2,4,8,16,24]
max_num_seqs=32  default=32  ladder=[1,2,4,8,16,24,32]
```

这一步不依赖 NPU，说明问题首先是可测试的配置逻辑缺口，而不是偶发硬件行为。复现脚本为 [cudagraph_offstride_repro.py](../scripts/cudagraph_offstride_repro.py)，远端输出保存在 `/root/vllm_project/results/raw/cudagraph_offstride_repro.json`。

## 3. 910B2 A/B 复现

两组服务使用同一 Qwen3-8B revision、同一模型长度、同一请求脚本和采样参数，只改变 capture 配置。客户端同时提交 12 个短请求，输出 32 tokens。

| 配置 | 实际 capture sizes | ACLGraph replay 日志 | E2E P50 | 请求/s |
|---|---|---:|---:|---:|
| 默认，`max_num_seqs=12` | `[1,2,4,8]`，启动日志出现截断到 8 | 0 次 | 2497 ms | 4.79 |
| 显式加入 12 | `[1,2,4,8,12]` | 1 次 | 760 ms | 15.59 |

原始结果：

- `/root/vllm_project/results/raw/qwen3_8b_offstride_c12.json`
- `/root/vllm_project/results/raw/qwen3_8b_offstride_fixed_c12.json`
- `/root/vllm_project/logs/qwen3_8b_offstride_server.log`
- `/root/vllm_project/logs/qwen3_8b_offstride_fixed_server.log`

显式补充最大桶后，P50 降低约 69.5%，吞吐提升约 3.25×；两组均 12/12 请求成功。这个结果已经超过项目的 3% 门槛。

## 4. Profiling 计划与最小修复候选

已用 `msprof` 包住显式 `[1,2,4,8,12]` 服务的一次 12 路请求，轨迹目录为：

`/root/vllm_project/profiles/qwen3_8b_offstride_fixed_msprof/PROF_000001_20261010162913736_00030192RMQEDOPL/`

下一步先从轨迹中核对 decode iteration 的 Runtime API、AI Core task 数量和 host launch gap。最小 patch 已在远端 pinned vLLM 安装副本中验证：在默认 ladder 生成后追加 `max_cudagraph_capture_size`（满足 `>=1` 且不重复），最后排序；保留用户显式 `cudagraph_capture_sizes` 的语义，不改变 Ascend 的 `*2` 内存策略。候选改动只涉及配置列表生成，不涉及算子、驱动或调度器。

远端验证副本的备份和补丁分别为 `/root/vllm_project/patches/vllm_config_vllm.py.before_offstride_fix` 和 `/root/vllm_project/patches/offstride_capture_max.patch`；实际安装文件是 `/data/vllm-ascend-env/lib/python3.11/site-packages/vllm/config/vllm.py`。这不是上游 Git checkout 的提交。

补丁后的默认启动日志自动得到 `[1,2,4,8,12]`，不再出现截断到 8 的警告，并记录 1 次 `Replaying aclgraph`。同一 12 路 workload 的补丁结果为 E2E P50 `762.907 ms`、`15.552 req/s`，与显式配置 `[1,2,4,8,12]` 的 `760.465 ms`、`15.590 req/s` 一致。`py_compile` 返回 0；固定 prompt、`temperature=0`、流式输出的旧版与补丁版文本 SHA256 完全相同，均为 63 个字符、12/12 成功。测试结束后 NPU 无残留进程。

停止条件：

1. 追加最大桶后捕获失败、HBM 超限或启动时间显著恶化；
2. 固定 12 路 workload 的 token 级输出不一致；
3. 多个并发/输出长度下收益低于 3%；
4. 上游已有等价修复合入，改为做回归验证而不是重复提交。

当前已完成 profiling 和旧版回移 patch 的单点验证。由于上游已有等价修复，本方向排除为新的上游选题；后续只保留版本回归和升级兼容性检查。
