# vLLM-Ascend 910B 推理优化调查仓库

这个仓库保存 910B 单卡 vLLM-Ascend 项目的源码调查、可复现实验脚本、基线结果、Profiler 摘要和迁移配置。当前内容对应第一阶段和 ACLGraph/Runtime 后续调查；还没有把任何未经验证的上游修改当作最终优化方案。

## 当前固定证据

- 2026-10-15 分区执行：新增 [GDN 报告 11](reports/11_gdn_length_specialization.md) 与 [prepare 恢复说明](docs/gdn_prepare_reproduction.md)。复现未合并的 #17604 并完成长度/tiling/graph 组件筛选；长输入 eager 的 BLOCK_T=4 相对 PR 默认局部耗时低约 6%–9%，尚无这项组合的整模型收益。另查明最新主线已合并 #18723、改变 QKV 输入布局，不能将 0.26 的节省宣称为最新主线仍有相同问题。

- 2026-10-15 算子阶段：新增 [GDN 报告 10](reports/10_gdn_operator_optimization.md) 与 [复现/迁移说明](docs/gdn_operator_reproduction.md)。已独立复现社区 WY 索引修复和 GDN 布局优化；8K prefill 在匹配 0.26 栈约有 3.2%–4.0% 的吞吐收益，短输入/长 decode 未获稳定收益。默认异步 graph 的生成波动另做 stock 控制保留，不宣传为已修复。两个源码 commit、补丁与 Git bundle 分开归档。

- 2026-10-15：原 0.22.1 环境没有 #16637 的动态配置特性；当前 Qwen3 的 4 MiB mask 没有 #15920 的原始平方增长路径。另建 0.26/CANN9.1 环境完成四组 A/B、交错验证和 32K/98K profile；两个历史候选都不作为已验证修复。见 [`报告 09`](reports/09_current_environment_and_026_ab.md) 和 [`0.26 恢复说明`](docs/stack026_reproduction.md)。

- 远端验证栈：Python 3.11.15、CANN 9.0.0、torch 2.10.0+cpu、torch-npu 2.10.0、vLLM 0.22.1、vLLM-Ascend 0.22.1rc1、Triton Ascend 3.2.1。
- 模型：Qwen3-8B，revision `b968826d9c46dd6066d109eabc6255188de91218`。权重不进入 Git，需要在新实例重新下载。
- ACLGraph off-stride 现象在旧栈上已复现，但对应上游 Issue #16049 已由 PR #16970 合并修复；仓库中的 patch 仅用于旧版本回归验证。
- 5 请求 graph bucket A/B 观察到 `5 -> 8` padding，但增加精确 bucket 后 E2E P50 只改善约 0.63%，低于项目门槛。
- 既有 msprof 轨迹中 `aclrtSynchronizeStreamWithTimeout` 累计约 544 ms，下一步应先做 marker 控制实验，再决定是否碰 replay 同步屏障。
- PA workspace cache 已确认不是社区空白：对应上游 [PR #15579](https://github.com/vllm-project/vllm-ascend/pull/15579) 已合并；本仓库的 0.22.1rc1 patch 只保留为 910B2 回移和独立复现实验证据。
- Issue #16637 的候选 padding 改动保存在 vLLM `f9de447` 和 vLLM-Ascend `b7a7f5e`。2026-10-15 发现官方 V1 动态 SD 配置强制 PIECEWISE，候选 scheduler patch 不能单独恢复 FULL；CPU 测试仅证明所写 padding 期望，不能称为完成端到端 Bug 修复。
- Issue #15920 的历史 profile shape 策略实验分别保存在 vLLM `b9f0001` 和 vLLM-Ascend `dad2ab9`。2026-10-15 复核发现 profile 跳过 attention，不能将这两个提交称为原始 OOM 的已验证修复；详见报告 07 的更正。

## 目录

- `reports/`：阶段报告和止损结论。
- `scripts/`：环境检查、模型下载、benchmark、Graph 和 scheduler 探针。
- `workloads/`：固定请求 workload。
- `configs/`：新实例的路径和版本模板，不包含凭据。
- `evidence/`：小型原始结果、Profiler API 摘要、关键数据库和源码快照。
- `patches/`：旧 v0.22.1 配置回移 patch 和源码摘录。
- `docs/`：迁移、恢复和实验运行说明。
- `code/`：两个 upstream checkout 的增量 Git bundle、完整 commit SHA 和校验值。

## 新实例恢复

1. 克隆仓库并切换到目标提交：

   ```bash
   git clone https://github.com/Nemurizz/vllm-Ascend.git
   cd vllm-Ascend
   git fetch --all --tags
   git checkout research/runtime-atb
   ```

2. 复制配置模板：

   ```bash
   cp configs/remote.env.example configs/remote.env
   ${EDITOR:-vi} configs/remote.env
   source configs/remote.env
   ```

3. 在新容器中准备 CANN、torch-npu 和 Python 环境，然后执行：

   ```bash
   bash scripts/check_stack.sh
   bash scripts/download_model.sh
   ```

4. 运行基线或 Graph 实验前，先检查 `MODEL_DIR`、`PROJECT_ROOT`、`RESULTS_ROOT` 和 `VLLM_ENV_ROOT`。模型权重、pip 缓存、Profiler 大文件都放在配置指定的数据盘/系统盘目录，不写入 Git。

完整恢复步骤见 [`docs/recovery.md`](docs/recovery.md)。

源码提交可以用 `python scripts/restore_sources.py --destination "$HOME/vllm_project/src/issue_checkouts" --issue 16637` 恢复。2026-10-14 又归档了 55 个原始 benchmark/日志文件，以及 Runtime Graph/Eager 两套完整的 `trace_view.json` 和 CSV 导出，见 `evidence/migration_20261014/`。用 `python scripts/restore_evidence.py --destination "$HOME/vllm_project/recovered_20261014"` 恢复小型记录；加 `--include-profiler` 恢复时间线。压缩归档约 158 MiB，全部解压约 2.2 GiB。

研究分支中的提交顺序保留了可审计的独立变更：`f65b351` 对应 #16637 报告，`21e2283` 对应 #15920 报告。两个源码 checkout 的实现分支和固定提交记录在 `evidence/issue_16637_regression.json`、`evidence/issue_15920_regression.json` 以及 `evidence/source_pins.json`；vLLM 主仓库的改动通过 `patches/` 迁移，不把两个上游仓库混成一个 Git 历史。

## 安全边界

仓库不保存 GitHub token、SSH 密码、私钥、远端登录地址或带凭据的 manifest。远端实验结果中的绝对路径是历史证据；新实例使用 `configs/remote.env` 覆盖这些路径。本次关键 Profiler 时间线已经压缩归档并完成凭据扫描；后续大文件可放在外部对象存储或按需重新采集。

