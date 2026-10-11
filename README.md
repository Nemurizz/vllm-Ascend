# vLLM-Ascend 910B 推理优化调查仓库

这个仓库保存 910B 单卡 vLLM-Ascend 项目的源码调查、可复现实验脚本、基线结果、Profiler 摘要和迁移配置。当前内容对应第一阶段和 ACLGraph/Runtime 后续调查；还没有把任何未经验证的上游修改当作最终优化方案。

## 当前固定证据

- 远端验证栈：Python 3.11.15、CANN 9.0.0、torch 2.10.0+cpu、torch-npu 2.10.0、vLLM 0.22.1、vLLM-Ascend 0.22.1rc1、Triton Ascend 3.2.1。
- 模型：Qwen3-8B，revision `b968826d9c46dd6066d109eabc6255188de91218`。权重不进入 Git，需要在新实例重新下载。
- ACLGraph off-stride 现象在旧栈上已复现，但对应上游 Issue #16049 已由 PR #16970 合并修复；仓库中的 patch 仅用于旧版本回归验证。
- 5 请求 graph bucket A/B 观察到 `5 -> 8` padding，但增加精确 bucket 后 E2E P50 只改善约 0.63%，低于项目门槛。
- 既有 msprof 轨迹中 `aclrtSynchronizeStreamWithTimeout` 累计约 544 ms，下一步应先做 marker 控制实验，再决定是否碰 replay 同步屏障。
- PA workspace cache 已确认不是社区空白：对应上游 [PR #15579](https://github.com/vllm-project/vllm-ascend/pull/15579) 已合并；本仓库的 0.22.1rc1 patch 只保留为 910B2 回移和独立复现实验证据。

## 目录

- `reports/`：阶段报告和止损结论。
- `scripts/`：环境检查、模型下载、benchmark、Graph 和 scheduler 探针。
- `workloads/`：固定请求 workload。
- `configs/`：新实例的路径和版本模板，不包含凭据。
- `evidence/`：小型原始结果、Profiler API 摘要、关键数据库和源码快照。
- `patches/`：旧 v0.22.1 配置回移 patch 和源码摘录。
- `docs/`：迁移、恢复和实验运行说明。

## 新实例恢复

1. 克隆仓库并切换到目标提交：

   ```bash
   git clone https://github.com/Nemurizz/vllm-Ascend.git
   cd vllm-Ascend
   git checkout main
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

## 安全边界

仓库不保存 GitHub token、SSH 密码、私钥、远端 IP/端口或带凭据的 manifest。远端实验结果中的绝对路径是历史证据；新实例使用 `configs/remote.env` 覆盖这些路径。Profiler 数据库只用于本阶段复核，后续大文件应放在外部对象存储或按需重新采集。

