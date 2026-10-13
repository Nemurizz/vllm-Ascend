# 新容器恢复手册

租赁实例被替换后，仓库中的代码、脚本、实验结果和版本记录仍然可恢复；模型权重和 Python/CANN 运行时需要重新安装或下载。恢复时不要把旧实例的绝对路径当成硬编码依赖。

## 1. 克隆和检查提交

```bash
git clone https://github.com/Nemurizz/vllm-Ascend.git
cd vllm-Ascend
git fetch --all --tags
git log --oneline --decorate --all -12
git checkout research/runtime-atb
git status --short
```

仓库当前用独立提交保存三类内容：运行脚本与 workload、阶段报告与证据、可迁移配置与文档。`research/runtime-atb` 保留 #16637 和 #15920 的 patch、固定提交、回归证据及限制说明；后续实验从该分支再创建新分支，保留这些提交作为基线。当前研究提交为 `f65b351`（#16637）和 `21e2283`（#15920）。

## 2. 检查新实例

```bash
cp configs/remote.env.example configs/remote.env
source configs/remote.env
bash scripts/collect_env.sh "${RESULTS_ROOT}/env_$(date +%Y%m%d_%H%M%S).txt"
bash scripts/check_stack.sh
```

必须重新确认 `npu-smi info`、架构、磁盘、Python、CANN、torch-npu、vLLM 和 vLLM-Ascend 版本。NPU device id、容器内 `/data` 和 `$HOME` 路径可能改变，不能据此判断实验不一致。

## 3. 安装软件栈

CANN 和 torch-npu 通常由镜像或宿主机提供，先确认其版本，再按对应的 [vLLM-Ascend 兼容矩阵](https://docs.vllm.ai/projects/ascend/en/latest/community/versioning_policy.html) 安装 vLLM/vLLM-Ascend。`scripts/install_stack.sh` 只使用配置中的版本和国内 pip 源，不升级驱动或 CANN。

旧实验栈的版本记录在 `evidence/source_pins.json`。如果新镜像使用更新版本，先重新生成环境证据并重新跑基线，不要直接把旧延迟数字与新版本比较。

源码修复的独立 checkout 可直接从本仓库中的 Git bundle 恢复：

```bash
python scripts/restore_sources.py --destination "$SYSTEM_ROOT/src/issue_checkouts" --issue 16637
# 后续切换到第二个问题：
python scripts/restore_sources.py --destination "$SYSTEM_ROOT/src/issue_checkouts" --issue 15920
```

脚本下载 `code/source-bundles.json` 中固定的 upstream base（官方 GitHub 优先，失败后用 Gitee 镜像），验证 bundle SHA256 并导入原始 Git 对象。vLLM 的恢复 HEAD 分别是 `f9de447`、`b9f0001`；vLLM-Ascend 分别是 `b7a7f5e`、`dad2ab9`。#15920 的原始源码分支包含前一个 #16637 commit；它自己的独立 diff 仍在 `patches/issue_15920_*.patch`。已有 checkout 不 clean 或分支偏离时脚本会停止，不会覆盖本地工作。详细说明见 [`code/README.md`](../code/README.md)。

这只恢复源码，不能将它直接当作当前 0.22.1 NPU 环境已通过验收的安装栈。仍须依据兼容矩阵建立独立环境；#16637/#15920 的 NPU 复现和性能验收尚未完成。

## 4. 下载模型

```bash
source configs/remote.env
bash scripts/download_model.sh
```

脚本默认下载固定 revision 的 Qwen3-8B 到 `${MODEL_DIR}`。如果镜像无法访问 `HF_ENDPOINT`，设置为空或改为可用的内部镜像；模型 token 通过 Hugging Face CLI 的现有登录状态提供，不写入配置文件。

## 5. 复现实验

```bash
source configs/remote.env
bash scripts/run_baseline.sh
bash scripts/run_graph_metrics.sh
```

输出写入 `${RESULTS_ROOT}/raw` 和 `${LOG_ROOT}`。Profiler 运行应使用新目录，例如 `${PROFILE_ROOT}/qwen3_8b_graph_msprof_${RUN_ID}`；不要把完整 trace 强制提交到 Git。

## 6. 结果归档

每次实验至少保存：命令、配置、环境检查、模型 revision、原始 JSON、服务日志、Profiler 输出目录、正确性摘要和 Git commit。报告中同时写清观察到的事实、推测和尚未在新实例复现的结论。

旧实例在 2026-10-14 又导出了 55 个原始结果/日志和 Runtime Graph/Eager 两套时间线及 CSV。它们在 `evidence/migration_20261014/`，不依赖旧容器继续存在：

```bash
# 校验全部分片（无需解压 2.2 GiB）：
python scripts/restore_evidence.py --verify-only --include-profiler
# 默认只恢复小型结果和日志，不覆盖已有文件：
python scripts/restore_evidence.py --destination "$SYSTEM_ROOT/recovered_20261014"
# 要复核详细 Profiler 时间线时，使用另一个新目录：
python scripts/restore_evidence.py --destination "$SYSTEM_ROOT/profiler_recovered_20261014" --include-profiler
```

分片按 manifest 顺序自动拼接，分别验证分片和完整压缩包 SHA256；解压路径、文件数和大小也会核对。压缩文件约 158 MiB，全部解压约 2.2 GiB，临时拼接另需约 126 MiB；建议系统盘至少留出 3 GiB。可用解压后的 `trace_view.json` 做时间线分析，CSV 保留 kernel、operator 和 API 统计。本次保留的是完整的已导出时间线和 CSV，没有将全部厂商原始二进制采集目录归档。旧包列表在 `dependencies-lock.json` 中用于版本核对，不能直接当作新镜像通用的 pip 安装清单。

## 7. 旧 patch 的使用边界

`patches/offstride_capture_max.patch` 是 vLLM 0.22.1/vLLM-Ascend 0.22.1rc1 的回移验证 patch；上游对应问题已经通过 PR #16970 合并修复。只有为了复现旧栈回归时才应用它。更新版本先检查源码和 Issue 状态，不要盲目套用。

