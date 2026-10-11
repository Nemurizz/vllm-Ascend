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

源码修复的独立 checkout 需要分别恢复：vLLM checkout 使用 `codex/issue-16637-dynamic-padding` / `f9de447` 或 `codex/issue-15920-profile-shape` / `b9f0001`；vLLM-Ascend checkout 使用对应的 `b7a7f5e` 或 `dad2ab9`。若只恢复本仓库，直接应用 `patches/issue_16637_*.patch` 或 `patches/issue_15920_*.patch`，不要把 vLLM patch 应用到 vLLM-Ascend checkout。

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

## 7. 旧 patch 的使用边界

`patches/offstride_capture_max.patch` 是 vLLM 0.22.1/vLLM-Ascend 0.22.1rc1 的回移验证 patch；上游对应问题已经通过 PR #16970 合并修复。只有为了复现旧栈回归时才应用它。更新版本先检查源码和 Issue 状态，不要盲目套用。

