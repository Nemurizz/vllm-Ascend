# 旧实例销毁后：在新的空实例继续工作

交接日期：2026-10-15。继续点是 **GDN 布局优化已验证，融合 prepare/tiling 完成组件筛选，组合模型 A/B 尚未开始**。本手册按当前进度恢复，不默认部署已经被否定的历史 Issue 补丁。

## 1. 哪些东西在哪里

| 内容 | 持久化位置 | 旧实例是否还需要存在 |
| --- | --- | --- |
| 报告、技术复盘、脚本、配置、主要结果、选定 CSV/timeline | GitHub `Nemurizz/vllm-Ascend` 的 `research/gdn-prepare-shapes` 分支 | 不需要 |
| WY 修复和布局优化的真实源码提交 | Git 中的 `code/gdn-operator-026.bundle`、两份 patch 和 manifest | 不需要；先取得固定 upstream base，再导入 bundle |
| 额外原始 PROF/FRAMEWORK 采集及少量补充记录 | 本地 `D:\AI_Inference_Project\vllm_ascend_preservation_20261015` | 下载验收后不需要 |
| CANN、ATB、Python、wheel 和依赖 | 不备份安装目录；按下表重新下载/安装 | 不需要 |
| 模型权重 | 不进入 Git 或本次本地备份；按固定 manifest 重新下载到新数据盘 | 不需要 |
| graph、Triton 编译缓存、venv、恢复验收目录 | 不迁移；重新生成 | 不需要 |

**正常续做只需要 GitHub、匹配的 NPU 环境和模型。** 本地补充包主要用于重新解析历史原始采集；不必先上传或解压它，才能启动新实验。不能只 clone 默认分支而遗漏研究分支。

## 2. 新实例的条件

优先选择单张 Ascend 910B2 64GB、aarch64、可 SSH 的实例。宿主机驱动和设备透传由平台提供；此前驱动为 26.1.1，新实例需检查兼容性，不能靠复制旧卡号或在容器内盲目升级驱动解决问题。

建议系统盘留出至少 20 GB 空间供 SDK、依赖缓存、源码、结果及可选历史数据；数据盘至少 15 GB 用于本轮约 9.32 GB 的 Qwen3.5-4B 权重。后续 profile 可能继续增长，应检查实际空闲空间。若还要重下 Qwen3-8B，另预留其权重空间。

```bash
npu-smi info
uname -m
df -h / /data
python3 --version
```

物理卡编号可能改变。只分配一张卡时先使用默认逻辑 `npu:0`，不要复制旧实例的物理编号到 `ASCEND_RT_VISIBLE_DEVICES`。

## 3. 首先克隆正确的分支

```bash
mkdir -p /root/vllm_project/{src,scripts,results,logs,profiles}
mkdir -p /data/vllm_project/models
git clone --branch research/gdn-prepare-shapes \
  https://github.com/Nemurizz/vllm-Ascend.git /root/vllm_project/research
cd /root/vllm_project/research
git rev-parse HEAD
git status --short
cp scripts/*.py scripts/*.sh /root/vllm_project/scripts/
```

本次补充前的研究检查点为 `0354d2b430b91ac1a346adcf7dd113d43576afb7`；本手册和备份审计在同一分支的后续提交。实验源码版本由下表固定，不跟随 main 漂移。GitHub 下载受限时可配置可信镜像，官方仓库仍是来源依据；不要把镜像上不同 SHA 的同名 tag 当作同一版本。

## 4. 下载什么、安装哪个版本

### 必需的核心栈

| 项目 | 固定版本/提交 | 下载或安装来源 |
| --- | --- | --- |
| Python | 3.11.15（恢复现有 cp311/aarch64 栈） | 平台镜像或 Python 发行源 |
| CANN Toolkit | 9.1.0，aarch64 | 华为官方 OBS，具体 URL 和 SHA256 见下方链接 |
| 910B ops 包 | 9.1.0，aarch64 | 同上，必须是 910B 包 |
| NNAL / ATB | 9.1.0，aarch64 | 同上 |
| torch | 2.10.0+cpu | 与 torch-npu 配套的 Ascend 镜像/CPU wheel，避免 CUDA wheel |
| torch-npu | 2.10.0.post4 | 华为 Ascend pip 源 |
| triton-ascend | 3.2.2 | 华为 Ascend pip 源 |
| vLLM | v0.26.0，`568afb3a13806beb53bb2e6bd518269357b237c0` | 固定源码，用 empty backend 安装 |
| vLLM-Ascend | v0.26.0rc1，`f2f74a16c3c50a76f4349d807918e83edec1e35c` | 对应 cp311/aarch64 官方 wheel；源码另保留审阅 |
| Transformers | 5.14.1 | 清华 pip 镜像 |
| NumPy | 1.26.4 | 清华 pip 镜像 |
| torchvision / torchaudio | 0.25.0 / 2.10.0 | 与 torch 2.10 配套；不要让 resolver 升级 torch |
| 辅助工具 | pytest、Ruff、requests、Matplotlib | 回归/下载/绘图；版本以归档环境清单作参考 |
| 本地补充包校验 | zstandard==0.23.0 | 只在核验/恢复 `.zst` 补充包时需要 |

官方国内源：`https://mirrors.huaweicloud.com/ascend/repos/pypi`；通用 Python 包：`https://pypi.tuna.tsinghua.edu.cn/simple`。三个 CANN 安装器的完整国内 URL 见 [stack026_reproduction.md](stack026_reproduction.md)，历史下载 SHA256 见 [cann-sha256.txt](../evidence/stack026/cann-sha256.txt)。不要下载模型或 SDK 到 Git 仓库。

### 优先用匹配镜像，再核验实际版本

官方为 A2 提供 `quay.io/ascend/vllm-ascend:v0.26.0rc1` 和 `:v0.26.0rc1-openeuler`。国内镜像前缀 `m.daocloud.io/quay.io/ascend/vllm-ascend:v0.26.0rc1` 也出现在官方文档。镜像与部署方式见 [该版本安装文档](https://docs.vllm.ai/projects/ascend/en/v0.26.0rc1/installation.html) 和 [官方性能文档](https://docs.vllm.ai/projects/ascend/en/v0.26.0rc1/developer_guide/performance_and_debug/performance_benchmark.html)。这些是环境来源，不代表镜像的所有包、Python 小版本和新驱动已在本项目的新实例上验收。

平台若能直接选择镜像，优先在创建实例时选择；已经处在托管容器内时不要求再运行嵌套 Docker。若是空容器/裸环境，按 CANN 安装说明建立独立 SDK 目录，先安装 Toolkit、910B ops、NNAL/ATB，再准备 Python 和配套 native wheel。

**安装限制必须保留：** 原实验 venv 复用了旧环境的通用依赖，未验收过完全从零的一键安装。vLLM 与 Ascend 的 FastAPI metadata 存在约束冲突，本项目离线引擎使用 FastAPI 0.136.3；pip check 并非全绿。历史 [pip-freeze.txt](../evidence/stack026/pip-freeze.txt) 包含旧实例的 `file://` 和 SDK 包，不能整体执行 `pip install -r`。应以核心版本、官方镜像及逐项验收为准，不把前端/多模态/全部可选特性视为已兼容。

### 固定源码与独立 venv

```bash
git clone --depth 1 --branch v0.26.0 https://gitee.com/mirrors/vllm.git \
  /root/vllm_project/src/stack026/vllm
git clone --depth 1 --branch v0.26.0rc1 https://gitee.com/mirrors/vllm-ascend.git \
  /root/vllm_project/src/stack026/vllm-ascend
git -C /root/vllm_project/src/stack026/vllm rev-parse HEAD
git -C /root/vllm_project/src/stack026/vllm-ascend rev-parse HEAD
```

输出必须分别等于表中的两个 SHA。不匹配就重新从官方固定提交获取，不进入性能测试。

若基础镜像已经具备表中核心依赖，可用 `python3.11 -m venv --system-site-packages /root/vllm_project/envs/vllm026` 复用镜像基础包，在 venv 内安装本地 Ascend wheel，再执行：

```bash
source /root/vllm_project/envs/vllm026/bin/activate
# 只在基础依赖已匹配后执行；不让 pip 自动升级到 CUDA/其他 torch 栈。
python -m pip install --no-deps --ignore-installed vllm-ascend==0.26.0rc1 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
VLLM_TARGET_DEVICE=empty python -m pip install --no-build-isolation --no-deps \
  -e /root/vllm_project/src/stack026/vllm
```

构建依赖参考：setuptools>=77,<81、setuptools-scm>=8、setuptools-rust>=1.9，以及镜像内的编译工具。上面是受控安装步骤，不是缺失所有依赖时可单独运行的完整安装器。

## 5. 激活环境，避免导入旧包

如果按原目录布局安装了 `/opt/vllm-project-sdk/cann` 和 `/opt/vllm-project-sdk/nnal/atb`，直接：

```bash
source /root/vllm_project/scripts/activate_stack026.sh
```

如果官方镜像的 SDK 在 `/usr/local/Ascend/...`，先定位真实的 CANN/ATB `set_env.sh`，在新的 shell 中 source 它们，再设置实验路径；不要凭旧实例路径 source 不存在的文件：

```bash
# 将两条路径替换为新镜像中已检查存在的实际文件。
source /actual/cann/set_env.sh
source /actual/atb/set_env.sh
unset ASCEND_RT_VISIBLE_DEVICES VLLM_TARGET_DEVICE
export PATH=/root/vllm_project/envs/vllm026/bin:$PATH
export PYTHONPATH=/root/vllm_project/src/stack026/vllm:/root/vllm_project/scripts:${PYTHONPATH:-}
export OMP_NUM_THREADS=1 TASK_QUEUE_ENABLE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

用实际导入路径验收，而不只看 `pip list`：

```bash
python - <<'PY'
import importlib.metadata as m
import torch, torch_npu, vllm, vllm_ascend
print('paths:', vllm.__file__, vllm_ascend.__file__)
for n in ['torch','torch-npu','vllm','vllm-ascend','triton-ascend','transformers','numpy']:
    print(n, m.version(n))
print('NPU:', torch.npu.is_available(), torch.npu.device_count())
x = torch.ones((16,16), device='npu', dtype=torch.bfloat16)
print('matmul:', float((x@x)[0,0].cpu()))
PY
```

vLLM 路径应落在固定 `src/stack026/vllm`；Ascend 应是未修改的匹配 wheel。不要把旧 `.pth` 中指向已销毁 venv 的路径复制到新实例。

## 6. 只重新下载当前需要的模型

继续 GDN 用 `Qwen/Qwen3.5-4B` BF16。不要因为旧第一基线是 Qwen3-8B，就先多下载一份模型。逐文件 revision、大小及 SHA256 已保存：

```bash
cd /root/vllm_project/research
python scripts/download_modelscope_pinned.py \
  --manifest evidence/stack026/qwen35-modelscope-files.json \
  --output /data/vllm_project/models/Qwen3.5-4B
```

脚本使用国内 ModelScope，所有文件应显示校验通过。manifest 固定的是每个文件的 revision，不伪装成统一模型 commit。下载阶段需要网络；激活脚本的离线变量用于随后推理。只有继续旧 Qwen3-8B 基线时，才另下载 revision `b968826d9c46dd6066d109eabc6255188de91218`。

## 7. 恢复源码与历史数据，先检查正确性

### 布局源码单独恢复，不覆盖已安装基线

```bash
cd /root/vllm_project/research
git -C /root/vllm_project/src/stack026/vllm-ascend bundle verify \
  "$PWD/code/gdn-operator-026.bundle"
git -C /root/vllm_project/src/stack026/vllm-ascend fetch \
  "$PWD/code/gdn-operator-026.bundle" \
  codex/gdn-operator-910b:codex/gdn-operator-910b
git -C /root/vllm_project/src/stack026/vllm-ascend worktree add \
  /root/vllm_project/src/gdn-operator-026 codex/gdn-operator-910b
git -C /root/vllm_project/src/gdn-operator-026 rev-parse HEAD
# 应为 140d8dfe4a59930f3f41d05574e198419e0f4807
```

原始安装栈仍用官方基线，实验 harness 独立加载变体。不要先安装 head-first 补丁版本，再让 harness 将相同补丁生成第二次。完整说明见 [布局复现](gdn_operator_reproduction.md)。

```bash
mkdir -p /root/vllm_project/results/resume-check
SOURCE=$(python -c 'from vllm_ascend.ops.triton.fla import wy_fast; print(wy_fast.__file__)')
python /root/vllm_project/scripts/test_wy_operator.py --source "$SOURCE" \
  --output /root/vllm_project/results/resume-check/wy.json
python /root/vllm_project/scripts/check_gdn_return_contract.py \
  --output /root/vllm_project/results/resume-check/return-contract.json
python /root/vllm_project/scripts/benchmark_gdn_operator.py \
  --output /root/vllm_project/results/resume-check/pipeline.json
```

检查 CPU 参考、输出和 final_state、返回契约，不能只看进程返回 0。历史 stock 等长失败是预期的对照，fixed/head-first 应通过。随后再重跑 [prepare 实验](gdn_prepare_reproduction.md) 和 [模型 A/B](gdn_operator_reproduction.md)。每次只运行一个 NPU benchmark，重新预热；旧延迟不直接与新硬件/镜像作 A/B。

### GitHub 的历史导出按需恢复

```bash
python scripts/restore_evidence.py --archive-root evidence/migration_gdn_20261015 \
  --include-profiler --destination /root/vllm_project/history-gdn
python scripts/restore_evidence.py --archive-root evidence/migration_gdn_dispatch_20261015 \
  --include-profiler --destination /root/vllm_project/history-prepare
```

更早 Graph/Runtime 和 0.26 Issue 证据分别在 `migration_20261014`、`migration_20261015`；同一恢复工具可用。目标必须是新目录，工具不覆盖已有文件。只有阅读报告时无需解压全部历史 trace。

## 8. 本地补充包如何使用

本地目录包含 `manifest.json`、`additional-records.tar.gz`、`native-profiler-source.tar.zst`、校验脚本、校验报告及本手册副本。完整文件列表和每个文件的 SHA256 在 manifest 中。

保留的是 GitHub 未覆盖的 PROF/FRAMEWORK 源数据及 metadata；可重建的派生数据库/重复导出已列为 omitted。GitHub 已有的历史导出仍可直接阅读。没有把原始采集目录的存在当作未完成实验的新证据。

在 Windows 本地只校验，不需要解压深目录：

```powershell
python -m pip install zstandard==0.23.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
python D:\AI_Inference_Project\vllm_ascend_preservation_20261015\verify_instance_supplement.py `
  --manifest D:\AI_Inference_Project\vllm_ascend_preservation_20261015\manifest.json
```

将该目录通过新实例自己的 SSH 地址上传到系统盘，例如 `/root/vllm_project/local-backup`；不需要旧实例的地址或凭据。在 Linux 上：

```bash
python -m pip install zstandard==0.23.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
python /root/vllm_project/research/scripts/verify_instance_supplement.py \
  --manifest /root/vllm_project/local-backup/manifest.json \
  --destination /root/vllm_project/recovered-native-source
```

脚本先核对压缩文件 SHA256，再逐成员核对大小、路径和 SHA256；不恢复 symlink，不覆盖已有目标。解压后的 `system/` 对应旧 `/root/vllm_project/`，`data/` 对应旧 `/data/vllm_project/`。保持历史树完整，按对应 CANN/torch-npu 版本重新导出 PROF；旧 9.0 采集和本轮 9.1 采集不要当作同一解析栈。

## 9. 以后继续时的第一项工作

向后续执行者提供本手册，并使用以下交接说明：

> 先读 docs/resume_on_new_instance.md、docs/technical_walkthrough.md 和报告 10/11。固定 0.26 栈恢复后，先验收正确性与实际导入路径，再复测已有 baseline。当前完成的是 WY 等长修复、布局复现和 prepare 组件筛选；下一步只做 baseline/仅布局/仅 prepare/组合四臂模型 A/B。适配真实 QKV 布局，避免重复归一化，记录实际路径命中。默认异步 graph 的基线生成波动仍需控制。不要把 #16637/#15920 历史 patch 默认启用，不要把组件收益当整模型收益，也不要声称这些社区 PR 是原创算法。

如果改用最新主线，先重新核查社区状态与 #18723 的输入布局，建立新的匹配环境和基线；这是另一组实验，不与旧 0.26 结果混合比较。
