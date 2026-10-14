# 0.26 实验恢复说明

此目录只恢复实验，不将历史候选补丁自动部署为优化。旧环境的恢复仍见 `recovery.md`；原始性能数字只能与同版本、同模型、同负载比较。

```bash
git clone --branch research/issue-026-validation https://github.com/Nemurizz/vllm-Ascend.git
cd vllm-Ascend
```

## 环境

目标为单卡 910B2 64GB、aarch64、Python 3.11、CANN/910B ops/ATB 9.1.0、torch 2.10.0+cpu、torch_npu 2.10.0.post4、triton-ascend 3.2.2。驱动仍为 26.1.1。新实例须先核查驱动透传、逻辑 NPU 编号与兼容性；禁止直接复制旧实例的物理卡 ID。

可使用[官方 v0.26.0rc1 Dockerfile](https://github.com/vllm-project/vllm-ascend/blob/v0.26.0rc1/Dockerfile) 对应的 CANN 9.1.0 基础镜像。如只有裸容器，官方国内 OBS 的三个 aarch64 包为：

```text
https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.1.0/Ascend-cann-toolkit_9.1.0_linux-aarch64.run
https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.1.0/Ascend-cann-910b-ops_9.1.0_linux-aarch64.run
https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.1.0/Ascend-cann-nnal_9.1.0_linux-aarch64.run
```

下载可加 `Referer: https://www.hiascend.com/`。使用独立 SDK 目录（本次 `/opt/vllm-project-sdk`），不要覆盖旧 CANN 或升级驱动。Toolkit 安装器要求目标父目录权限合规，本次 `/root` 的 700 权限导致首次安装失败，改 `/opt` 后成功。安装器会替换 `/etc/Ascend/ascend_cann_install.info`，安装前保存旧记录，安装后按需恢复旧环境记录；不要通过放宽 `/root` 权限解决这个问题。依次安装 toolkit `--full`、910B ops `--install --type=toolkit`、NNAL `--install --whitelist=atb`，均显式传 `--install-path`。必须以各包 `--help` 和实际安装结果验收。

代码固定点：

```bash
git clone --depth 1 -b v0.26.0 https://gitee.com/mirrors/vllm.git /root/vllm_project/src/stack026/vllm
git -C /root/vllm_project/src/stack026/vllm rev-parse HEAD
# 必须为 568afb3a13806beb53bb2e6bd518269357b237c0
git clone --depth 1 -b v0.26.0rc1 https://gitee.com/mirrors/vllm-ascend.git /root/vllm_project/src/stack026/vllm-ascend
# 对应 f2f74a16c3c50a76f4349d807918e83edec1e35c
```

官方 GitHub 为上游，Gitee 仅作为国内下载渠道；若镜像 tag hash 不匹配，停止而不是用同名 tag 替代。pip 使用清华 `https://pypi.tuna.tsinghua.edu.cn/simple` 与华为 Ascend `https://mirrors.huaweicloud.com/ascend/repos/pypi`。使用独立 venv；torch 必须是 Ascend 对应 2.10 CPU 构建，避免自动下载 vLLM 的 CUDA wheel/Torch 2.11。vLLM 按 `VLLM_TARGET_DEVICE=empty` 从源码安装，native 包按上述版本安装，transformers 固定 5.14.1、numpy 1.26.4。

本次复用了旧环境通用依赖，用 `--no-build-isolation` 和已装 torch 构建 empty backend，额外安装 setuptools-rust≥1.9、setuptools≥77/<81 和 setuptools-scm≥8。FastAPI metadata 存在上游约束冲突，本次保留 vLLM 所需 0.136.3，仅验收离线引擎；不声称在线 API 或 pip check 全绿。完整安装日志与最终依赖清单作为环境证据保存，不把未经测试的 fresh-install 命令冒充已验证一键安装。

复制本仓库 `scripts/` 至 `/root/vllm_project/scripts`，然后在新 shell 中 `source scripts/activate_stack026.sh`。该脚本显式优先导入固定源码，避免旧 vLLM 包遮蔽 editable 安装；不改系统 shell 配置。SDK/项目路径可由 `SDK_ROOT`、`PROJECT_ROOT` 调整。

## 模型与验收

权重只放数据盘。manifest 保存每文件固定版本和 SHA256，下载无需使用可变 master 重新猜版本：

```bash
python scripts/download_modelscope_pinned.py --manifest evidence/stack026/qwen35-modelscope-files.json \
  --output /data/vllm_project/models/Qwen3.5-4B
python scripts/smoke_stack026.py --model /data/vllm_project/models/Qwen3.5-4B --mtp --graph \
  --output /root/vllm_project/results/raw/stack026-smoke.json
```

先检查结果 status、版本、实际导入路径与 generation，再跑性能。成功 smoke 只是基本兼容验收，不是完整正确性验证。

## 四组 A/B 与长上下文

前三组分别从干净原始 scheduler 运行，第四组独立应用候选补丁，禁止同时修改两条执行路径：

```bash
python scripts/benchmark_issue16637.py --model /data/vllm_project/models/Qwen3.5-4B \
  --variant static_graph --output /root/vllm_project/results/raw/static_graph.json
# 同负载再分别 --variant dynamic_graph 和 eager_dynamic
# 独立实验分支中 git apply patches/issue16637-v0.26-candidate.patch
# 然后 --variant dynamic_patched；它仍保留官方 PIECEWISE 可靠性保护。
```

默认 c4/c8/c16、32 请求、2048 输入、256 基础输出（按请求增加0/32/64/96）、三轮。不同输出长度产生请求替换，所有组 prompt IDs 和输出长度相同。数据保存每请求 token IDs；对 dynamic 原版与 patched 逐请求比对。每请求平均 TPOT 不冒充逐 token TPOT 分布。`--profile-dir` 会单独启用 torch_npu Level1/PipeUtilization；不要将 profiler 耗时作为吞吐收益。

长上下文对照不应用历史 #15920 shape patch：

```bash
python scripts/probe_issue15920_long_context.py --model /data/vllm_project/models/Qwen3.5-4B \
  --input-tokens 32768 --max-num-seqs 1 --output /root/vllm_project/results/raw/long32-seqs1.json
# 同样检查 max-num-seqs 256，再逐级升到 98304；一次只运行一个进程。
# ISSUE15920_INIT_PROFILE_DIR 指定 startup profiler；--profile-dir 指定真实长请求 profiler。
```

记录 init 与真实 prefill 的峰值、mask 大小、attention 是否被 profile 执行。OOM 必须按实际 traceback/profile 归因，不自动算作 #15920 的 V0 dense mask Bug。

## 原始证据与候选源码恢复

```bash
python scripts/restore_evidence.py --archive-root evidence/migration_20261015 \
  --destination /root/vllm_project/restored-026-evidence --include-profiler
```

压缩包包含 114 个历史/本次结果与日志、180 个本次 torch_npu 时间线/CSV 与 msprof CSV。两个 profiler 分片合计 81,464,972 bytes，SHA256 及全部成员记录在 manifest。恢复脚本先校验每片及合并 hash，再拒绝覆盖已有结果。模型权重、SDK/wheel、厂商原始采集目录及数据库没有进入 Git；重新采集可依据固定模型/依赖/源码。

`code/stack026-bundle.json` 保存被否定的 0.26 padding 候选 commit 和 bundle hash，**不默认部署**。固定 base 下载后可以精确恢复 Git 对象：

```bash
git -C /root/vllm_project/src/stack026/vllm bundle verify "$PWD/code/vllm-16637-026.bundle"
git -C /root/vllm_project/src/stack026/vllm fetch "$PWD/code/vllm-16637-026.bundle" \
  refs/heads/codex/issue16637-026-experiment:refs/heads/codex/issue16637-026-experiment
```

恢复候选分支会更改 scheduler，运行其他实验时须回到固定官方 base。依赖清单、CANN 包校验、source pins 在 `evidence/stack026/`；详细实验结论见报告 09。新镜像仍需 smoke 验收，不能把旧镜像的 allocator、Graph 或前端兼容性结果直接套用。
