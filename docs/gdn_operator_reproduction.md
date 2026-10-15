# GDN 算子复现与迁移（910B2 / 0.26）

本轮采用社区方案复现：等长 WY 索引参考 [#14333](https://github.com/vllm-project/vllm-ascend/pull/14333)，head-first 输出及跳过无用转换参考 [#16538](https://github.com/vllm-project/vllm-ascend/pull/16538)（与 [#16524](https://github.com/vllm-project/vllm-ascend/pull/16524) 有重叠）。不是原创 GDN 算法，也没有重写现有 AscendC/CANN 算子。

基础环境、国内下载地址、模型逐文件 SHA 和安装限制沿用 [0.26 恢复说明](stack026_reproduction.md)。本轮不新增权重；使用数据盘上的 Qwen3.5-4B BF16。0.22.1 的 Qwen3-8B 基线仍保留，本轮结果不能外推为其加速收益。

## 固定版本与恢复

- vLLM：`568afb3a13806beb53bb2e6bd518269357b237c0`（v0.26.0）。
- vLLM-Ascend 基线：`f2f74a16c3c50a76f4349d807918e83edec1e35c`（v0.26.0rc1）。
- torch/torch-npu：2.10.0+cpu / 2.10.0.post4；Triton Ascend 3.2.2；CANN/ATB 9.1.0。
- 代码补丁分开保存为修复与性能两个 commit，SHA 与 bundle 校验见 `code/gdn-operator-bundle.json`。
- GitHub 主线核对固定在 `f73001e0b9b82419d34eb23919a5eef8e6018682`；NPU 运行环境是上述 release 组合。

直接推送带上游历史的源码分支时，GitHub 因凭据缺少 workflow scope 而拒绝，本轮未创建该远端源码分支。研究分支内的 bundle 保留完整的两个增量源码提交，补丁可直接审阅；无需修改 GitHub workflow 权限即可恢复。

先克隆本研究仓库，切到 `research/gdn-operator-910b`。恢复上一阶段的 0.26 环境与模型后，可使用 bundle 恢复代码历史：

```bash
git clone https://gitee.com/mirrors/vllm-ascend.git /root/vllm_project/src/gdn-operator-026
git -C /root/vllm_project/src/gdn-operator-026 checkout f2f74a16c3c50a76f4349d807918e83edec1e35c
git -C /root/vllm_project/src/gdn-operator-026 bundle verify /path/to/research/code/gdn-operator-026.bundle
git -C /root/vllm_project/src/gdn-operator-026 fetch /path/to/research/code/gdn-operator-026.bundle codex/gdn-operator-910b:codex/gdn-operator-910b
git -C /root/vllm_project/src/gdn-operator-026 checkout codex/gdn-operator-910b
```

也可在基线上依次应用 `patches/gdn_wy_equal_length_026.patch`、`patches/gdn_head_first_026.patch`。不要向其他版本盲目应用，也不要修改系统默认 CANN/驱动。

## 实验入口

将研究仓库的 `scripts/` 同步到 `/root/vllm_project/scripts/`，然后：

```bash
source /root/vllm_project/scripts/activate_stack026.sh
mkdir -p /root/vllm_project/results/gdn-operator /root/vllm_project/logs/gdn-operator
SOURCE=/root/vllm_project/envs/vllm026/lib/python3.11/site-packages/vllm_ascend/ops/triton/fla/wy_fast.py

# 只生成隔离模块，不改已安装 package。
python /root/vllm_project/scripts/test_wy_operator.py \
  --source "$SOURCE" --output /root/vllm_project/results/gdn-operator/wy-layout-correctness.json
python /root/vllm_project/scripts/benchmark_gdn_operator.py \
  --output /root/vllm_project/results/gdn-operator/gdn-pipeline.json \
  --profile-dir /root/vllm_project/profiles/gdn-operator/pipeline
python /root/vllm_project/scripts/check_gdn_return_contract.py \
  --output /root/vllm_project/results/gdn-operator/return-contract.json

# 相同 engine 内切换算子实现，先预热两组，再执行 ABBA/BAAB。
python /root/vllm_project/scripts/benchmark_gdn_engine.py \
  --model /data/vllm_project/models/Qwen3.5-4B \
  --output /root/vllm_project/results/gdn-operator/engine-eager.json
python /root/vllm_project/scripts/benchmark_gdn_engine.py \
  --model /data/vllm_project/models/Qwen3.5-4B --graph \
  --output /root/vllm_project/results/gdn-operator/engine-graph.json \
  --profile-dir /root/vllm_project/profiles/gdn-operator/engine-graph
python /root/vllm_project/scripts/benchmark_gdn_engine.py \
  --model /data/vllm_project/models/Qwen3.5-4B --prefill-only \
  --output /root/vllm_project/results/gdn-operator/engine-confirmation.json
```

每次只运行一个 NPU benchmark。首次编译、模型初始化和两组 shape warmup 不进入测量。最终脚本启用 `disable_log_stats=False` 收集请求 metrics；早期 eager/graph 记录若 `ttft_s=null`，只能使用原始 workload wall time，不得将其冒充 TTFT。是否记录 metrics 以 JSON 的实际 `engine_config` 为准。

`gdn_experiment.GDNExperimentWorker` 通过 collective RPC 切换 `chunk_gated_delta_rule_fwd`，保留图模式、模型、请求参数和 decode 实现；调用计数可核对目标分支确实执行。隔离模块由固定基线生成，遇到锚点不匹配应停止并重新审计。

## 正确性与分析

WY 测试覆盖 BF16/FP32、等长两批次、变长/空段/尾块、不同 K/V 维度，并与独立 CPU 分块矩阵乘参考比较。完整 GDN 使用官方 metadata builder，在合法形状上比较输出、最终 recurrent state 及输入 state 是否被改写；返回中间值的 `SUPPRESS_LEVEL >= 3` 单独验证。

独立调用完整 GDN 时不要用“没有 metadata 的基线崩溃”作为布局优化的收益。该路径另有社区 [#14192](https://github.com/vllm-project/vllm-ascend/pull/14192) 讨论。本轮初始失败日志也保留，不能当成新问题或已定位根因。

```bash
python /root/vllm_project/scripts/summarize_gdn_operator.py \
  --evidence /root/vllm_project/results/gdn-operator \
  --output /root/vllm_project/results/gdn-operator/summary.json

for d in /root/vllm_project/profiles/gdn-operator/pipeline/*/*/PROF_*; do
  msprof --export=on --output="$d"
done
python /root/vllm_project/scripts/summarize_npu_profiles.py \
  --profile-root /root/vllm_project/profiles/gdn-operator \
  --output /root/vllm_project/results/gdn-operator/profile-summary.json
```

Torch profiler 的 kernel CSV 与 `msprof` 的原生导出共同核对 transpose 次数及设备耗时。NPU event 测量跨度包含 host 发射间隙；kernel duration 求和与模型 wall time 是不同指标。长 decode、短 prefill、长 prefill 的回退/收益都应保留。

本轮仅验证单张 910B2、TP=PCP=1、BF16 文本路径；PCP 多卡、950、量化、多模态和其他模型的收益均未验证。完整 serving frontend 的依赖冲突仍见 0.26 环境报告。

## 额外对照与原始记录恢复

`--prefill-only` 是后续扩大到 c8 的入口，本轮首次运行因旧 metrics 字段失败（零个测量 case），不计作已完成的 c8 数据。本轮额外完成的固定批次图模式对照为：

```bash
python /root/vllm_project/scripts/benchmark_gdn_engine.py \
  --model /data/vllm_project/models/Qwen3.5-4B --graph --sync-scheduling --uniprocess \
  --output /root/vllm_project/results/gdn-operator/engine-graph-fixed-batch.json

# 审计默认异步图模式自身的输出变化，此文件不作 A/B 收益样本。
python /root/vllm_project/scripts/benchmark_gdn_engine.py \
  --model /data/vllm_project/models/Qwen3.5-4B --graph --no-log-stats --baseline-control \
  --output /root/vllm_project/results/gdn-operator/graph-baseline-control.json

python scripts/restore_evidence.py --archive-root evidence/migration_gdn_20261015 \
  --include-profiler --destination /root/vllm_project/restored-gdn-evidence
```

恢复目标必须是新目录，工具会拒绝覆盖已有文件。归档包含原始 benchmark JSON、失败日志、隔离实验模块和选定的 Torch/msprof CSV/timeline；SDK、权重、编译缓存及原始 profiler 数据库不在 Git 中。最终结果与代码以报告、JSON manifest 和 bundle 中的 SHA 为准。
