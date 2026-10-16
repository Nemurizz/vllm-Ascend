# Ascend 910B2 单卡推理优化：GDN 问题定位与性能验证

在 **Ascend 910B2 64GB + vLLM / vLLM-Ascend** 上，从源码和社区问题出发，完成最小复现、补丁移植、正确性回归、NPU profiling 与模型 A/B。本仓库保存研究代码、原始结果和恢复资料。

**已验证的主要结果：Qwen3.5-4B 的 8K prefill 吞吐提升约 3.2%–4.0%；32K 单层 GDN 流水线耗时降低 10.73%；另复现并修复 WY 等长路径的编译/索引错误。** 收益限于下述环境与负载，短请求和长 decode 的回退同样保留。

> 完整研究内容位于 [`research/gdn-prepare-shapes`](https://github.com/Nemurizz/vllm-Ascend/tree/research/gdn-prepare-shapes)。默认 `main` 的 README 同步展示结果；复现实验请克隆研究分支。实验与社区核查截至 **2026-10-15**。

## 问题背景

项目最初以 Qwen3-8B 调查动态 batch、AclGraph padding 和 Runtime 开销。实测中，精确 graph bucket 的 E2E P50 只改善约 0.63%；PA workspace cache 已有社区实现；去除 replay 前同步的尝试发生超时，因此没有继续扩大这些改动。过程与负向结果保留在[阶段报告](https://github.com/Nemurizz/vllm-Ascend/tree/research/gdn-prepare-shapes/reports)。

随后针对 Qwen3.5-4B 的 GDN（Gated DeltaNet）路径，定位到三类可独立验证的问题：

| 问题 | 源码/实测现象 | 处理方式 |
| --- | --- | --- |
| WY 等长路径正确性 | `cu_seqlens=None` 时触发 Triton 编译错误，且批次/head 地址偏移有误；24 个 shape/dtype 对照中 6 个等长用例失败 | 将等长索引修复独立提交，保留失败用例和 CPU 数学参考 |
| GDN 布局搬运冗余 | WY 产生的 w/u 需要转换为 AscendC 消费的 head-first 布局；部分不会返回的 h/v_new 仍执行转换 | WY 直接输出目标布局，按返回契约跳过无用转换；保留 PCP guard |
| prepare 发射与 tiling | 旧准备链分别完成 QKV 拆分、归一化和 gating；融合后不同 T、eager/graph 的最优配置仍有差别 | 复现融合 prepare，筛选有限个 BLOCK_T 配置并单独测试 graph 输入更新 |

上述修复/融合主要来自社区 [#14333](https://github.com/vllm-project/vllm-ascend/pull/14333)、[#16538](https://github.com/vllm-project/vllm-ascend/pull/16538)、[#16524](https://github.com/vllm-project/vllm-ascend/pull/16524)、[#17604](https://github.com/vllm-project/vllm-ascend/pull/17604)。本项目的工作是匹配旧栈移植、组合兼容、独立验证、消融与归档；没有将社区方案称为原创算法。

## 验证结果

### 1. 模型吞吐：8K prefill 达到 3% 目标，也保留回退点

模型为 **Qwen3.5-4B BF16，单卡 TP=PCP=1，纯文本，无量化/MTP**。每个配置内预热后按 `A B B A B A A B` 交错测量，每臂四次；下表为相同负载下 token/s 中位数之比，正值表示吞吐提升。A 为未修改 stock，B 为 GDN 布局优化。

| 输入 token / 并发 / 每请求输出 token | Eager / 默认异步 | FULL_DECODE_ONLY / 默认异步 | FULL_DECODE_ONLY / 固定批次同步 |
| --- | ---: | ---: | ---: |
| 2K / 1 / 1 | +1.83% | +2.81% | **−4.66%** |
| **8K / 1 / 1** | **+3.99%** | **+3.56%** | **+3.32%** |
| 32K / 1 / 1 | +2.85% | +3.11% | +2.93% |
| **8K / 4 / 1** | **+3.21%** | **+3.46%** | **+3.43%** |
| 2K / 4 / 128 | +0.34% | **−2.35%** | **−0.76%** |

输出 1 token 的负载用于 prefill/首 token 研究，不代表一般 decode 吞吐。固定批次组仅在自身配置内做 A/B；其 8K、并发 1/4 的 TTFT 分别降低 **3.23% / 3.35%**。Graph 配置为 `FULL_DECODE_ONLY`，不能把这些结果解释为 prefill 命中完整大图的收益。每臂四次是有限样本，未完成 P99/SLA 或长期稳定性验收。

数据：[模型 A/B 原始汇总](https://github.com/Nemurizz/vllm-Ascend/blob/research/gdn-prepare-shapes/evidence/gdn_operator/summary.json) · [报告 10：配置、全部结果与限制](https://github.com/Nemurizz/vllm-Ascend/blob/research/gdn-prepare-shapes/reports/10_gdn_operator_optimization.md)

### 2. 为什么有效：减少布局复制，数学计算基本不变

32K、BF16、H=32/Hg=16、Dk=Dv=128 的单层 GDN 完整流水线，NPU event span 从 **22.058 ms → 19.693 ms，耗时降低 10.73%**。六轮交错测量，每轮三次，排除首次编译；event span 包含 host 发射间隙，区别于纯设备 kernel duration。

`torch_npu.profiler` 与 `msprof` 原生导出提供了对应证据：

| 32K 流水线设备指标（三次调用合计） | Stock | 布局优化 |
| --- | ---: | ---: |
| kernel task 数 | 72 | 60 |
| transpose 数 | 36 | 24 |
| transpose 累计耗时 | 10.055 ms | 3.110 ms |
| WY kernel 累计耗时 | 17.132 ms | 17.157 ms |

每次 GDN 调用减少四次 transpose，WY 计算耗时基本不变。独立的 8K 模型 profile 中，24 个 GDN 层共少 **96 次 transpose（528 → 432）**，验证改动确实进入模型路径。Profiler 运行与吞吐 A/B 分开，不将带采集扰动的时间算作吞吐样本。

证据：[设备采集摘要](https://github.com/Nemurizz/vllm-Ascend/blob/research/gdn-prepare-shapes/evidence/gdn_operator/profile-summary.json) · [流水线消融](https://github.com/Nemurizz/vllm-Ascend/blob/research/gdn-prepare-shapes/evidence/gdn_operator/gdn-pipeline.json)

### 3. 正确性与 prepare 筛选

- **WY Bug：** stock 为 18/24 用例通过、6 个等长失败；修复及 head-first 版本均为 **24/24 通过**。独立恢复源码后的 NPU pytest 另有 **12 个参数化用例通过**。模型走 packed 路径，因此这个等长 Bug 修复不是模型提速的来源。
- **GDN 数值/返回契约：** 对照独立 CPU FP32 递推参考，输出和最终 state 检查通过；大 shape 消融逐元素一致，`SUPPRESS_LEVEL=0/3` 返回契约通过。单卡验证不代表 PCP 多卡已验收。
- **生成检查：** eager 和固定批次 graph 各 88 个请求，A/B token 一致。默认异步 graph 观察到输出分叉，未修改 stock 的独立对照也生成相同分叉输出；原因尚未确定，保留为限制。
- **融合 prepare：** 2K profile 的每次调用设备任务 **7 → 1**。三次独立进程中，`BLOCK_T=4` 相对 PR 默认配置，8K eager 组件耗时降低 **6.24%–8.43%**，32K 降低 **8.98%–9.45%**；46 组 CPU 参考检查通过，graph 原地更新输入后重放检查通过。
- **尚未完成：** prepare 与布局优化的四臂整模型组合实验、生产分派阈值、所有长度收益验证。**prepare 的局部百分比不能与模型收益相加。** T=1 的 graph prepare 未获得稳定 3% 收益，8K 的 eager/graph 最优 tiling 也不同。

详见 [正确性原始记录](https://github.com/Nemurizz/vllm-Ascend/blob/research/gdn-prepare-shapes/evidence/gdn_operator/wy-layout-correctness.json)、[报告 11：prepare 与长度分区](https://github.com/Nemurizz/vllm-Ascend/blob/research/gdn-prepare-shapes/reports/11_gdn_length_specialization.md)。这些结果未被替换为默认生产实现，也未证明优于最新主线；核查时主线已合并的 #18723 改变了输入布局，需要重新适配与建立基线。

## 实验环境与技术栈

| 层次 | GDN 阶段的固定配置 |
| --- | --- |
| 硬件 / 系统 | 单张 Ascend 910B2 64GB，aarch64 / openEuler |
| 模型 | Qwen3.5-4B，BF16，纯文本；权重逐文件 revision/SHA256 固定 |
| 引擎 | vLLM 0.26.0 `568afb3`；vLLM-Ascend 0.26.0rc1 `f2f74a1` |
| 算子与运行时 | CANN / 910B ops / ATB 9.1.0；Triton Ascend 3.2.2；复用现有 AscendC H/O kernel |
| Python 栈 | Python 3.11.15；torch 2.10.0+cpu；torch-npu 2.10.0.post4；Transformers 5.14.1；NumPy 1.26.4 |
| 性能与回归工具 | NPU events、torch_npu.profiler、msprof、CPU 数学参考、pytest、交错 A/B 与消融、Git bundle/SHA256 恢复校验 |

技术涉及 Triton 指针/布局与 launch grid、AscendC 接口对接、vLLM worker RPC 与固定批次控制、eager/NPUGraph、CPU/NPU 时间线和数据持久化。没有重写 ATB 或完整 Attention/GDN 算子。早期 Qwen3-8B / 0.22 / CANN 9.0 的实验独立保存，不混入上述结果。

固定版本和完整 SHA 见[恢复 manifest](https://github.com/Nemurizz/vllm-Ascend/blob/research/gdn-prepare-shapes/configs/resume026-manifest.json)。已验证的是离线文本引擎；原环境存在 pip metadata 约束冲突，完整空镜像一键安装及 API 前端/多模态尚未验收。

## 代码、复盘与新实例恢复

| 需要查看的内容 | 入口 |
| --- | --- |
| 从发现问题到验证结果的完整过程 | [技术复盘：背景、原理、调用链、实施和止损](https://github.com/Nemurizz/vllm-Ascend/blob/research/gdn-prepare-shapes/docs/technical_walkthrough.md) |
| WY Bug 与布局修改、独立提交/补丁/源码 bundle | [布局复现说明](https://github.com/Nemurizz/vllm-Ascend/blob/research/gdn-prepare-shapes/docs/gdn_operator_reproduction.md) · [代码索引](https://github.com/Nemurizz/vllm-Ascend/blob/research/gdn-prepare-shapes/code/README.md) |
| 融合 prepare、tiling 与 graph 实验 | [prepare 复现说明](https://github.com/Nemurizz/vllm-Ascend/blob/research/gdn-prepare-shapes/docs/gdn_prepare_reproduction.md) |
| 空实例安装、国内源、模型下载、校验与续做 | [新实例续做手册](https://github.com/Nemurizz/vllm-Ascend/blob/research/gdn-prepare-shapes/docs/resume_on_new_instance.md) |
| 所有阶段报告、脚本与历史失败实验 | [文档导航](https://github.com/Nemurizz/vllm-Ascend/blob/research/gdn-prepare-shapes/docs/README.md) |

```bash
git clone --branch research/gdn-prepare-shapes https://github.com/Nemurizz/vllm-Ascend.git
cd vllm-Ascend
# 按 docs/resume_on_new_instance.md 验收环境、下载固定版本依赖和模型，再恢复实验。
```

代码、主要结果、配置和选定 profiler 导出已保存在 Git。额外原始 PROF/FRAMEWORK 采集位于本地 `D:\AI_Inference_Project\vllm_ascend_preservation_20261015`，3360 个归档成员已完成哈希验收，见[持久化审计](https://github.com/Nemurizz/vllm-Ascend/blob/research/gdn-prepare-shapes/evidence/preservation_20261015/summary.json)。模型重新下载到数据盘，缓存、结果和采集放系统盘；正常新实验无需先恢复本地历史大包。

继续点：先验证新实例与已有基线，再评估 baseline / 仅布局 / 仅 prepare / 组合四臂模型 A/B。#16637/#15920 的历史候选未被确认为已验证修复，不默认启用，原因见[报告 09](https://github.com/Nemurizz/vllm-Ascend/blob/research/gdn-prepare-shapes/reports/09_current_environment_and_026_ab.md)。
