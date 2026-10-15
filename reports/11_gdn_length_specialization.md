# GDN 分区执行调查与预处理实验

日期：2026-10-15。目标仍为单卡 910B2、BF16、Qwen3.5-4B；沿用报告 10 的匹配 0.26 环境。本报告不把社区复现称为原创算法，也不把局部 kernel 收益外推到模型。

## 能否按不同长度使用不同 kernel

可以，但分派依据必须是当前算子输入的实际 shape 和执行阶段。请求总长 32K 可能被切成多个 prefill chunk；多个短请求又可能合成一个大 token batch。因此不能只用 HTTP 请求的 prompt 长度写一个短/长分支。

可行结构是有限个预编译实现，加一个开销很低的分派层。先区分 prefill/decode，再检查总 token 数 T、序列数/分段 metadata、H/Hg/D、dtype、PCP、输入布局及 graph 模式。只使用已有 host metadata 和 tensor shape，避免为分派新增 `.item()`、设备同步或在线 autotune。图捕获时确定对应实现，重放沿用捕获结果。

收益目标是覆盖更多代表性负载并控制回退；不能保证所有长度、并发、模型都有正收益。短请求可能主要受 host 调度、投影/卷积或 kernel 发射影响，而长 prefill 更容易暴露布局搬运成本。未测的模型和硬件组合保留基线。

## 最新版本是否已集成全部 PR

不是。vLLM 主仓库、vLLM-Ascend 插件与 CANN/torch-npu 算子栈需要分别检查；安装最新 vLLM 不等于安装或启用最新 Ascend kernel。PR 合并、进入某个 release、满足设备能力、实际命中该路径也是不同条件。

本次 GitHub API 主线固定到 `c5abc52f0c5e68b297779c722dedd940b07ef2ee`。API 的最新 vLLM 正式 release 为 `v0.31.0`；Ascend 最近发布的是 `v0.30.0rc2`（预发布，2026-10-13），GitHub 标为 latest 的非预发布版本为 `v0.23.0.post1`。这些版本不能任意组合；本轮没有升级现有实验环境。[vLLM release](https://github.com/vllm-project/vllm/releases/tag/v0.31.0)、[Ascend rc2](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.30.0rc2)、[Ascend 正式 release](https://github.com/vllm-project/vllm-ascend/releases/tag/v0.23.0.post1)。

| PR | 本次状态 | 实际覆盖与组合关系 |
| --- | --- | --- |
| [#14333](https://github.com/vllm-project/vllm-ascend/pull/14333) | open / 未合并 | WY 等长索引修复；属于正确性前置，不是模型 packed 路径的吞吐来源 |
| [#16538](https://github.com/vllm-project/vllm-ascend/pull/16538) | open / 未合并 | WY 直接输出 head-first，并跳过不返回的中间值转换；与 prepare 融合位于不同阶段，可组合验证 |
| [#16524](https://github.com/vllm-project/vllm-ascend/pull/16524) | open / 未合并 | 跳过 suppressed 中间值转换；已包含在 #16538 的覆盖范围内，不能重复算收益 |
| [#17604](https://github.com/vllm-project/vllm-ascend/pull/17604) | open / 未合并 | qkv split/copy + q/k norm + gating 融合；含两档 BLOCK_T 和 CompiledKernel 缓存，适合检验短 prefill/decode 的准备阶段 |
| [#16811](https://github.com/vllm-project/vllm-ascend/pull/16811) | open / 未合并 | 合并 q/k norm、复用混合 batch 的归一化结果、清 state；与 #17604 部分重叠，应先当替代方案消融 |
| [#12647](https://github.com/vllm-project/vllm-ascend/pull/12647) | merged，2026-08-07 | 可选 fused CANN GDN prefill 路径；存在源码不代表本机 CANN/910B2 支持或当前模型命中 |
| [#18723](https://github.com/vllm-project/vllm-ascend/pull/18723) | merged，2026-10-14 | 主线刚加入 head-first conv 输出、混合 batch 分段及 decode 去 concat；需要与 #17604 重新处理输入布局，不能机械 cherry-pick |
| [#8872](https://github.com/vllm-project/vllm-ascend/pull/8872) | open / 未合并 | MegaGDN/PTO，范围更大；不在本轮重写或盲目引入 |

已直接读取该 main 的 `ops/gdn.py`、`ops/triton/fla/chunk.py` 和 `wy_fast.py`。`chunk.py` 仍在调用 AscendC 前转换 w/u，并在检查 SUPPRESS_LEVEL 前转换 h/v_new；WY 仍保留等长路径旧索引。本次还读取 `v0.30.0rc2` 的 chunk.py，与所查 main 文件相同。相关快照、原始 PR base/head SHA 和 release 元数据保存在 `evidence/gdn_dispatch/`。

即便将来这些 PR 全部合并，也不能据此称为所有场景的 SOTA：通用实现需要兼顾多代硬件、多种模型、精度、TP/PCP、投机解码及图捕获。是否最优需要定义同一设备/软件栈/模型/负载/正确性标准下的对照。当前 0.26 上的复现收益也不能声称超过最新主线或最新商用算子库。

尤其要区分最新主线与本机 0.26：主线新合并的 #18723 已用 head-first 卷积输出的 view 提供 prefill QKV，并为 decode 增加 `_rearrange_decode_qkv` 避免中间 concat；#17604 所查 head 仍按 token-major packed conv 输出编写。融合 prepare 必须接收真实 stride/layout，或选择对应实现，不能为了套旧 PR 先复制回旧布局。本轮 microbenchmark 用 0.26 的真实旧准备链作对照，只证明旧栈上的可行性；其节省不能当作最新主线仍有相同冗余。补充原文见 `latest-layout-merge.json`。

## 分区方案与证据边界

| 区间/阶段 | 当前证据 | 下一步实现策略 |
| --- | --- | --- |
| 低 token batch 的 prefill | 报告 10 的 2K 模型数据波动较大，尚无稳定收益；不能据此断言布局优化导致短请求回退 | 先评估 #17604 的融合 prepare，比较不同 BLOCK_T；必要时比较 #16811 的较小融合方案 |
| 中长 prefill | 报告 10 的 8K 模型吞吐约 +3.2%–4.0%；32K GDN pipeline 耗时 -10.73% | 保留已验证的 head-first 路径，并与 prepare 融合做四臂消融 |
| decode | 既有 WY 改动不修改 recurrent kernel，长 decode 无稳定收益 | 单独比较 eager 和 graph；先验证 prepare 融合在图内是否仍有价值，不拿 eager 的发射节省替代 graph 收益 |
| 混合长短/尾块/投机子集 | 本轮前没有模型级覆盖 | 独立检查索引 gather、空段、非整块；未通过则回到对应基线，不将纯 prefill 结果泛化 |

初期只使用小而有限的 kernel 配置集合。#17604 已有 `T <= vectorcore 数` 时 BLOCK_T=1、其余按 section 大小选 BLOCK_T 的逻辑；Qwen3.5-4B 的 Nk=16、Dk=128 对应另一档为 2。本轮另筛选 BLOCK_T=1/4 作为实验控制，检查其是否存在不同的优势区间。强制配置不等于已经验证的生产分派策略。

## 本轮实验方法

`scripts/benchmark_gdn_prepare.py` 使用固定 PR head `8bf4a11d205ffae9bd0c158dc70770d9733ecfb8` 的原始 `fused_gdn_prepare.py`，通过独立模块加载，不改 site-packages 或模型执行路径。legacy 对照直接提取并执行固定 vLLM 安装源码中的 `rearrange_mixed_qkv` 方法，再调用同环境的 `l2norm_fwd` 与 `fused_gdn_gating_patch`，避免凭 PR 描述重造一个更慢的基线。

先对零长、1/32/64/65/128 token、普通与索引子集做独立 CPU FP32 参考比较。q/k/beta 使用 BF16 对应容差；FP32 g 使用更紧容差；v 必须逐元素相同。随后 1/32/128/512/2048/8192/32768 token 比较 legacy、PR 默认、强制 BLOCK_T=1/4，预热后八轮随机交错、每轮十次。记录 NPU event span 和 wall time；event span 包含 host 发射空隙，不能当作纯 kernel duration。另用 torch_npu.profiler 采集 2K 时间线。

这一步只能决定哪些候选值得进入整模型测试，不能决定最终阈值。后续模型四臂为 baseline、仅布局、仅融合 prepare、两者组合；再分别评估短/中/长 prefill、长 decode、并发 1/4/8 与长短混合。至少用两次独立进程复核优势区间，预热覆盖所有 kernel 配置，计时排除首次编译。

必须同时检查输出/最终 state/生成结果、TTFT/吞吐、额外显存、graph 捕获及实际路径命中。报告 10 中默认异步 graph 的基线输出波动仍需单独控制；固定 batch 的一致性不能替代默认调度验证。若融合只在 microbenchmark 快，或模型收益达不到门槛，就不扩大移植范围。

## 实机结果

在同一台 910B2 上完成三次独立进程：首轮八轮×十次、复测十二轮×二十次、确认十六轮×二十次。首轮覆盖七种长度；确认轮覆盖 1/2K/8K/32K。三次均完成每轮启动前的 46 组 CPU 参考检查，所有 q/k/v/g/beta 通过；timing shapes 另与 legacy 对照。原始文件为 `prepare-screen.json`、`prepare-graph-repeat.json`、`prepare-graph-confirmation.json`。

以下为最后一次确认的 **eager prepare 组件 event span，单位 ms**，并非模型 TTFT 或纯 kernel duration：

| T | 0.26 legacy | PR 默认 | BLOCK_T=4 |
| ---: | ---: | ---: | ---: |
| 1 | 0.8119 | 0.1453 | 0.1462 |
| 2048 | 0.8749 | 0.1456 | 0.1421 |
| 8192 | 0.8962 | 0.2454 | 0.2301 |
| 32768 | 2.3568 | 1.0469 | 0.9480 |

三次进程中，BLOCK_T=4 相对 PR 默认的 eager 耗时降低分别为：8K `8.43% / 6.36% / 6.24%`，32K `8.98% / 9.20% / 9.45%`。小 T 的 forced tiling 差异小，没有选择人为阈值包装成普遍收益。这里复现融合链本身的主要设计来自 #17604；BLOCK_T 筛选是现有 kernel 的参数实验，不是新算法，也未证明社区从未测过该配置。

两次独立进程还对同一 prepare 组件捕获并重放 NPUGraph：

- T=1，legacy/PR 默认的 event span 分别为 `0.2887/0.2852 ms` 与 `0.2863/0.2861 ms`，没有稳定达到 3%。这个单段 graph 测量包含每次 replay 的 host 成本，不能据此说融合在整模型图中完全没有设备收益。
- T=8K，BLOCK_T=1 在两次 graph 对照中都快于默认，尽管其 eager 比默认慢；BLOCK_T=4 相对默认的 graph 差异一正一负，不能直接采用。需要用设备时间线、真实引擎和更多复测决定是否分派。
- T=32K，BLOCK_T=4 的组件图测量也快于默认。不过这些大 T 图是合成的组件捕获，不代表当前 vLLM 的 prefill 已经命中大图；本模型先前实测模式为 FULL_DECODE_ONLY。

因此已经得到“不同 shape/执行模式可能需要不同实现”的实测依据，但尚未得到一个能保证所有请求提速的生产 dispatcher。

最后单独补做 graph 输入更新验收：在已捕获地址原地修改 mixed_qkv/a/b，再重放并与重新计算的 legacy 比较，1/65/8192/32768 token × 四种实现全部通过。结果为 `prepare-graph-input-updates.json`；该次只有一轮一次，专用于正确性，不加入性能范围。前三轮采样未修改输入，归档中保留当时脚本；Git 中最新 harness 已加入此项检查。

2K 的 torch_npu.profiler 记录四个顺序组，每组三次调用、组间同步，合计 30 个设备任务，数量完整。legacy 三次调用有 21 个任务、kernel duration 求和 `305.167 us`；PR 默认为 3 个任务、`104.923 us`；BLOCK_T=1 为 `142.043 us`，BLOCK_T=4 为 `116.323 us`。这说明融合实际把每次准备从七个任务减少到一个；2K 下强制 BLOCK_T=4 的设备时间未超过默认，eager 的细小差别不能当成该 shape 的设备优势。原生 PROF 目录的 `msprof --export=on` 返回 0。profiler 使用默认 level，保留其 RECORD 结束警告，不声称采到全套 AIC 利用率计数。

本轮到此保留为组件筛选，没有将修改注入模型或默认生产环境。下一步的明确边界是：先适配真实输入布局与归一化语义，再做 baseline/仅布局/仅 prepare/组合四臂模型 A/B；不能将两个局部百分比相加。最新主线还需单独接入 #18723 后重新比较。

归档、manifest、恢复命令见 `docs/gdn_prepare_reproduction.md`。Linux 新目录恢复 26 个文件通过；Windows 通过归档哈希校验，但深路径完整解压受系统路径长度限制。最后 NPU 无运行进程；两个生产 source checkout 保持 clean。本轮没有运行上游全量 CI，harness 的 Ruff/format/语法检查通过。
