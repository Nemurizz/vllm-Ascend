# 文档导航

如果第一次阅读这个项目，先看 [完整技术复盘](technical_walkthrough.md)。它按背景、技术栈、模型与调用链、问题发现、社区核查、实现原理、验证、失败处理和迁移恢复解释全过程，区分已经完成的工作与尚未验证的方案。

## 按阅读目标选择

| 想了解的内容 | 入口 |
| --- | --- |
| 为什么选这个问题、具体怎么做、为什么有效 | [完整技术复盘](technical_walkthrough.md) |
| WY Bug 修复与布局优化的全部结果、回退和限制 | [报告 10](../reports/10_gdn_operator_optimization.md) |
| 融合 prepare、不同长度/tiling/graph 的组件实验 | [报告 11](../reports/11_gdn_length_specialization.md) |
| 在新实例安装匹配环境、下载模型和国内来源 | [0.26 环境恢复](stack026_reproduction.md) |
| 恢复两份源码提交并复现布局实验 | [GDN 布局恢复](gdn_operator_reproduction.md) |
| 重跑 prepare 筛选、恢复 profiler 导出 | [GDN prepare 恢复](gdn_prepare_reproduction.md) |
| 查看更早 Graph/Runtime/历史 Issue 的调查 | [阶段报告目录](../reports/) |
| 浏览精确修改 | [Bug 修复 patch](../patches/gdn_wy_equal_length_026.patch)、[布局 patch](../patches/gdn_head_first_026.patch) |

## 结果与来源

布局结果在 [`evidence/gdn_operator/`](../evidence/gdn_operator/)，prepare 结果在 [`evidence/gdn_dispatch/`](../evidence/gdn_dispatch/)。两处都有原始数据、来源审计和正确性记录。源码完整增量历史由 [Git bundle manifest](../code/gdn-operator-bundle.json) 标识；压缩 trace 的 SHA256、成员列表及恢复方法见相应复现文档。

阅读结果时注意适用边界：8K 的约 3.2%–4.0% 是布局方案的模型 prefill 吞吐；prepare 的约 6%–9% 是相对 PR 默认 tiling 的长输入组件耗时差异。后者尚无组合模型 A/B，不能与前者相加，也不能外推到其他模型/硬件。
