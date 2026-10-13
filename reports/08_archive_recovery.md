# 代码与证据持久化验收

日期：2026-10-14（Asia/Shanghai）。本次任务是归档和验证恢复，不是新的 NPU 性能实验。

## 保存内容

| 内容 | 保存方式 | 恢复入口 |
|---|---|---|
| #16637、#15920 四个原始源码 commit | 两个增量 Git bundle，保留原 SHA | `scripts/restore_sources.py` |
| 独立改动及单元测试 diff | 四个 patch，按 Issue、upstream repo 区分 | `patches/README.md` |
| 55 个原始 benchmark/服务日志/逐步 worker 记录 | `benchmark-records.tar.gz`，约 459 KiB | `scripts/restore_evidence.py` |
| Runtime Graph 全部已导出时间线和 CSV（6 文件） | 约 32 MiB 压缩包，未压缩约 521 MiB | 加 `--include-profiler` |
| Runtime Eager 全部已导出时间线和 CSV（6 文件） | 约 125 MiB，拆为 4 个分片，未压缩约 1.68 GiB | 按 manifest 自动拼接 |
| 已安装 Python 包版本及已有 CANN/模型配置 | `dependencies-lock.json`、`source_pins.json`、配置模板 | 按兼容栈准备，不能盲目安装所有旧包 |

Git bundle 要求先获取 manifest 中的固定 upstream base。官方 GitHub 失败时恢复脚本会尝试 Gitee 镜像；SHA 固定，镜像不是版本依据。恢复成功不等于源码 base 已在 910B2 上验收。

## 实际验证

- 新目录初始化 Git，从本地 upstream checkout 提取同一 base，导入 bundle，两个 #16637 HEAD 精确匹配原始 SHA，clean。
- 在恢复后的目录切换到 #15920，两个 HEAD 精确匹配原始 SHA，clean。此处验证的是 Git 对象和恢复流程，未测试公网下载或 NPU 软件兼容性。
- 全部压缩包/分片 SHA256 通过；所有压缩包成员的名称、大小和数量核对通过。
- 55 个小型记录已实际解压到新目录。
- 所有压缩包解压流都扫描了 GitHub token、私钥和已知 SSH 凭据特征，没有匹配；源码 bundle 仅包含已审查的代码改动。
- 恢复脚本的 Python 语法检查通过；`git diff --check` 通过。

机器可读证据见 [`evidence/migration_20261014/recovery-verification.json`](../evidence/migration_20261014/recovery-verification.json)。

## 仍需保留的边界

#16637 的 CPU scheduler 回归证明 padding 问题可以复现并修复；#15920 当前只证明 profile-shape 策略分支的源码和 CPU 回归。两者没有新增 NPU OOM/吞吐结论，旧实例的 vLLM 0.22.1 + vLLM-Ascend 0.22.1rc1 结果不能替代其匹配版本的实验。

#15920 原始源码历史接在 #16637 后面；分析单独收益时要比较相邻提交，或将独立 diff 应用到干净 base。源码 bundle、patch 和证据报告各自保留，不把社区已有 PA workspace cache、off-stride 修复作为原创成果。

模型权重、CANN/torch_npu wheel 和全部厂商原始二进制采集目录不包含在本次 Git 归档；关键导出时间线、原始 benchmark、日志、配置、代码及失败记录已经保存。恢复后先检查新实例软硬件与基线，再继续实验。
