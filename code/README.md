# 原始源码提交的迁移归档

两个 `.bundle` 文件保存 vLLM、vLLM-Ascend 中本项目创建的 Git commit、tree 和 blob 对象。它们是相对于固定 upstream base 的增量 bundle，不包含模型权重或全部上游历史。完整 SHA、文件校验值、国内镜像及 prerequisites 见 `source-bundles.json`。

在新实例 clone 本研究仓库之后：

```bash
python scripts/restore_sources.py --destination "$HOME/vllm_project/src/issue_checkouts" --issue 16637
# 切到后续 profile-shape 分支：
python scripts/restore_sources.py --destination "$HOME/vllm_project/src/issue_checkouts" --issue 15920
```

脚本优先从官方仓库下载固定 upstream base，失败后使用 Gitee 镜像；随后验证 bundle SHA256、导入原始对象、校验 HEAD 和 clean 状态。Git bundle 保留原 commit SHA，不依赖旧 Windows checkout 或租赁容器。脚本不安装依赖、不修改当前 NPU 安装栈、不重置已有分支；发现本地修改或分支偏离时停止。

两个原始源码历史均按研究顺序排列：base → #16637 → #15920。因此 `codex/issue-15920-profile-shape` 包含前一个 #16637 commit；只比较 #15920 的变更时用 `git diff codex/issue-16637-dynamic-padding..codex/issue-15920-profile-shape`。单独的 #15920 diff patch 仍在 `patches/`，不能把累计分支对 base 的结果全归因给 #15920。

2026-10-15 更正：#15920 提交是历史 shape 策略实验。V1 profile 绕过 attention，helper 策略单元测试不构成原始 OOM 修复证据；恢复历史代码不等于应部署该补丁。见 [`reports/07_issue_15920_profile_shape.md`](../reports/07_issue_15920_profile_shape.md)。

这两个源码 base 是之前调查时从镜像取得的固定快照，均不是已经在 910B2 上验收的软件组合。恢复成功只说明代码和提交可恢复；NPU 兼容性、原始 Issue 的实际 OOM、吞吐收益需要另外验证。不要覆盖已经可运行的 0.22.1 环境来冒充验证。

另有 `vllm-16637-026.bundle`，保存官方 0.26 base 上的独立候选 `ad7a83f`，SHA 和 prerequisite 在 `stack026-bundle.json`。它已经做 NPU A/B 并被否定为可宣称的性能修复；保存用于复现负结果，不能默认部署。独立新目录导入后 HEAD 精确匹配，clean，见 `evidence/stack026/source-recovery-proof.txt`。
