# 证据目录说明

这里保存的是可审计的小型结果和必要的 Profiler 数据。所有 JSON 中的绝对路径都是原始实例路径，仅用于追溯；恢复时应使用 `configs/remote.env` 重新生成结果。

- `remote_env_20261010.txt`：远端环境和模型 revision 记录，已去除 SSH 凭据。
- `source_pins.json`：官方固定版本、远端实际版本和源码入口。
- `aclgraph_followup/`：Graph padding A/B、Runtime API 摘要和 gzip 压缩的两个 msprof SQLite 数据库。解压后文件名分别是 `api_event_graph_msprof.db` 和 `runtime_graph_msprof.db`。
- `source_snapshot/v0.22.1rc1/`：从已安装旧版本提取的关键源码，仅用于审计，不是完整上游 checkout。
- `runtime_atb_*.json`：Qwen3-8B FULL/FIA、eager、FULL_DECODE_ONLY/PA 和 replay-sync 失败实验的轻量摘要；`runtime_atb_worker_*.json` 保留逐段计时统计。
- `runtime_atb_upstream_status.json`：社区重复性核查；记录 Issue #15575、已合并 PR #15579 和当前主线 `UpdatableGraph` 的直接证据。PA workspace cache 在本项目中只作为旧版本回移/复现，不作为原创优化。
- `patches/runtime_pa_workspace_cache.patch`：仅覆盖 PA graph update 的 workspace 复用实验补丁；已在 910B2 上做固定负载正确性和性能对照，未写回系统安装副本。
- `migration_20261014/`：55 个原始 benchmark/日志、Runtime Graph/Eager 完整导出时间线及 CSV 的压缩归档；包含分片 SHA256、成员清单、安装包版本和恢复验证记录。通过 `scripts/restore_evidence.py` 恢复。
- `../code/`：两个源码仓库的原始研究 commit 对象归档；通过 `scripts/restore_sources.py` 恢复。

2026-10-14 的归档保存了 Runtime 的完整已导出 `trace_view.json`，但未保存全部 msprof 厂商二进制采集目录，也未保存模型权重。Graph msprof 的两个压缩数据库、API/op 统计和旧源码快照仍在已有证据目录中。

