# 证据目录说明

这里保存的是可审计的小型结果和必要的 Profiler 数据。所有 JSON 中的绝对路径都是原始实例路径，仅用于追溯；恢复时应使用 `configs/remote.env` 重新生成结果。

- `remote_env_20261010.txt`：远端环境和模型 revision 记录，已去除 SSH 凭据。
- `source_pins.json`：官方固定版本、远端实际版本和源码入口。
- `aclgraph_followup/`：Graph padding A/B、Runtime API 摘要和 gzip 压缩的两个 msprof SQLite 数据库。解压后文件名分别是 `api_event_graph_msprof.db` 和 `runtime_graph_msprof.db`。
- `source_snapshot/v0.22.1rc1/`：从已安装旧版本提取的关键源码，仅用于审计，不是完整上游 checkout。
- `runtime_atb_*.json`：Qwen3-8B FULL/FIA、eager、FULL_DECODE_ONLY/PA 和 replay-sync 失败实验的轻量摘要；`runtime_atb_worker_*.json` 保留逐段计时统计。
- `patches/runtime_pa_workspace_cache.patch`：仅覆盖 PA graph update 的 workspace 复用实验补丁；已在 910B2 上做固定负载正确性和性能对照，未写回系统安装副本。

完整 msprof JSON、task trace 和模型权重不进入仓库；它们应放在实例数据盘或外部对象存储，并在报告中记录路径和生成命令。

