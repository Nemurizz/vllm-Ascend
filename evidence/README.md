# 证据目录说明

这里保存的是可审计的小型结果和必要的 Profiler 数据。所有 JSON 中的绝对路径都是原始实例路径，仅用于追溯；恢复时应使用 `configs/remote.env` 重新生成结果。

- `remote_env_20261010.txt`：远端环境和模型 revision 记录，已去除 SSH 凭据。
- `source_pins.json`：官方固定版本、远端实际版本和源码入口。
- `aclgraph_followup/`：Graph padding A/B、Runtime API 摘要和 gzip 压缩的两个 msprof SQLite 数据库。解压后文件名分别是 `api_event_graph_msprof.db` 和 `runtime_graph_msprof.db`。
- `source_snapshot/v0.22.1rc1/`：从已安装旧版本提取的关键源码，仅用于审计，不是完整上游 checkout。

完整 msprof JSON、task trace 和模型权重不进入仓库；它们应放在实例数据盘或外部对象存储，并在报告中记录路径和生成命令。

