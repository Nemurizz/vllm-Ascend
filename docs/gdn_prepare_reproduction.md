# GDN prepare 分区实验恢复

沿用 [0.26 环境恢复](stack026_reproduction.md) 和 [GDN 布局恢复](gdn_operator_reproduction.md)，本轮不需新的权重、SDK 或依赖。固定版本与限制见 [报告 11](../reports/11_gdn_length_specialization.md)。

```bash
source /root/vllm_project/scripts/activate_stack026.sh
mkdir -p /root/vllm_project/results/gdn-prepare
cp scripts/benchmark_gdn_prepare.py scripts/benchmark_wy_operator.py /root/vllm_project/scripts/
python /root/vllm_project/scripts/benchmark_gdn_prepare.py \
  --source "$PWD/evidence/gdn_dispatch/pr17604_fused_gdn_prepare.py" \
  --output /root/vllm_project/results/gdn-prepare/prepare-screen.json \
  --profile-dir /root/vllm_project/results/gdn-prepare/profile

# 单独的独立进程，不与上一个 NPU benchmark 并行运行。
python /root/vllm_project/scripts/benchmark_gdn_prepare.py \
  --source "$PWD/evidence/gdn_dispatch/pr17604_fused_gdn_prepare.py" \
  --output /root/vllm_project/results/gdn-prepare/prepare-graph.json \
  --graph --rounds 12 --repetitions 20
```

PR 原始文件固定到 `8bf4a11d205ffae9bd0c158dc70770d9733ecfb8`，官方字节 SHA256 `5df365868fe81c1e6c1a675b2cf77985cb0fc3a072aef6a626fc17afb48c971b`。实测使用的传输快照在文件末尾多一个空行，SHA256 为 `438ba4f6f6a0f0b5552b1bb4cdc477be7a488e9b69a898df407394f6e4be303b`；归档保留该运行快照，Git 快照已恢复官方原始字节，差异校验在 `snapshot-byte-verification.json`。所有可执行代码相同，保留其版权/许可，没有改写 kernel 算法。实验模块分别使用 PR 原始策略及强制 BLOCK_T=1/4；这些强制版本只存在于实验进程。模块缓存和启动接口沿用原 PR；不能将其全局 cache 当作新生产分派设计。

legacy 基线来自固定 vLLM 安装文件的实际 `rearrange_mixed_qkv` 方法 AST，以及同环境 l2norm/gating。该基线是 0.26 的旧路径，**不是最新 main 的 #18723 路径**。源文件位置/哈希、各轮随机顺序、event/wall samples、正确性差异均保存在 JSON。PR 默认与 BLOCK_T=1 在小 T 时会编译相同配置，这是用于观察噪声的重复控制。

普通计时先预热，再交错多个版本；首次编译不计时。`--graph` 在计时后原地修改输入并验证 replay 输出，防止只验证同一份静态数据。每个 shape 的图单独捕获，大 T 图是合成组件实验，不代表 vLLM 的实际 graph 命中率。模型、完整 recurrent state、混合在线请求和 graph 更新集成仍需后续验证。

保存的三轮性能结果在 `evidence/gdn_dispatch/prepare-*.json`。另有单独 graph 输入更新验收，少量重复只用于正确性，不作为性能确认。Torch 和 msprof CSV/timeline 已压缩保存，恢复命令：

```bash
python scripts/restore_evidence.py \
  --archive-root evidence/migration_gdn_dispatch_20261015 --include-profiler \
  --destination /root/vllm_project/restored-gdn-dispatch
```

归档含 11 个 benchmark/源码/日志文件和 15 个选定 profiler 导出，共约 89 KB；SHA256 与成员大小在 manifest 中。完整恢复已在 Linux 新目录验证。Windows 深目录恢复触发了系统路径长度限制；本地 archive 哈希验证通过，Linux 迁移验收通过，未将 Windows 的部分解压当成完整成功。恢复工具拒绝覆盖已有目标，应使用新目录。
