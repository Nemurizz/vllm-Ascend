# 历史 patch 说明

`offstride_capture_max.patch` 是在 vLLM 0.22.1/vLLM-Ascend 0.22.1rc1 安装副本上做的最小配置回移验证：将 Ascend platform maximum 追加到 capture size 列表。

对应上游 Issue [#16049](https://github.com/vllm-project/vllm-ascend/issues/16049) 已由 [PR #16970](https://github.com/vllm-project/vllm-ascend/pull/16970) 合并修复，因此它不是当前版本的新贡献。只有在复现旧环境时才应用；应用前必须备份安装文件并记录 Python 包版本。


`runtime_pa_workspace_cache.patch` 是针对 vLLM-Ascend 0.22.1rc1 的 PA FULL graph 回移/复现补丁：在一次 `update_graph_params()` 调用内按 workspace 签名复用 `_npu_paged_attention_get_workspace()` 结果。它与已合并的上游 [PR #15579](https://github.com/vllm-project/vllm-ascend/pull/15579) 是同一类改动，因此不是新的社区缺口，也不应直接提交为原创 PR。它只影响 PA 分支，不影响默认 FIA、MLA、投机解码或 ENPU。910B2 的 5×128 固定负载中输出吞吐提升 19.77%，且 5 个请求的文本和 usage 与未打补丁对照完全一致；该数字用于证明旧版本回移有效，不能用于宣称原创收益。
