# 历史 patch 说明

`offstride_capture_max.patch` 是在 vLLM 0.22.1/vLLM-Ascend 0.22.1rc1 安装副本上做的最小配置回移验证：将 Ascend platform maximum 追加到 capture size 列表。

对应上游 Issue [#16049](https://github.com/vllm-project/vllm-ascend/issues/16049) 已由 [PR #16970](https://github.com/vllm-project/vllm-ascend/pull/16970) 合并修复，因此它不是当前版本的新贡献。只有在复现旧环境时才应用；应用前必须备份安装文件并记录 Python 包版本。

