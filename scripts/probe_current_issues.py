"""Probe #16637's config gate and #15920's generation-mask allocation path.

Run in the installed Ascend environment. This is not a long-request benchmark.
"""

import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
from types import SimpleNamespace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packages = ["vllm", "vllm-ascend", "torch", "torch-npu", "triton-ascend"]
    result = {"packages": {name: metadata.version(name) for name in packages}}
    from vllm.config import SpeculativeConfig

    field = "num_speculative_tokens_per_batch_size"
    result["issue_16637"] = {"dynamic_schedule_field_present": field in SpeculativeConfig.__dataclass_fields__}
    if not result["issue_16637"]["dynamic_schedule_field_present"]:
        try:
            SpeculativeConfig(method="ngram", num_speculative_tokens=3,
                              num_speculative_tokens_per_batch_size=[[1, 8, 3], [9, 32, 0]])
        except Exception as error:
            result["issue_16637"]["config_rejection"] = {
                "type": type(error).__name__, "message": str(error)
            }
    import torch
    import torch_npu  # noqa: F401
    from vllm_ascend.attention.attention_mask import AttentionMaskBuilder

    builder = AttentionMaskBuilder(torch.device("npu:0"))
    rows = []
    for length in [2048, 32768, 98304]:
        torch.npu.synchronize()
        before = torch.npu.memory_allocated()
        config = SimpleNamespace(runner_type="generate", max_model_len=length, dtype=torch.bfloat16)
        mask = builder.get_attention_mask(True, config)
        torch.npu.synchronize()
        rows.append({"configured_length": length, "shape": list(mask.shape),
                     "dtype": str(mask.dtype), "tensor_bytes": mask.numel() * mask.element_size(),
                     "data_ptr": mask.data_ptr(), "allocation_delta_bytes": torch.npu.memory_allocated() - before})
    result["issue_15920"] = {
        "scope": "actual NPU generation-mask helper; not full engine or all attention backends",
        "rows": rows,
    }
    result["installed_sources"] = []
    for package, rel in [
        ("vllm", "vllm/config/speculative.py"),
        ("vllm", "vllm/v1/core/sched/scheduler.py"),
        ("vllm-ascend", "vllm_ascend/attention/attention_mask.py"),
        ("vllm-ascend", "vllm_ascend/worker/model_runner_v1.py"),
    ]:
        path = Path(metadata.distribution(package).locate_file(rel))
        result["installed_sources"].append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
