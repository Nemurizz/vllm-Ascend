"""Verify the public/suppressed intermediate contract of the GDN reproduction."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401
from benchmark_gdn_operator import data
from gdn_experiment import modules
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    from vllm_ascend.utils import enable_custom_op

    assert enable_custom_op(), "Pinned A2 custom operators must be available"
    init_device_properties_triton()
    mods, _ = modules(args.output.parent / "contract-variants")
    torch.manual_seed(1902)
    kwargs = data([64, 128], heads=4, key_heads=2, dim=128)
    result = []
    for level in [0, 3]:
        for mod in mods.values():
            mod.SUPPRESS_LEVEL = level
            mod.get_pcp_group = lambda: SimpleNamespace(world_size=1)
            mod.get_forward_context = lambda: SimpleNamespace(attn_metadata=None)
        baseline = mods["stock"].chunk_gated_delta_rule_fwd(**kwargs)
        for name, mod in mods.items():
            values = mod.chunk_gated_delta_rule_fwd(**kwargs)
            torch.npu.synchronize()
            for a, b in zip(baseline, values):
                if a is None:
                    assert b is None
                else:
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            result.append(
                {
                    "level": level,
                    "variant": name,
                    "status": "passed",
                    "shapes": [None if x is None else list(x.shape) for x in values],
                }
            )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
