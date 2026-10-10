#!/usr/bin/env python3
"""Reproduce the Ascend off-stride default ACLGraph bucket calculation.

Run this with the pinned vLLM/vLLM-Ascend environment. It only exercises
configuration logic; no model or NPU allocation is required.
"""

import argparse
import json
from types import SimpleNamespace

from vllm_ascend.platform import NPUPlatform


def sizes_for(max_num_seqs: int):
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            max_cudagraph_capture_size=None,
            cudagraph_capture_sizes=None,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        speculative_config=None,
    )
    maximum = NPUPlatform._get_default_max_cudagraph_capture_size(config)
    ladder = [1, 2, 4] + list(range(8, min(maximum + 1, 256), 8))
    ladder = sorted(set(x for x in ladder if x <= maximum))
    return {
        "max_num_seqs": max_num_seqs,
        "platform_default_max": maximum,
        "ladder": ladder,
        "covers_max_num_seqs": max_num_seqs in ladder,
        "largest_captured_size": max(ladder) if ladder else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-num-seqs", nargs="+", type=int, default=[8, 12, 16, 20, 24, 32])
    ap.add_argument("--output")
    args = ap.parse_args()
    rows = [sizes_for(n) for n in args.max_num_seqs]
    text = json.dumps(rows, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
