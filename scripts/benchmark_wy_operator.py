"""Paired NPU timing and correctness for isolated WY task-geometry variants."""

import argparse
import importlib.util
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch
from build_wy_variants import build


def load(path, name):
    spec = importlib.util.spec_from_file_location(
        f"vllm_ascend.ops.triton.fla.wy_experiment_{name}", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.recompute_w_u_fwd


def inputs(length, heads, dim, packed=True, batch=1, dtype=torch.bfloat16):
    key_heads = heads // 2
    k = torch.randn(batch, length, key_heads, dim, device="npu", dtype=dtype) * 0.1
    v = torch.randn(batch, length, heads, dim, device="npu", dtype=dtype) * 0.1
    beta = torch.rand(batch, length, heads, device="npu", dtype=torch.float32)
    g = -torch.rand_like(beta)
    # Small strictly lower part plus identity; nontrivial matrix products without
    # the numerical conditioning problem of an arbitrary inverse-triangle matrix.
    a = torch.randn(batch, length, heads, 64, device="npu", dtype=dtype) * 0.01
    cu = torch.tensor([0, length], device="npu", dtype=torch.int32) if packed else None
    indices = (
        torch.tensor(
            [[0, i] for i in range((length + 63) // 64)],
            device="npu",
            dtype=torch.int32,
        )
        if packed
        else None
    )
    return {
        "k": k,
        "v": v,
        "beta": beta,
        "g_cumsum": g,
        "A": a,
        "cu_seqlens": cu,
        "chunk_indices": indices,
    }


def event_time(fn, kwargs, repetitions=5):
    start, end = (
        torch.npu.Event(enable_timing=True),
        torch.npu.Event(enable_timing=True),
    )
    torch.npu.synchronize()
    wall = time.perf_counter()
    start.record()
    for _ in range(repetitions):
        _output = fn(**kwargs)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repetitions, (
        time.perf_counter() - wall
    ) * 1000 / repetitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[128, 2048, 8192, 32768]
    )
    args = parser.parse_args()
    torch.manual_seed(1234)
    records = build(args.source, args.output.parent / "variants")
    functions = {
        name: load(Path(rec["path"]), name)
        for name, rec in records.items()
        if not name.startswith("chunk_")
    }
    result = {"variants": records, "cases": [], "failures": []}
    for length in args.lengths:
        case = {
            "length": length,
            "heads": 32,
            "key_heads": 16,
            "dim": 128,
            "packed": True,
        }
        try:
            kwargs = inputs(length, 32, 128)
            reference = functions["stock"](**kwargs)
            torch.npu.synchronize()
            for name in ["fixed", "parallel"]:
                outputs = functions[name](**kwargs)
                torch.npu.synchronize()
                case[name + "_max_abs_diff"] = [
                    float((a.float() - b.float()).abs().max())
                    for a, b in zip(reference, outputs)
                ]
                for a, b in zip(reference, outputs):
                    torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-4)
            for fn in functions.values():
                for _ in range(3):
                    fn(**kwargs)
            torch.npu.synchronize()
            samples = {n: [] for n in functions}
            for repeat in range(4):
                order = (
                    ["stock", "fixed", "parallel"]
                    if repeat % 2 == 0
                    else ["parallel", "fixed", "stock"]
                )
                for name in order:
                    device, wall = event_time(functions[name], kwargs)
                    samples[name].append({"device_ms": device, "wall_ms": wall})
            case["samples"] = samples
            case["median_ms"] = {
                n: statistics.median(x["device_ms"] for x in rows)
                for n, rows in samples.items()
            }
            print(json.dumps(case), flush=True)
            result["cases"].append(case)
        except Exception:  # noqa: BLE001 - preserve failed benchmark evidence.
            case["error"] = traceback.format_exc()
            result["failures"].append(case)
            print(json.dumps(case), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
