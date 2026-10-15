"""GDN pipeline correctness, ablations, event timing, and native NPU profiles."""

import argparse
import json
import statistics
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu
from benchmark_wy_operator import event_time
from gdn_experiment import modules


def recurrent_reference(q, k, v, g, beta, state, lengths):
    q, k, v, g, beta, state = [x.float().cpu() for x in [q, k, v, g, beta, state]]
    output = torch.empty_like(v)
    heads = v.shape[2]
    groups = heads // k.shape[2]
    scale = k.shape[-1] ** -0.5
    offset = 0
    for seq, length in enumerate(lengths):
        s = state[seq].clone()
        for pos in range(offset, offset + length):
            key = k[0, pos].repeat_interleave(groups, 0)
            query = q[0, pos].repeat_interleave(groups, 0)
            s = s * g[0, pos].exp()[:, None, None]
            residual = v[0, pos] - torch.einsum("hk,hkv->hv", key, s)
            s = s + torch.einsum("hk,hv->hkv", key, residual * beta[0, pos, :, None])
            output[0, pos] = torch.einsum("hk,hkv->hv", query * scale, s)
        state[seq] = s
        offset += length
    return output, state


def data(lengths, heads=32, key_heads=16, dim=128):
    length = sum(lengths)
    q = torch.randn(1, length, key_heads, dim, device="npu", dtype=torch.bfloat16) * 0.1
    k = torch.randn_like(q) * 0.1
    v = torch.randn(1, length, heads, dim, device="npu", dtype=torch.bfloat16) * 0.1
    g = -torch.rand(1, length, heads, device="npu", dtype=torch.float32) * 0.05
    beta = torch.rand_like(g) * 0.1
    state = (
        torch.randn(len(lengths), heads, dim, dim, device="npu", dtype=torch.float32)
        * 0.01
    )
    cu = [0]
    for length in lengths:
        cu.append(cu[-1] + length)
    # Use the same prebuilt metadata as the model path. Bare chunk calls with
    # None metadata have a separately reported upstream crash (#14192); that
    # optional API path is not a valid baseline for a layout performance A/B.
    from vllm_ascend.ops.gdn_attn_builder import (
        _build_non_spec_chunked_prefill_metadata,
    )

    builder = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(linear_num_value_heads=heads)
            ),
            parallel_config=SimpleNamespace(tensor_parallel_size=1),
        )
    )
    meta = _build_non_spec_chunked_prefill_metadata(
        builder, torch.tensor(cu, dtype=torch.int32), torch.device("npu:0")
    )
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "scale": dim**-0.5,
        "initial_state": state,
        "output_final_state": True,
        "cu_seqlens": torch.tensor(cu, device="npu", dtype=torch.int32),
        "prebuilt_meta": meta,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lengths", type=int, nargs="+", default=[2048, 8192, 32768])
    p.add_argument("--profile-dir", type=Path)
    args = p.parse_args()
    torch.manual_seed(721)
    from vllm_ascend.utils import enable_custom_op

    assert enable_custom_op(), "Pinned A2 custom operators must be available"
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    init_device_properties_triton()
    mods, records = modules(args.output.parent / "variants")
    for mod in mods.values():
        mod.get_pcp_group = lambda: SimpleNamespace(world_size=1)
        mod.get_forward_context = lambda: SimpleNamespace(attn_metadata=None)
    result = {"variants": records, "correctness": [], "cases": []}
    # Packed recurrence reference includes ragged tails. All live inputs/state
    # are cloned by the called kernels, and their preservation is checked below.
    for lengths in [[64], [64, 128]]:
        kwargs = data(lengths, heads=4, key_heads=2, dim=128)
        expected = recurrent_reference(
            *(kwargs[n] for n in ["q", "k", "v", "g", "beta", "initial_state"]), lengths
        )
        state_before = kwargs["initial_state"].clone()
        for name, mod in mods.items():
            actual = mod.chunk_gated_delta_rule_fwd(**kwargs)
            torch.npu.synchronize()
            output, state = actual[1], actual[3]
            errors = [
                float((a.float().cpu() - b).abs().max())
                for a, b in zip([output, state], expected)
            ]
            for a, b in zip([output, state], expected):
                torch.testing.assert_close(a.float().cpu(), b, rtol=3e-2, atol=1e-3)
            torch.testing.assert_close(
                kwargs["initial_state"], state_before, rtol=0, atol=0
            )
            result["correctness"].append(
                {
                    "lengths": lengths,
                    "variant": name,
                    "max_abs_error": errors,
                    "status": "passed",
                }
            )
    for length in args.lengths:
        kwargs = data([length])
        baseline = mods["stock"].chunk_gated_delta_rule_fwd(**kwargs)
        case = {
            "length": length,
            "heads": 32,
            "key_heads": 16,
            "dim": 128,
            "differences": {},
        }
        for name, mod in mods.items():
            output = mod.chunk_gated_delta_rule_fwd(**kwargs)
            torch.npu.synchronize()
            case["differences"][name] = [
                float((baseline[i].float() - output[i].float()).abs().max())
                for i in [1, 3]
            ]
            for i in [1, 3]:
                torch.testing.assert_close(baseline[i], output[i], rtol=0, atol=0)
            for _ in range(2):
                mod.chunk_gated_delta_rule_fwd(**kwargs)
        samples = {name: [] for name in mods}
        for repeat in range(6):
            order = list(mods) if repeat % 2 == 0 else list(reversed(mods))
            for name in order:
                device, wall = event_time(
                    mods[name].chunk_gated_delta_rule_fwd, kwargs, repetitions=3
                )
                samples[name].append({"device_ms": device, "wall_ms": wall})
        case["samples"] = samples
        case["median_ms"] = {
            name: statistics.median(x["device_ms"] for x in rows)
            for name, rows in samples.items()
        }
        result["cases"].append(case)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(case), flush=True)
        if args.profile_dir and length == args.lengths[-1]:
            for name in ["stock", "head_first"]:
                config = torch_npu.profiler._ExperimentalConfig(
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                )
                with torch_npu.profiler.profile(
                    activities=[
                        torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU,
                    ],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=False,
                    experimental_config=config,
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                        str(args.profile_dir / name)
                    ),
                ) as prof:
                    for _ in range(3):
                        with torch.profiler.record_function(f"GDN_{name}"):
                            mods[name].chunk_gated_delta_rule_fwd(**kwargs)
                    torch.npu.synchronize()
                    prof.step()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(traceback.format_exc(), flush=True)
        raise
