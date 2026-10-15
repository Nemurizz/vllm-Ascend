"""Isolated PR #17604 reproduction and length/tiling screen on an NPU.

Uses an unmodified, pinned PR source file. Forced BLOCK_T variants are experiment
controls, not a proposed production dispatcher. No installed source is patched.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import random
import statistics
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu
from benchmark_wy_operator import event_time


def load_module(path, name, block=None):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if block is not None:
        mod._pick_block_t = lambda *args: block
    return mod


def installed_rearrange():
    import vllm

    path = (
        Path(vllm.__file__).parent
        / "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
    )
    source = path.read_text()
    nodes = [
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.FunctionDef) and n.name == "rearrange_mixed_qkv"
    ]
    assert len(nodes) == 1
    tree = ast.Module(body=nodes, type_ignores=[])
    namespace = {"torch": torch}
    # Execute only the method AST from the pinned, installed vLLM checkout.
    exec(compile(tree, str(path), "exec"), namespace)  # noqa: S102
    return namespace["rearrange_mixed_qkv"], {
        "path": str(path),
        "sha256": hashlib.sha256(source.encode()).hexdigest(),
        "line": nodes[0].lineno,
    }


def cpu_reference(inputs, nk, nv, dim):
    mixed, a, b, alog, bias, index = inputs
    mixed, a, b, alog, bias = [t.float().cpu() for t in (mixed, a, b, alog, bias)]
    if index is not None:
        a, b = a[index.cpu().long()], b[index.cpu().long()]
    t = mixed.shape[0]
    q, k, v = mixed.split([nk * dim, nk * dim, nv * dim], dim=-1)

    def norm(x):
        x = x.reshape(1, t, nk, dim)
        return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6)

    return (
        norm(q),
        norm(k),
        v.reshape(1, t, nv, dim),
        (-alog.exp() * torch.nn.functional.softplus(a + bias)).unsqueeze(0),
        b.sigmoid().unsqueeze(0),
    )


def compare(actual, expected):
    rows = []
    for name, got, want in zip(("q", "k", "v", "g", "beta"), actual, expected):
        got, want = got.float().cpu(), want.float().cpu()
        if name == "v":
            passed = torch.equal(got, want)
        else:
            # BF16 q/k/beta outputs may differ by one rounding unit. Gating
            # accumulates in FP32 and uses a separate tighter tolerance.
            rtol, atol = (2e-5, 2e-6) if name == "g" else (0.016, 0.0005)
            passed = torch.allclose(got, want, rtol=rtol, atol=atol)
        rows.append(
            {
                "tensor": name,
                "pass": passed,
                "bit_equal": torch.equal(got, want),
                "max_abs": (got - want).abs().max().item() if got.numel() else 0,
            }
        )
    assert all(r["pass"] for r in rows), rows
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[1, 32, 128, 512, 2048, 8192, 32768]
    )
    parser.add_argument("--blocks", type=int, nargs="*", default=[1, 4])
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(913)
    rng = random.Random(913)
    from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
    from vllm_ascend.ops.triton.fused_gdn_gating import fused_gdn_gating_patch
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    init_device_properties_triton()
    rearrange, baseline_record = installed_rearrange()
    config = SimpleNamespace(
        key_dim=2048, value_dim=4096, tp_size=1, head_k_dim=128, head_v_dim=128
    )
    mods = {"pr_default": load_module(args.source, "prepare_pr_default")}
    for block in args.blocks:
        mods[f"block_{block}"] = load_module(
            args.source, f"prepare_block_{block}", block
        )
    results = {
        "source": str(args.source),
        "sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "baseline_rearrange": baseline_record,
        "config": vars(config),
        "scope": "prepare component only; event span includes host launch gaps",
        "rounds": args.rounds,
        "repetitions": args.repetitions,
        "correctness": [],
        "cases": [],
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))

    def legacy(mixed, a, b, alog, bias, index):
        q, k, v = rearrange(config, mixed)
        g, beta = fused_gdn_gating_patch(alog, a, b, bias)
        if index is not None:
            g, beta = g.index_select(1, index), beta.index_select(1, index)
        return l2norm_fwd(q), l2norm_fwd(k), v, g, beta

    functions = {"legacy": legacy}
    for name, mod in mods.items():
        functions[name] = lambda mixed, a, b, alog, bias, index, mod=mod: (
            mod.fused_gdn_prepare_impl(
                mixed, a, b, alog, bias, 16, 32, 128, 128, 2048, 4096, 1, index
            )
        )

    def data(length, indexed=False):
        total = length + 19 if indexed else length
        mixed = torch.randn(length, 8192, device="npu", dtype=torch.bfloat16)
        a = torch.randn(total, 32, device="npu", dtype=torch.bfloat16)
        b = torch.randn_like(a)
        alog = torch.randn(32, device="npu", dtype=torch.float32) * 0.1
        bias = torch.randn_like(alog)
        index = (
            torch.randperm(total, device="npu")[:length].to(torch.int32)
            if indexed
            else None
        )
        return mixed, a, b, alog, bias, index

    try:
        # Cover the tier boundary, non-multiple tails, zero-length, and the
        # spec/mixed subset gather before running any timing.
        for length in (0, 1, 32, 64, 65, 128):
            for indexed in (False, True):
                inputs = data(length, indexed)
                reference = cpu_reference(inputs, 16, 32, 128)
                names = list(functions) if length else list(mods)
                for name in names:
                    actual = functions[name](*inputs)
                    checks = compare(actual, reference)
                    results["correctness"].append(
                        {
                            "T": length,
                            "indexed": indexed,
                            "variant": name,
                            "checks": checks,
                        }
                    )
                print("correctness", length, indexed, flush=True)
                save()

        for length in args.lengths:
            inputs = data(length)
            expected = legacy(*inputs)
            checks = {
                name: compare(fn(*inputs), expected) for name, fn in functions.items()
            }
            for _ in range(3):
                for fn in functions.values():
                    fn(*inputs)
            torch.npu.synchronize()
            rows = []
            for repeat in range(args.rounds):
                names = list(functions)
                rng.shuffle(names)
                for name in names:
                    fn = functions[name]
                    event_ms, wall_ms = event_time(
                        lambda fn=fn, inputs=inputs: fn(*inputs), {}, args.repetitions
                    )
                    rows.append(
                        {
                            "round": repeat,
                            "variant": name,
                            "event_ms": event_ms,
                            "wall_ms": wall_ms,
                        }
                    )
            medians = {
                name: {
                    key: statistics.median(r[key] for r in rows if r["variant"] == name)
                    for key in ("event_ms", "wall_ms")
                }
                for name in functions
            }
            results["cases"].append(
                {
                    "T": length,
                    "checks_vs_legacy": checks,
                    "samples": rows,
                    "medians": medians,
                }
            )
            save()
            print("timing", length, medians, flush=True)
            if args.graph:
                graphs, outputs = {}, {}
                for name, fn in functions.items():
                    torch.npu.synchronize()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        outputs[name] = fn(*inputs)
                    graph.replay()
                    torch.npu.synchronize()
                    compare(outputs[name], expected)
                    graphs[name] = graph
                graph_rows = []
                for repeat in range(args.rounds):
                    names = list(graphs)
                    rng.shuffle(names)
                    for name in names:
                        event_ms, wall_ms = event_time(
                            graphs[name].replay, {}, args.repetitions
                        )
                        graph_rows.append(
                            {
                                "round": repeat,
                                "variant": name,
                                "event_ms": event_ms,
                                "wall_ms": wall_ms,
                            }
                        )
                graph_medians = {
                    name: {
                        key: statistics.median(
                            r[key] for r in graph_rows if r["variant"] == name
                        )
                        for key in ("event_ms", "wall_ms")
                    }
                    for name in graphs
                }
                results["cases"][-1]["graph_samples"] = graph_rows
                results["cases"][-1]["graph_medians"] = graph_medians
                # Check actual replay semantics with new values at the captured
                # addresses. Matching a fixed input alone could miss a stale or
                # empty capture. Mutate only after all timing samples.
                inputs[0].mul_(0.9).add_(0.03)
                inputs[1].add_(0.2)
                inputs[2].neg_()
                updated_expected = legacy(*inputs)
                torch.npu.synchronize()
                replay_checks = {}
                for name, graph in graphs.items():
                    graph.replay()
                    torch.npu.synchronize()
                    replay_checks[name] = compare(outputs[name], updated_expected)
                results["cases"][-1]["graph_input_update_checks"] = replay_checks
                save()
                print("graph", length, graph_medians, flush=True)
        if args.profile_dir:
            inputs = data(2048)
            with torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                record_shapes=True,
                profile_memory=True,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                    str(args.profile_dir)
                ),
            ):
                for name, fn in functions.items():
                    with torch.profiler.record_function("prepare_" + name):
                        for _ in range(3):
                            fn(*inputs)
                        torch.npu.synchronize()
            results["profile_dir"] = str(args.profile_dir)
        results["status"] = "complete"
    except Exception:
        results["status"] = "failed"
        results["error"] = traceback.format_exc()
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
