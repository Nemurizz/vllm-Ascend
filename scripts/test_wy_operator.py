"""NPU WY regression matrix, with an independent per-sequence CPU reference."""

import argparse
import json
import traceback
from pathlib import Path

import torch
from benchmark_wy_operator import load
from build_wy_variants import build


def reference(data, lengths, packed):
    k, v, beta, g, a = [
        data[n].float().cpu() for n in ["k", "v", "beta", "g_cumsum", "A"]
    ]
    batch, total, heads, _dim = v.shape
    key_dim = k.shape[-1]
    group = heads // k.shape[2]
    w = torch.empty(batch, total, heads, key_dim)
    u = torch.empty_like(v)
    start = 0
    for seq, length in enumerate(lengths):
        b, bos = (0, start) if packed else (seq, 0)
        for offset in range(0, length, 64):
            count = min(64, length - offset)
            sl = slice(bos + offset, bos + offset + count)
            for h in range(heads):
                matrix = a[b, sl, h, :count]
                vb = v[b, sl, h] * beta[b, sl, h, None]
                kb = (
                    k[b, sl, h // group]
                    * beta[b, sl, h, None]
                    * g[b, sl, h, None].exp()
                )
                u[b, sl, h] = matrix @ vb
                w[b, sl, h] = matrix @ kb
        start += length
    return w, u


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    variants = build(args.source, args.output.parent / "variants")
    functions = {
        name: load(Path(rec["path"]), name)
        for name, rec in variants.items()
        if not name.startswith("chunk_")
    }
    torch.manual_seed(621)
    cases = []
    for dtype in [torch.bfloat16, torch.float32]:
        for packed, lengths in [
            (True, [1]),
            (True, [65]),
            (True, [31, 0, 97, 1]),
            (False, [96, 96]),
        ]:
            for key_dim, value_dim in [(64, 64), (128, 128), (32, 96)]:
                batch, length = (
                    (1, sum(lengths)) if packed else (len(lengths), lengths[0])
                )
                heads, key_heads = 4, 2
                data = {
                    "k": torch.randn(
                        batch, length, key_heads, key_dim, device="npu", dtype=dtype
                    )
                    * 0.1,
                    "v": torch.randn(
                        batch, length, heads, value_dim, device="npu", dtype=dtype
                    )
                    * 0.1,
                    "beta": torch.rand(
                        batch, length, heads, device="npu", dtype=torch.float32
                    ),
                    "g_cumsum": -torch.rand(
                        batch, length, heads, device="npu", dtype=torch.float32
                    ),
                    "A": torch.randn(
                        batch, length, heads, 64, device="npu", dtype=dtype
                    )
                    * 0.01,
                }
                cu, indices, offset = [0], [], 0
                for seq, count in enumerate(lengths):
                    indices.extend([[seq, i] for i in range((count + 63) // 64)])
                    offset += count
                    cu.append(offset)
                data["cu_seqlens"] = (
                    torch.tensor(cu, device="npu", dtype=torch.int32)
                    if packed
                    else None
                )
                data["chunk_indices"] = (
                    torch.tensor(indices, device="npu", dtype=torch.int32)
                    if packed
                    else None
                )
                expected = reference(data, lengths, packed)
                case = {
                    "dtype": str(dtype),
                    "packed": packed,
                    "lengths": lengths,
                    "key_dim": key_dim,
                    "value_dim": value_dim,
                }
                for variant in ["stock", "fixed", "parallel", "head_first"]:
                    try:
                        out = functions[variant](
                            **data,
                            **(
                                {"head_first_output": True}
                                if variant == "head_first"
                                else {}
                            ),
                        )
                        if variant == "head_first":
                            out = [x.transpose(1, 2) for x in out]
                        torch.npu.synchronize()
                        actual = [x.float().cpu() for x in out]
                        case[variant] = {
                            "max_abs_error": [
                                float((a - b).abs().max())
                                for a, b in zip(actual, expected)
                            ]
                        }
                        for a, b in zip(actual, expected):
                            torch.testing.assert_close(a, b, rtol=2e-2, atol=5e-4)
                        case[variant]["status"] = "passed"
                    except Exception:  # noqa: BLE001 - compile failures are test results.
                        case[variant] = {
                            "status": "failed",
                            "error": traceback.format_exc(),
                        }
                cases.append(case)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps({"variants": variants, "cases": cases}, indent=2) + "\n"
                )
                print(json.dumps(case), flush=True)


if __name__ == "__main__":
    main()
