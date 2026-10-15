"""Build isolated WY experiments from the pinned installed source; never edit it."""

import argparse
import hashlib
from pathlib import Path


def build(source, directory):
    directory.mkdir(parents=True, exist_ok=True)
    original = source.read_text()
    # Equal-length repair corresponds to community PR #14333. Keep it separate
    # from the geometry experiment; it does not affect the packed production path.
    fixed = (
        original.replace(
            "for i_bh in range(H):\n        i_b, i_h = i_bh // H, i_bh % H",
            "for i_h in range(H):",
        )
        .replace(
            "        else:\n            bos, eos = i_b * T, i_b * T + T",
            "            bos_bh = bos\n        else:\n"
            "            i_b = tl.program_id(1)\n            i_t = i_t_o\n"
            "            bos, eos = i_b * T, i_b * T + T\n            bos_bh = i_b * H * T_max",
        )
        .replace("g + bos + i_h", "g + bos_bh + i_h")
        .replace("beta + bos + i_h", "beta + bos_bh + i_h")
    )
    assert "bos_bh = i_b * H * T_max" in fixed
    start = fixed.index("    for i_h in range(H):\n")
    end = fixed.index("\n\ndef recompute_w_u_fwd(", start)
    body = fixed[start:end].splitlines()[1:]
    parallel = (
        fixed[:start]
        + "    i_bh = tl.program_id(1)\n    i_h = i_bh % H\n"
        + "\n".join(line.removeprefix("    ") for line in body)
        + fixed[end:]
    )
    parallel = parallel.replace("i_b = tl.program_id(1)", "i_b = i_bh // H")
    parallel = parallel.replace(
        "recompute_w_u_fwd_kernel[(NT, B)]", "recompute_w_u_fwd_kernel[(NT, B * H)]"
    )
    assert "for i_h in range(H)" not in parallel
    # Community PR #16538: emit the layout consumed by the existing AscendC
    # h/o operators. Combine with #14333 so batched equal-length addressing is
    # actually valid. This is explicitly a reproduction, not a novel algorithm.
    head_first = fixed.replace(
        "    IS_VARLEN: tl.constexpr,",
        "    IS_VARLEN: tl.constexpr,\n    HEAD_FIRST_OUTPUT: tl.constexpr,",
    )
    for tensor, width, offset in [("u", "V", "offs_v"), ("w", "K", "offs_k")]:
        old = f"            ptr_{tensor} = {tensor} + (bos * H + i_h) * {width} + offs_t_2d * (H * {width}) + {offset} * 1"
        new = (
            f"            if HEAD_FIRST_OUTPUT:\n"
            f"                ptr_{tensor} = {tensor} + (bos_bh + i_h * T_max + global_offs_t[:, None]) * {width} + {offset}\n"
            f"            else:\n    " + old
        )
        assert old in head_first
        head_first = head_first.replace(old, new)
    head_first = head_first.replace(
        "    chunk_indices: torch.Tensor | None = None,",
        "    chunk_indices: torch.Tensor | None = None,\n    head_first_output: bool = False,",
    )
    head_first = head_first.replace(
        "    u = torch.empty_like(v)\n    w = k.new_empty(B, T, H, K)",
        "    if head_first_output:\n        u = v.new_empty(B, H, T, V)\n        w = k.new_empty(B, H, T, K)\n    else:\n        u = torch.empty_like(v)\n        w = k.new_empty(B, T, H, K)",
    )
    head_first = head_first.replace(
        "        BV=BV,", "        BV=BV,\n        HEAD_FIRST_OUTPUT=head_first_output,"
    )
    records = {}
    for name, text in [
        ("stock", original),
        ("fixed", fixed),
        ("parallel", parallel),
        ("head_first", head_first),
    ]:
        path = directory / f"wy_{name}.py"
        path.write_text(text)
        records[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
    chunk = source.with_name("chunk.py").read_text()
    conversions = "    v_new = v_new.to(torch.bfloat16).transpose(1, 2).contiguous()\n    h = h.to(torch.bfloat16).transpose(1, 2).contiguous()\n"
    assert conversions in chunk
    skipped = chunk.replace(conversions, "").replace(
        "    elif SUPPRESS_LEVEL >= 3:\n",
        "    elif SUPPRESS_LEVEL >= 3:\n"
        + "\n".join("    " + line for line in conversions.splitlines())
        + "\n",
    )
    integrated = skipped.replace(
        "    w, u = recompute_w_u_fwd(\n",
        "    head_first_output = get_pcp_group().world_size == 1 and SUPPRESS_LEVEL < 3\n    w, u = recompute_w_u_fwd(\n",
    )
    # Anchor only the WY call, not earlier KKT/solve calls that share metadata.
    wy_start = integrated.index("    w, u = recompute_w_u_fwd(\n")
    wy_end = integrated.index("\n    )", wy_start)
    integrated = (
        integrated[:wy_end]
        + "\n        head_first_output=head_first_output,"
        + integrated[wy_end:]
    )
    for tensor in ["w", "u"]:
        old = f"    {tensor}_ascendc = {tensor}.to(torch.bfloat16).transpose(1, 2).contiguous()"
        new = f"    {tensor}_ascendc = {tensor}.to(torch.bfloat16) if head_first_output else {tensor}.to(torch.bfloat16).transpose(1, 2).contiguous()"
        assert old in integrated
        integrated = integrated.replace(old, new)
    for name, text in [("skip_outputs", skipped), ("head_first", integrated)]:
        path = directory / f"chunk_{name}.py"
        path.write_text(text)
        records["chunk_" + name] = {
            "path": str(path),
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
    return records


if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output_dir), indent=2))
