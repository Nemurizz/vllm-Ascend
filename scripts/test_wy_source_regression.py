# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU regression for a source checkout, using the installed matching runtime."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401


def source_function():
    from vllm_ascend.ops.triton.fla import wy_fast

    # Test the containing checkout without replacing the matching live runtime.
    # When used as a standalone script, fall back to the installed source.
    path = next(
        (
            p / "vllm_ascend/ops/triton/fla/wy_fast.py"
            for p in Path(__file__).resolve().parents
            if (p / "vllm_ascend/ops/triton/fla/wy_fast.py").exists()
        ),
        Path(wy_fast.__file__),
    )
    spec = importlib.util.spec_from_file_location(
        "vllm_ascend.ops.triton.fla.wy_source_regression", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.recompute_w_u_fwd


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("key_dim,value_dim", [(32, 96), (64, 64), (128, 128)])
@pytest.mark.parametrize("head_first", [False, True])
@torch.inference_mode()
def test_batched_wy_identity(dtype, key_dim, value_dim, head_first):
    torch.manual_seed(1901)
    batch, length, key_heads, heads, chunk = 2, 96, 2, 4, 64
    k = torch.randn(batch, length, key_heads, key_dim, device="npu", dtype=dtype)
    v = torch.randn(batch, length, heads, value_dim, device="npu", dtype=dtype)
    batch_values = torch.tensor([0.25, 0.75], device="npu", dtype=torch.float32).view(
        batch, 1, 1
    )
    beta = batch_values.expand(batch, length, heads).contiguous()
    g = batch_values.log().expand_as(beta).contiguous()
    a = torch.zeros(batch, length, heads, chunk, device="npu", dtype=dtype)
    tokens = torch.arange(length, device="npu")
    a[:, tokens, :, tokens % chunk] = 1
    w, u = source_function()(k, v, beta, g, a, head_first_output=head_first)
    torch.npu.synchronize()
    if head_first:
        assert w.is_contiguous() and u.is_contiguous()
        assert tuple(w.shape) == (batch, heads, length, key_dim)
        w, u = w.transpose(1, 2), u.transpose(1, 2)
    expected_w = (
        k.repeat_interleave(heads // key_heads, dim=2).float()
        * beta[..., None]
        * g.exp()[..., None]
    )
    expected_u = v.float() * beta[..., None]
    tolerance = 1e-2 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(w.float(), expected_w, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(u.float(), expected_u, rtol=tolerance, atol=tolerance)
