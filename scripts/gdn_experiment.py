"""Isolated community-PR reproduction modules and an opt-in experiment worker."""

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import torch
from benchmark_wy_operator import load
from build_wy_variants import build


def load_chunk(path, name):
    spec = importlib.util.spec_from_file_location(
        f"vllm_ascend.ops.triton.fla.chunk_experiment_{name}", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def modules(directory):
    from vllm_ascend.ops.triton.fla import chunk, wy_fast

    records = build(Path(wy_fast.__file__), directory)
    fixed = load(Path(records["fixed"]["path"]), "fixed")
    head = load(Path(records["head_first"]["path"]), "head_first")
    skipped = load_chunk(Path(records["chunk_skip_outputs"]["path"]), "skip_outputs")
    optimized = load_chunk(Path(records["chunk_head_first"]["path"]), "head_first")
    skipped.recompute_w_u_fwd = fixed
    optimized.recompute_w_u_fwd = head
    # `fixed` uses stock layout and changes only the community equal-length bug.
    fixed_chunk = load_chunk(Path(chunk.__file__), "fixed")
    fixed_chunk.recompute_w_u_fwd = fixed
    stock = load_chunk(Path(chunk.__file__), "stock")
    return {
        "stock": stock,
        "fixed": fixed_chunk,
        "skip_outputs": skipped,
        "head_first": optimized,
    }, records


def worker_class():
    from vllm_ascend.worker.worker import NPUWorker

    class GDNExperimentWorker(NPUWorker):
        def init_device(self):
            super().init_device()
            from vllm_ascend.ops.triton.fla import chunk

            self._gdn_original_fwd = chunk.chunk_gated_delta_rule_fwd
            self._gdn_variants, self._gdn_records = modules(
                Path("/root/vllm_project/results/gdn-operator/engine-variants")
            )
            self._gdn_variant = "stock"
            self._gdn_calls = Counter()

        def gdn_variant(self, variant):
            from vllm_ascend.ops.triton.fla import chunk

            assert variant in self._gdn_variants
            torch.npu.synchronize()
            target = self._gdn_variants[variant].chunk_gated_delta_rule_fwd

            def observed(*args, **kwargs):
                query = kwargs["q"] if "q" in kwargs else args[0]
                self._gdn_calls[(variant, int(query.shape[1]))] += 1
                return target(*args, **kwargs)

            chunk.chunk_gated_delta_rule_fwd = observed
            self._gdn_variant = variant
            return {"variant": variant, "modules": self._gdn_records}

        def gdn_memory(self):
            torch.npu.synchronize()
            return {
                "variant": self._gdn_variant,
                "allocated": torch.npu.memory_allocated(),
                "peak_allocated": torch.npu.max_memory_allocated(),
                "calls": [
                    {"variant": variant, "tokens": tokens, "count": count}
                    for (variant, tokens), count in self._gdn_calls.items()
                ],
            }

    return GDNExperimentWorker


# Named at module level so the vLLM worker loader can import it. The changes
# exist only in this benchmark worker; installed production files are untouched.
GDNExperimentWorker = worker_class()
