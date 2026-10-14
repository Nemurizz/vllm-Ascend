"""Observe real init profiling, masks and memory; do not change dummy shapes."""
from collections import Counter
import os

import torch
import torch_npu
from vllm_ascend.worker.worker import NPUWorker

_mask_calls = Counter()
_dummy_runs = []
_installed = False


def install():
    global _installed
    if _installed:
        return
    from vllm_ascend.attention.attention_mask import AttentionMaskBuilder
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    for name in ("get_attention_mask", "get_attn_mask", "get_splitfuse_attn_mask"):
        if not hasattr(AttentionMaskBuilder, name):
            continue
        original = getattr(AttentionMaskBuilder, name)

        def observed(self, *args, _original=original, _name=name, **kwargs):
            result = _original(self, *args, **kwargs)
            if isinstance(result, torch.Tensor):
                _mask_calls[(_name, tuple(result.shape), str(result.dtype),
                             result.numel() * result.element_size())] += 1
            return result

        setattr(AttentionMaskBuilder, name, observed)
    original_dummy = NPUModelRunner._dummy_run

    def dummy(self, *args, **kwargs):
        import inspect
        bound = inspect.signature(original_dummy).bind(self, *args, **kwargs)
        if bound.arguments.get("is_profile"):
            _dummy_runs.append({"num_tokens": bound.arguments.get("num_tokens"),
                                "is_profile": True,
                                "force_attention": bound.arguments.get("force_attention", False),
                                "max_num_reqs": self.max_num_reqs})
        return original_dummy(self, *args, **kwargs)

    NPUModelRunner._dummy_run = dummy
    _installed = True


class Issue15920ProbeWorker(NPUWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        install()
        self.init_observation = None

    def determine_available_memory(self):
        directory = os.getenv("ISSUE15920_INIT_PROFILE_DIR")
        profiler = None
        if directory:
            profiler = torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU,
                            torch_npu.profiler.ProfilerActivity.NPU],
                record_shapes=True, profile_memory=True, with_stack=False,
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(directory))
            profiler.start()
        try:
            available = super().determine_available_memory()
            torch.npu.synchronize()
            self.init_observation = {
                "available_kv_cache_bytes": available,
                "peak_allocated_bytes": torch.npu.max_memory_allocated(),
                "peak_reserved_bytes": torch.npu.max_memory_reserved(),
                "dummy_runs": list(_dummy_runs),
                "mask_calls": self._mask_snapshot(),
            }
            return available
        finally:
            if profiler:
                profiler.stop()

    @staticmethod
    def _mask_snapshot():
        return [{"method": name, "shape": list(shape), "dtype": dtype, "tensor_bytes": size, "count": n}
                for (name, shape, dtype, size), n in _mask_calls.items()]

    def issue_long_control(self):
        torch.npu.synchronize()
        torch.npu.reset_peak_memory_stats()
        _mask_calls.clear()
        return {"allocated_bytes": torch.npu.memory_allocated(),
                "reserved_bytes": torch.npu.memory_reserved(),
                "init_observation": self.init_observation}

    def issue_long_snapshot(self):
        torch.npu.synchronize()
        return {"peak_allocated_bytes": torch.npu.max_memory_allocated(),
                "peak_reserved_bytes": torch.npu.max_memory_reserved(),
                "mask_calls": self._mask_snapshot(), "init_observation": self.init_observation}
