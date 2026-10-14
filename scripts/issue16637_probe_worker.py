"""Observation-only graph dispatch/replay counters for the pinned 0.26 stack.

Activate after warmup via collective RPC. Counters do not alter tensor values,
padding, graph selection, stream synchronization, or speculative policy.
"""
from collections import Counter
import inspect

import numpy as np
import torch
from vllm_ascend.worker.worker import NPUWorker

_active = False
_dispatch = Counter()
_replays = Counter()
_shapes = Counter()
_installed = False


def install():
    global _installed
    if _installed:
        return
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from vllm_ascend.compilation import acl_graph
    original = NPUModelRunner._determine_batch_execution_and_padding
    signature = inspect.signature(original)

    def determine(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if _active:
            bound = signature.bind(self, *args, **kwargs).arguments
            tokens, reqs = bound["num_tokens"], bound["num_reqs"]
            queries = bound["num_scheduled_tokens_np"][:reqs]
            # Operational denominator: all requests already have computed tokens,
            # and each query is <= max speculative K + 1. Mixed prefill is separate.
            decode = bool(reqs and np.all(self.input_batch.num_computed_tokens_cpu[:reqs] > 0)
                          and np.all(queries <= self.uniform_decode_query_len))
            phase = "decode" if decode else "prefill_or_mixed"
            mode = result[0].name
            _dispatch[(phase, mode)] += 1
            _shapes[(phase, mode, int(reqs), int(tokens), int(result[1].num_tokens),
                     tuple(int(q) for q in queries))] += 1
        return result

    NPUModelRunner._determine_batch_execution_and_padding = determine
    original_call = acl_graph.ACLGraphWrapper.__call__

    def call(self, *args, **kwargs):
        replay = False
        if _active:
            context = acl_graph.get_forward_context()
            if context.cudagraph_runtime_mode == self.runtime_mode:
                entry = self.concrete_aclgraph_entries.get(context.batch_descriptor)
                replay = entry is not None and entry.aclgraph is not None
                draft = bool(acl_graph._EXTRA_CTX.is_draft_model)
        result = original_call(self, *args, **kwargs)
        if replay:
            _replays[("draft" if draft else "target", self.runtime_mode.name)] += 1
        return result

    acl_graph.ACLGraphWrapper.__call__ = call
    _installed = True


class Issue16637ProbeWorker(NPUWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        install()

    def issue_probe_control(self, active=True, reset=True):
        global _active
        if reset:
            _dispatch.clear()
            _replays.clear()
            _shapes.clear()
            torch.npu.reset_peak_memory_stats()
        _active = active
        return {"active": _active}

    def issue_probe_snapshot(self):
        return {
            "dispatch": [{"phase": p, "mode": m, "count": n} for (p, m), n in _dispatch.items()],
            "successful_replays": [{"model": model, "mode": m, "count": n}
                                   for (model, m), n in _replays.items()],
            "query_shapes": [{"phase": p, "mode": m, "requests": reqs,
                              "tokens": tokens, "padded_tokens": padded, "queries": list(queries), "count": n}
                             for (p, m, reqs, tokens, padded, queries), n in _shapes.items()],
            "peak_allocated_bytes": torch.npu.max_memory_allocated(),
            "peak_reserved_bytes": torch.npu.max_memory_reserved(),
        }
