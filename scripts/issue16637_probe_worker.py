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
_paired_original_schedule = None
_paired_schedule_class = None
_padding_events = Counter()


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
            _padding_events.clear()
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
            "scheduler_padding_events": dict(_padding_events),
        }

    def issue_probe_scheduler_variant(self, variant, candidate_method_source=None):
        """Experiment only: switch the host method in a single-process executor.

        No source file, graph, tensor or KV policy is changed by this RPC itself.
        The candidate is the schedule method from the archived 16-line patch.
        """
        import ast
        import hashlib
        import os
        import textwrap
        from vllm.v1.core.sched import scheduler as module
        global _paired_original_schedule, _paired_schedule_class
        if _paired_original_schedule is None:
            # Ascend's async patch can replace module.Scheduler with AsyncScheduler.
            # Change the large base method, retaining the async wrapper's super cell.
            _paired_schedule_class = next(
                cls for cls in module.Scheduler.__mro__
                if "schedule" in cls.__dict__
                and cls.__name__ == "Scheduler"
                and inspect.getsourcefile(cls.__dict__["schedule"]).replace("\\", "/").endswith("/vllm/v1/core/sched/scheduler.py")
                and "scheduled_spec_decode_tokens: dict[str, list[int]] = {}" in inspect.getsource(cls.__dict__["schedule"])
            )
            _paired_original_schedule = _paired_schedule_class.__dict__["schedule"]
            if "__class__" in _paired_original_schedule.__code__.co_freevars:
                raise RuntimeError("Refusing to transplant a method with a super closure")
            if "_effective_spec_k" in inspect.getsource(_paired_original_schedule):
                raise RuntimeError("Paired experiment must start from the stock scheduler")
        source = (inspect.getsource(_paired_original_schedule) if variant == "stock"
                  else candidate_method_source)
        if variant not in ("stock", "candidate") or not source:
            raise ValueError("Invalid paired variant")
        source = textwrap.dedent(source)
        tree = ast.parse(source)
        assert len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
        assert tree.body[0].name == "schedule"
        found = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "pad_spec_decode":
                node.body.insert(0, ast.Expr(ast.Call(ast.Name("_paired_note_padding", ast.Load()), [], [])))
                found += 1
        assert found == 1, "Scheduler source changed; refuse instrumentation"
        ast.fix_missing_locations(tree)

        def note_padding():
            _padding_events[variant] += 1

        namespace = dict(_paired_original_schedule.__globals__, _paired_note_padding=note_padding)
        exec(compile(tree, "<paired_scheduler_observation>", "exec"), namespace)
        _paired_schedule_class.schedule = namespace["schedule"]
        return {"variant": variant, "pid": os.getpid(),
                "schedule_owner": _paired_schedule_class.__qualname__,
                "uninstrumented_method_sha256": hashlib.sha256(source.encode()).hexdigest()}
