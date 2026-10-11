"""Observation-only worker for the pinned vLLM-Ascend 0.22.1rc1 stack.

Adds ranges without changing stream dependencies, workspace or tensor values.
Use only for profiling; compare performance with the ordinary worker separately.
"""
import ast
import contextlib
import functools
import inspect
import json
import os
from pathlib import Path
import statistics
import textwrap
import time

import torch
import torch_npu
from vllm_ascend.worker.worker import NPUWorker

_active = False
_samples = {}
_installed = False
_timer_only = os.getenv("RUNTIME_PROBE_TIMER_ONLY") == "1"


@contextlib.contextmanager
def span(name):
    if not _active:
        yield
        return
    start = time.perf_counter_ns()
    with (contextlib.nullcontext() if _timer_only else torch.profiler.record_function(name)):
        try:
            yield
        finally:
            _samples.setdefault(name, []).append((time.perf_counter_ns() - start) / 1e6)


def wrap(function, name):
    @functools.wraps(function)
    def measured(*args, **kwargs):
        with span(name):
            return function(*args, **kwargs)
    return measured


def instrument_replay(original):
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    targets = {
        "torch.npu.current_stream().synchronize()": "runtime_probe::replay_pre_sync",
        "entry.aclgraph.replay()": "runtime_probe::graph_replay_submit",
    }
    counts = dict.fromkeys(targets, 0)

    class Ranges(ast.NodeTransformer):
        def visit_Expr(self, node):
            label = targets.get(ast.unparse(node.value))
            if label is None:
                return self.generic_visit(node)
            counts[ast.unparse(node.value)] += 1
            return ast.copy_location(ast.With(
                items=[ast.withitem(context_expr=ast.Call(
                    func=ast.Name(id="_runtime_probe_span", ctx=ast.Load()),
                    args=[ast.Constant(label)], keywords=[]))], body=[node]), node)

    tree = Ranges().visit(tree)
    if any(count != 1 for count in counts.values()):
        raise RuntimeError(f"Source changed; refusing instrumentation: {counts}")
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__, _runtime_probe_span=span)
    exec(compile(tree, inspect.getsourcefile(original), "exec"), namespace)
    return namespace[original.__name__]


def install():
    global _installed
    if _installed:
        return
    from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
    from vllm_ascend.worker import model_runner_v1
    ACLGraphWrapper.__call__ = instrument_replay(ACLGraphWrapper.__call__)
    model_runner_v1.update_full_graph_params = wrap(
        model_runner_v1.update_full_graph_params, "runtime_probe::attention_params_update")
    for name in ("_npu_paged_attention_get_workspace", "_npu_paged_attention",
                 "_npu_reshape_and_cache", "_npu_matmul_add_fp32"):
        if hasattr(torch_npu, name):
            setattr(torch_npu, name, wrap(getattr(torch_npu, name), f"runtime_probe::{name}"))
    _installed = True


class RuntimeProbeWorker(NPUWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        install()

    def profile(self, is_start=True, profile_prefix=None):
        global _active
        if is_start:
            _samples.clear()
            if not _timer_only:
                super().profile(is_start, profile_prefix)
            _active = True
        else:
            _active = False
            if not _timer_only:
                super().profile(is_start, profile_prefix)
            output = Path(os.environ["RUNTIME_PROBE_STATS_DIR"])
            output.mkdir(parents=True, exist_ok=True)
            data = {}
            for name, values in _samples.items():
                ordered = sorted(values)
                data[name] = {"count": len(values), "total_ms": sum(values),
                              "p50_ms": statistics.median(values),
                              "p99_ms": ordered[min(len(values)-1, int(len(values)*0.99))],
                              "min_ms": min(values), "max_ms": max(values), "samples_ms": values}
            (output / f"worker_{os.getpid()}.json").write_text(json.dumps({
                "timer_only": _timer_only, "ranges": data}, indent=2))
