"""ABBA check in one warm engine to distinguish a small gain from block drift.

The single-process NPU executor shares a process with the host scheduler. Swap
only Scheduler.schedule via an explicit experiment RPC. Count real padding
events. Production files and graph reliability guards remain unchanged.
"""
import argparse
import ast
import json
from pathlib import Path
import statistics
import subprocess
import textwrap
import traceback

from benchmark_issue16637 import run_workload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate-ref", default="codex/issue16637-026-experiment")
    parser.add_argument("--source", type=Path, default=Path("/root/vllm_project/src/stack026/vllm"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    source = subprocess.check_output(["git", "-C", str(args.source), "show",
                                      f"{args.candidate_ref}:vllm/v1/core/sched/scheduler.py"], text=True)
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "schedule")
    candidate = textwrap.dedent(ast.get_source_segment(source, method))
    result = {"baseline_file": str(args.baseline), "workload_sha256": baseline["workload_sha256"],
              "candidate_ref": args.candidate_ref, "runs": []}
    try:
        from vllm import LLM, SamplingParams
        llm = LLM(**baseline["engine_config"])
        executor = llm.llm_engine.vllm_config.parallel_config.distributed_executor_backend
        if executor != "uni":
            raise RuntimeError(f"Shared scheduler RPC requires uni executor, got {executor}")
        result["executor"] = executor
        prompts = baseline["prompt_token_ids"]
        llm.generate([{"prompt_token_ids": p[:256]} for p in prompts[:8]],
                     SamplingParams(temperature=0, max_tokens=64, ignore_eos=True), use_tqdm=False)
        for concurrency in [4, 8, 16]:
            for order, variant in enumerate(["stock", "candidate", "candidate", "stock"]):
                switched = llm.collective_rpc("issue_probe_scheduler_variant", kwargs={
                    "variant": variant, "candidate_method_source": candidate})
                llm.collective_rpc("issue_probe_control", kwargs={"active": True, "reset": True})
                run = run_workload(llm, prompts, concurrency, baseline["args"]["output_tokens"], f"c{concurrency}-o{order}")
                run.update(variant=variant, order=order, switched=switched)
                run["observations"] = llm.collective_rpc("issue_probe_snapshot")
                result["runs"].append(run)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2))
                print(json.dumps({k: run[k] for k in ["concurrency", "variant", "order", "output_tokens_per_s"]}), flush=True)
        result["comparisons"] = []
        for concurrency in [4, 8, 16]:
            runs = [r for r in result["runs"] if r["concurrency"] == concurrency]
            means = {v: statistics.mean(r["output_tokens_per_s"] for r in runs if r["variant"] == v)
                     for v in ["stock", "candidate"]}
            reference = {r["index"]: r["token_ids"] for r in runs[0]["request_records"].values()}
            matched = all({r["index"]: r["token_ids"] for r in run["request_records"].values()} == reference for run in runs)
            result["comparisons"].append({"concurrency": concurrency, "mean_tokens_per_s": means,
                                          "change_percent": (means["candidate"] / means["stock"] - 1) * 100,
                                          "all_outputs_match": matched,
                                          "padding_events": sum(sum(w["scheduler_padding_events"].values())
                                                                for r in runs for w in r["observations"])})
        result["status"] = "completed"
    except Exception:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
