"""Pinned single-NPU, continuous-admission MTP graph/padding A/B harness.

Run each variant in a fresh process. Apply the scheduler patch only for
dynamic_patched; the harness verifies that its source hash changed. Profiler
runs are separate from timing runs. Save all output token IDs for comparison.
"""
import argparse
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import statistics
import time
import traceback


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * q))]


def run_workload(llm, prompts, concurrency, max_output, run_id):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind
    engine = llm.llm_engine
    pending = list(range(len(prompts)))
    live, records = {}, {}
    start = time.perf_counter()

    def admit():
        while pending and len(live) < concurrency:
            index = pending.pop(0)
            request_id = f"{run_id}-{index}"
            # Unequal output lengths cause request replacement during decode,
            # exercising new admissions rather than only a synchronized batch.
            output_len = max_output + (index % 4) * (max_output // 8)
            record = {"index": index, "submitted_s": time.perf_counter() - start,
                      "expected_output_tokens": output_len, "token_ids": [], "text": ""}
            live[request_id] = record
            engine.add_request(request_id, {"prompt_token_ids": prompts[index]},
                               SamplingParams(temperature=0, max_tokens=output_len, ignore_eos=True,
                                              output_kind=RequestOutputKind.DELTA, seed=0))

    admit()
    while live:
        for output in engine.step():
            record = live[output.request_id]
            completion = output.outputs[0]
            if completion.token_ids and "first_token_s" not in record:
                record["first_token_s"] = time.perf_counter() - start
            record["token_ids"].extend(completion.token_ids)
            record["text"] += completion.text
            if output.finished:
                record["finished_s"] = time.perf_counter() - start
                if len(record["token_ids"]) != record["expected_output_tokens"]:
                    raise AssertionError(f"Output length mismatch: {record}")
                records[output.request_id] = live.pop(output.request_id)
        admit()
    elapsed = time.perf_counter() - start
    ttft = [(r["first_token_s"] - r["submitted_s"]) * 1000 for r in records.values()]
    # This is per-request average TPOT, not a per-token/chunk latency distribution.
    tpot = [(r["finished_s"] - r["first_token_s"]) * 1000 / (len(r["token_ids"]) - 1)
            for r in records.values()]
    total = sum(len(r["token_ids"]) for r in records.values())
    return {"concurrency": concurrency, "requests": len(records), "elapsed_s": elapsed,
            "output_tokens": total, "output_tokens_per_s": total / elapsed,
            "requests_per_s": len(records) / elapsed,
            "ttft_ms": {"p50": statistics.median(ttft), "p95": percentile(ttft, .95),
                        "p99": percentile(ttft, .99)},
            "request_average_tpot_ms": {"p50": statistics.median(tpot), "p99": percentile(tpot, .99)},
            "request_records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["eager_dynamic", "static_graph", "dynamic_graph", "dynamic_patched"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--input-tokens", type=int, default=2048)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--prefix-cache", action="store_true")
    parser.add_argument("--shared-prompts", action="store_true")
    args = parser.parse_args()
    result = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "versions": {n: metadata.version(n) for n in
                           ["vllm", "vllm-ascend", "torch", "torch-npu", "triton-ascend", "transformers"]},
              "runs": []}
    try:
        import vllm
        from vllm import LLM, SamplingParams
        from vllm.v1.core.sched import scheduler
        if "/src/stack026/vllm/" not in vllm.__file__:
            raise RuntimeError("Pinned vLLM source is shadowed")
        source = Path(scheduler.__file__).read_bytes()
        patched = b"_effective_spec_k" in source
        if patched != (args.variant == "dynamic_patched"):
            raise RuntimeError("Variant does not match the active scheduler patch")
        result["scheduler"] = {"path": scheduler.__file__, "sha256": hashlib.sha256(source).hexdigest(),
                               "patched": patched}
        speculative = {"method": "mtp", "num_speculative_tokens": 3}
        if args.variant != "static_graph":
            speculative["num_speculative_tokens_per_batch_size"] = [[1, 8, 3], [9, 32, 0]]
        kwargs = dict(model=args.model, dtype="bfloat16", tensor_parallel_size=1,
                      max_model_len=4096, max_num_seqs=32, max_num_batched_tokens=8192,
                      gpu_memory_utilization=0.8, seed=0, enable_prefix_caching=args.prefix_cache,
                      limit_mm_per_prompt={"image": 0, "video": 0}, speculative_config=speculative,
                      worker_cls="issue16637_probe_worker.Issue16637ProbeWorker",
                      enforce_eager=args.variant == "eager_dynamic")
        if args.variant != "eager_dynamic":
            kwargs["compilation_config"] = {"cudagraph_mode": "FULL_DECODE_ONLY",
                                            "cudagraph_capture_sizes": [4, 8, 16, 32, 64, 128]}
        if args.profile_dir:
            kwargs["profiler_config"] = {"profiler": "torch", "torch_profiler_dir": str(args.profile_dir),
                                         "torch_profiler_with_memory": True, "torch_profiler_with_stack": False}
        result["engine_config"] = kwargs
        llm = LLM(**kwargs)
        tokenizer = llm.get_tokenizer()
        prompts = []
        for index in range(args.requests):
            text = (f"Document {0 if args.shared_prompts else index}. Explain why matrix multiplication is useful. " * 500)
            ids = tokenizer.encode(text, add_special_tokens=False)[:args.input_tokens]
            assert len(ids) == args.input_tokens
            prompts.append(ids)
        result["workload_sha256"] = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
        result["prompt_token_ids"] = prompts
        llm.generate([{"prompt_token_ids": p[:256]} for p in prompts[:8]],
                     SamplingParams(temperature=0, max_tokens=64, ignore_eos=True), use_tqdm=False)
        for repeat in range(args.repeats):
            for concurrency in args.concurrency:
                llm.collective_rpc("issue_probe_control", kwargs={"active": True, "reset": True})
                if args.profile_dir:
                    llm.start_profile()
                run = run_workload(llm, prompts, concurrency, args.output_tokens, f"r{repeat}-c{concurrency}")
                if args.profile_dir:
                    llm.stop_profile()
                run["worker_observations"] = llm.collective_rpc("issue_probe_snapshot")
                run["repeat"] = repeat
                result["runs"].append(run)
                result["status"] = "running"
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps({k: run[k] for k in ["concurrency", "repeat", "output_tokens_per_s", "ttft_ms"]}), flush=True)
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
