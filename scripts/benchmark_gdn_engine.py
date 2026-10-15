"""Warm paired ABBA of community GDN layout reproduction on one unchanged engine."""

import argparse
import json
import time
import traceback
from importlib import metadata
from pathlib import Path


def workload(llm, tokenizer, tokens, concurrency, outputs):
    from vllm import SamplingParams

    prompts = []
    for i in range(concurrency):
        unit = tokenizer.encode(
            f"Document {i}: explain efficient inference on Ascend hardware. ",
            add_special_tokens=False,
        )
        ids = (unit * (tokens // len(unit) + 1))[:tokens]
        prompts.append({"prompt_token_ids": ids})
    params = SamplingParams(
        temperature=0, max_tokens=outputs, ignore_eos=True, seed=0, logprobs=5
    )
    start = time.perf_counter()
    results = llm.generate(prompts, params, use_tqdm=False)
    elapsed = time.perf_counter() - start
    records = []
    for result in results:
        completion = result.outputs[0]
        assert len(completion.token_ids) == outputs
        assert len(result.prompt_token_ids) == tokens
        metrics = result.metrics
        records.append(
            {
                "token_ids": completion.token_ids,
                "text": completion.text,
                "ttft_s": None
                if metrics is None
                else float(metrics.first_token_latency),
                "first_logprobs": {
                    str(k): v.logprob for k, v in completion.logprobs[0].items()
                },
            }
        )
    return {
        "input_tokens": tokens,
        "concurrency": concurrency,
        "output_tokens_per_request": outputs,
        "elapsed_s": elapsed,
        "input_tokens_per_s": tokens * concurrency / elapsed,
        "output_tokens_per_s": outputs * concurrency / elapsed,
        "requests": records,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile-dir", type=Path)
    p.add_argument("--graph", action="store_true")
    p.add_argument(
        "--sync-scheduling",
        action="store_true",
        help="Disable async scheduling for fixed-batch correctness control",
    )
    p.add_argument(
        "--uniprocess",
        action="store_true",
        help="Enqueue all requests before execution in the correctness control",
    )
    p.add_argument(
        "--baseline-control",
        action="store_true",
        help="Repeat stock alone to audit generation variability",
    )
    p.add_argument(
        "--no-log-stats",
        action="store_true",
        help="Match the original runs without request metrics",
    )
    p.add_argument(
        "--prefill-only",
        action="store_true",
        help="Fresh-process confirmation at concurrency 1/4/8",
    )
    args = p.parse_args()
    if args.uniprocess:
        import os

        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm import LLM

    kwargs = {
        "model": args.model,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_model_len": 33024,
        "max_num_batched_tokens": 32768,
        "max_num_seqs": 8,
        "gpu_memory_utilization": 0.8,
        "seed": 0,
        "disable_log_stats": args.no_log_stats,
        "async_scheduling": not args.sync_scheduling,
        "enforce_eager": not args.graph,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": True,
        "limit_mm_per_prompt": {"image": 0, "video": 0},
        "worker_cls": "gdn_experiment.GDNExperimentWorker",
    }
    if args.graph:
        kwargs["compilation_config"] = {
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 2, 4, 8],
        }
    if args.profile_dir:
        kwargs["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": str(args.profile_dir),
            "torch_profiler_with_memory": True,
            "torch_profiler_with_stack": False,
        }
    result = {
        "engine_config": kwargs,
        "versions": {
            n: metadata.version(n)
            for n in ["vllm", "vllm-ascend", "torch", "torch-npu", "triton-ascend"]
        },
        "order": [
            "stock",
            "head_first",
            "head_first",
            "stock",
            "head_first",
            "stock",
            "stock",
            "head_first",
        ],
        "runs": [],
    }
    cases = [(2048, 1, 1), (8192, 1, 1), (32768, 1, 1), (8192, 4, 1), (2048, 4, 128)]
    if args.baseline_control:
        result["order"] = ["stock"] * 8
    if args.prefill_only:
        cases = [(8192, 1, 1), (8192, 4, 1), (8192, 8, 1)]
    try:
        llm = LLM(**kwargs)
        tokenizer = llm.get_tokenizer()
        # Both variants see all shapes before measured work. No initializer or
        # compilation time is included in the measured paired blocks.
        for variant in ["stock", "head_first"]:
            llm.collective_rpc("gdn_variant", args=(variant,))
            for tokens, concurrency, output in cases:
                workload(llm, tokenizer, tokens, concurrency, min(output, 8))
        for block, variant in enumerate(result["order"]):
            result["variant_records"] = llm.collective_rpc(
                "gdn_variant", args=(variant,)
            )
            for tokens, concurrency, output in cases:
                row = workload(llm, tokenizer, tokens, concurrency, output)
                row.update(block=block, variant=variant)
                result["runs"].append(row)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                print(
                    json.dumps({k: v for k, v in row.items() if k != "requests"}),
                    flush=True,
                )
        result["memory"] = llm.collective_rpc("gdn_memory")
        if args.profile_dir:
            result["profiled_runs"] = []
            for variant in ["stock", "head_first"]:
                llm.collective_rpc("gdn_variant", args=(variant,))
                llm.start_profile()
                profiled = workload(llm, tokenizer, 8192, 1, 8)
                llm.stop_profile()
                result["profiled_runs"].append({"variant": variant, "result": profiled})
        llm.collective_rpc("gdn_variant", args=("stock",))
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
