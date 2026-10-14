"""Compare actual long prefills with startup profile on current V1, without a fix.

Use Qwen3.5's native long-context limit. One real request, max_num_seqs 1/256,
same token budget, BF16, eager, chunked prefill disabled. Profiler time is not
a performance baseline. A failure is recorded and is not automatically #15920.
"""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-tokens", type=int, required=True)
    parser.add_argument("--max-num-seqs", type=int, choices=[1, 256], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()
    result = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "versions": {n: metadata.version(n) for n in ["vllm", "vllm-ascend", "torch-npu"]}}
    try:
        from vllm import LLM, SamplingParams
        kwargs = dict(model=args.model, dtype="bfloat16", tensor_parallel_size=1,
                      max_model_len=args.input_tokens + 128,
                      max_num_batched_tokens=args.input_tokens + 128,
                      max_num_seqs=args.max_num_seqs, gpu_memory_utilization=0.8,
                      seed=0, enforce_eager=True, enable_prefix_caching=False,
                      enable_chunked_prefill=False, limit_mm_per_prompt={"image": 0, "video": 0},
                      worker_cls="issue15920_probe_worker.Issue15920ProbeWorker")
        if args.profile_dir:
            kwargs["profiler_config"] = {"profiler": "torch", "torch_profiler_dir": str(args.profile_dir)}
        result["engine_config"] = kwargs
        llm = LLM(**kwargs)
        tokenizer = llm.get_tokenizer()
        unit = tokenizer.encode("This document explains efficient inference on Ascend hardware. ", add_special_tokens=False)
        ids = (unit * (args.input_tokens // len(unit) + 1))[:args.input_tokens]
        result["before_request"] = llm.collective_rpc("issue_long_control")
        if args.profile_dir:
            llm.start_profile()
        start = time.perf_counter()
        outputs = llm.generate([{"prompt_token_ids": ids}],
                               SamplingParams(temperature=0, max_tokens=16, ignore_eos=True), use_tqdm=False)
        result["elapsed_s"] = time.perf_counter() - start
        if args.profile_dir:
            llm.stop_profile()
        result["after_request"] = llm.collective_rpc("issue_long_snapshot")
        result["output_token_ids"] = outputs[0].outputs[0].token_ids
        result["output_text"] = outputs[0].outputs[0].text
        assert len(result["output_token_ids"]) == 16
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
