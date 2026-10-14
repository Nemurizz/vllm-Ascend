"""Small generation/graph acceptance check before issue performance experiments."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--mtp", action="store_true")
    args = parser.parse_args()
    result = {"args": vars(args) | {"output": str(args.output)},
              "versions": {n: metadata.version(n) for n in
                           ["vllm", "vllm-ascend", "torch", "torch-npu", "triton-ascend", "transformers"]}}
    try:
        import vllm
        import vllm_ascend
        from vllm import LLM, SamplingParams
        result["source_paths"] = {"vllm": vllm.__file__, "vllm_ascend": vllm_ascend.__file__}
        if "/src/stack026/vllm/" not in vllm.__file__:
            raise RuntimeError("Old package shadowed the pinned vLLM source")
        kwargs = dict(model=args.model, dtype="bfloat16", tensor_parallel_size=1,
                      max_model_len=4096, max_num_seqs=8, max_num_batched_tokens=4096,
                      gpu_memory_utilization=0.75, seed=0, enable_prefix_caching=False,
                      enforce_eager=not args.graph)
        if args.graph:
            kwargs["compilation_config"] = {"cudagraph_mode": "FULL_DECODE_ONLY",
                                             "cudagraph_capture_sizes": [4, 8, 16, 32] if args.mtp else [1, 2, 4, 8]}
        if args.mtp:
            kwargs["speculative_config"] = {"method": "mtp", "num_speculative_tokens": 3}
            kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        engine = LLM(**kwargs)
        outputs = engine.generate(["The capital of France is", "Compute 2 + 2:"],
                                  SamplingParams(temperature=0, max_tokens=32, ignore_eos=True))
        result["outputs"] = [{"text": out.outputs[0].text, "token_ids": out.outputs[0].token_ids}
                             for out in outputs]
        assert all(len(out.outputs[0].token_ids) == 32 for out in outputs)
        result["status"] = "passed_generation_length_check"
    except Exception:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
