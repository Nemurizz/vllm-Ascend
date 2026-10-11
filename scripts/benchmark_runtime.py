"""Fixed-token completion benchmark; preserve generated text and server usage."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import time
import urllib.request

from benchmark_concurrent import percentile


def call(base, prompt, tokens, index, timeout=60):
    payload = {"model": "Qwen3-8B", "prompt": prompt, "max_tokens": tokens,
               "temperature": 0, "seed": 0, "ignore_eos": True, "stream": True,
               "stream_options": {"include_usage": True}}
    request = urllib.request.Request(base + "/v1/completions", data=json.dumps(payload).encode(),
                                    headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first = None
    text = []
    usage = {}
    error = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[6:])
                usage.update(event.get("usage") or {})
                for choice in event.get("choices", []):
                    chunk = choice.get("text", "")
                    if chunk and first is None:
                        first = time.perf_counter()
                    text.append(chunk)
    except Exception as exc:
        error = repr(exc)
    end = time.perf_counter()
    count = usage.get("completion_tokens", 0)
    return {"index": index, "ttft_ms": (first-start)*1000 if first else None,
            "e2e_ms": (end-start)*1000,
            "tpot_ms": (end-first)*1000/(count-1) if first and count>1 else None,
            "usage": usage, "text": "".join(text), "error": error}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8020")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--prompt-repeat", type=int, default=1)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prompt = "Explain the main benefits and limitations of a cache in an inference server. " * args.prompt_repeat
    # Warmup is outside the recorded profile and timings.
    warmup = call(args.base_url, prompt, args.warmup_tokens, -1, args.timeout)
    if warmup["error"]:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps({"config": vars(args), "warmup": warmup,
                                                "status": "warmup_failed"}, indent=2))
        raise RuntimeError(warmup["error"])
    if args.profile:
        urllib.request.urlopen(urllib.request.Request(args.base_url+"/start_profile", data=b""), timeout=60).close()
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        rows = list(pool.map(lambda i: call(args.base_url, prompt, args.tokens, i, args.timeout), range(args.requests)))
    elapsed = time.perf_counter()-start
    good = [r for r in rows if r["error"] is None]
    summary = {}
    for field in ("ttft_ms", "e2e_ms", "tpot_ms"):
        values = [r[field] for r in good if r[field] is not None]
        summary[field] = ({f"p{q}": percentile(values, q) for q in (50, 95, 99)}
                          if values else None)
    result = {"config": vars(args), "warmup": warmup, "wall_time_s": elapsed, "errors": len(rows)-len(good),
              "output_token_per_s": sum(r["usage"].get("completion_tokens",0) for r in good)/elapsed,
              "summary": summary, "requests": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    if args.profile:
        try:
            urllib.request.urlopen(urllib.request.Request(args.base_url+"/stop_profile", data=b""), timeout=120).close()
            result["profile_stop"] = "ok"
        except Exception as exc:
            result["profile_stop"] = repr(exc)
        Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({k:v for k,v in result.items() if k!="requests"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
