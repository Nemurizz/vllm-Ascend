#!/usr/bin/env python3
"""Send a long request, then a short request, to expose scheduler head-of-line behavior."""

import argparse
import concurrent.futures
import json
import time
import urllib.request


def stream_request(base_url, case, submit_delay):
    if submit_delay:
        time.sleep(submit_delay)
    body = json.dumps({
        "model": "Qwen3-8B",
        "messages": [{"role": "user", "content": case["prompt"]}],
        "max_tokens": case["max_tokens"],
        "temperature": 0,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t_submit = time.perf_counter()
    first = None
    done = None
    error = None
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                if first is None:
                    first = time.perf_counter()
        done = time.perf_counter()
    except Exception as exc:
        error = repr(exc)
        done = time.perf_counter()
    return {
        "name": case["name"],
        "submit_delay_s": submit_delay,
        "submit_to_first_ms": None if first is None else (first - t_submit) * 1000,
        "submit_to_done_ms": (done - t_submit) * 1000,
        "error": error,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--delay-s", type=float, default=0.10)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    long_case = {"name": "long_first", "prompt": "背景材料。" * 4096, "max_tokens": 32}
    short_case = {"name": "short_second", "prompt": "用一句话回答：什么是缓存？", "max_tokens": 32}
    rows = []
    for rep in range(args.repeats):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f_long = pool.submit(stream_request, args.base_url, long_case, 0.0)
            f_short = pool.submit(stream_request, args.base_url, short_case, args.delay_s)
            long_row = f_long.result()
            short_row = f_short.result()
        long_row["repeat"] = rep
        short_row["repeat"] = rep
        rows.extend([long_row, short_row])
    result = {"config": vars(args), "rows": rows}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
