#!/usr/bin/env python3
"""Concurrent OpenAI-compatible benchmark with TTFT and E2E measurements."""

import argparse
import concurrent.futures
import json
import random
import statistics
import time
import urllib.request


def percentile(values, q):
    if not values:
        return None
    xs = sorted(values)
    k = (len(xs) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def request_once(base_url, case, repeat, request_id):
    prompt = case["prompt"] * int(case.get("repeat", 1))
    body = json.dumps(
        {
            "model": case.get("model", "Qwen/Qwen3-8B"),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": case["max_tokens"],
            "temperature": 0,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    first_event = None
    usage = {}
    text_parts = []
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
                event = json.loads(payload)
                if first_event is None:
                    first_event = time.perf_counter()
                usage.update(event.get("usage") or {})
                for choice in event.get("choices", []):
                    text_parts.append(choice.get("delta", {}).get("content") or "")
    except Exception as exc:  # preserve failures in the raw result
        error = repr(exc)
    done = time.perf_counter()
    return {
        "request_id": request_id,
        "case": case["name"],
        "repeat": repeat,
        "ttft_ms": None if first_event is None else (first_event - t0) * 1000,
        "e2e_ms": (done - t0) * 1000,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "text_chars": len("".join(text_parts)),
        "error": error,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cases = json.load(open(args.cases, encoding="utf-8"))
    jobs = [(case, rep) for rep in range(args.repeats) for case in cases]
    random.Random(args.seed).shuffle(jobs)
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(request_once, args.base_url, case, rep, idx)
            for idx, (case, rep) in enumerate(jobs)
        ]
        rows = [f.result() for f in futures]
    elapsed = time.perf_counter() - t0

    summary = {}
    for name in sorted({row["case"] for row in rows}):
        good = [row for row in rows if row["case"] == name and not row["error"]]
        case_rows = [row for row in rows if row["case"] == name]
        ttft = [row["ttft_ms"] for row in good if row["ttft_ms"] is not None]
        e2e = [row["e2e_ms"] for row in good]
        summary[name] = {
            "n": len(good),
            "errors": len(case_rows) - len(good),
            "ttft_ms": {f"p{q}": percentile(ttft, q) for q in (50, 95, 99)},
            "e2e_ms": {f"p{q}": percentile(e2e, q) for q in (50, 95, 99)},
        }
    result = {
        "config": vars(args),
        "wall_time_s": elapsed,
        "request_per_s": len(rows) / elapsed if elapsed else None,
        "summary": summary,
        "requests": sorted(rows, key=lambda row: row["request_id"]),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
