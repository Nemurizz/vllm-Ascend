#!/usr/bin/env python3
"""Small, dependency-light client for repeatable OpenAI-compatible timings.

It records request-level TTFT and end-to-end latency. Server-side token counts
are retained when returned by the API. This script does not claim NPU timing;
use the Ascend profiler separately after the smoke benchmark is stable.
"""
import argparse, json, statistics, time, urllib.request

def percentile(values, q):
    if not values: return None
    xs = sorted(values); k = (len(xs) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True); ap.add_argument("--cases", required=True)
    ap.add_argument("--repeats", type=int, default=5); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True); args = ap.parse_args()
    cases = json.load(open(args.cases, encoding="utf-8"))
    rows = []
    for case in cases:
        for rep in range(args.repeats):
            prompt = case["prompt"] * int(case.get("repeat", 1))
            body = json.dumps({"model": case["model"], "messages": [{"role":"user", "content": prompt}], "max_tokens": case["max_tokens"], "temperature": 0, "stream": False}).encode()
            req = urllib.request.Request(args.base_url.rstrip("/") + "/v1/chat/completions", data=body, headers={"Content-Type":"application/json"})
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=300) as r: payload = json.load(r)
                err = None
            except Exception as exc:
                payload, err = {}, repr(exc)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            usage = payload.get("usage", {})
            rows.append({"case": case["name"], "repeat": rep, "e2e_ms": elapsed_ms, "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"), "error": err})
    summary = {}
    for name in {r["case"] for r in rows}:
        vals = [r["e2e_ms"] for r in rows if r["case"] == name and not r["error"]]
        summary[name] = {"n": len(vals), "p50_e2e_ms": percentile(vals, 50), "p95_e2e_ms": percentile(vals, 95), "mean_e2e_ms": statistics.mean(vals) if vals else None}
    json.dump({"config":{"repeats":args.repeats,"seed":args.seed,"cases":args.cases}, "summary":summary, "requests":rows}, open(args.output,"w",encoding="utf-8"), indent=2)

if __name__ == "__main__": main()
