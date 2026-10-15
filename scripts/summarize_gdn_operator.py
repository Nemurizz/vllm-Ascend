"""Summarize all GDN arms, keeping regression cases and token differences."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def engine_summary(path):
    raw = json.loads(path.read_text())
    groups = defaultdict(lambda: defaultdict(list))
    for row in raw["runs"]:
        groups[
            (row["input_tokens"], row["concurrency"], row["output_tokens_per_request"])
        ][row["variant"]].append(row)
    results = []
    for key, arms in groups.items():
        reference = arms["stock"][0]["requests"]
        item = {
            "input_tokens": key[0],
            "concurrency": key[1],
            "output_tokens_per_request": key[2],
            "arms": {},
        }
        for name, rows in arms.items():
            time_values = [r["elapsed_s"] for r in rows]
            rates = [r["input_tokens_per_s"] for r in rows]
            ttfts = [
                r["ttft_s"]
                for row in rows
                for r in row["requests"]
                if r["ttft_s"] is not None
            ]
            different = []
            logprob_difference = 0.0
            logprob_key_difference = False
            for row in rows:
                for index, req in enumerate(row["requests"]):
                    base = reference[index]
                    if req["token_ids"] != base["token_ids"]:
                        different.append(
                            {"block": row["block"], "request_index": index}
                        )
                    a, b = base["first_logprobs"], req["first_logprobs"]
                    logprob_key_difference |= a.keys() != b.keys()
                    logprob_difference = max(
                        logprob_difference,
                        max(
                            (abs(a[k] - b[k]) for k in a.keys() & b.keys()), default=0.0
                        ),
                    )
            item["arms"][name] = {
                "runs": len(rows),
                "median_elapsed_s": statistics.median(time_values),
                "elapsed_min_s": min(time_values),
                "elapsed_max_s": max(time_values),
                "median_input_tokens_per_s": statistics.median(rates),
                "median_ttft_s": statistics.median(ttfts) if ttfts else None,
                "output_differences_vs_first_stock": different,
                "first_logprob_max_abs_diff": logprob_difference,
                "first_logprob_keys_differ": logprob_key_difference,
            }
        a, b = item["arms"]["stock"], item["arms"]["head_first"]
        item["throughput_gain_percent"] = (
            b["median_input_tokens_per_s"] / a["median_input_tokens_per_s"] - 1
        ) * 100
        item["elapsed_reduction_percent"] = (
            1 - b["median_elapsed_s"] / a["median_elapsed_s"]
        ) * 100
        item["ttft_reduction_percent"] = (
            (1 - b["median_ttft_s"] / a["median_ttft_s"]) * 100
            if a["median_ttft_s"]
            else None
        )
        results.append(item)
    return {
        "file": path.name,
        "status": raw.get("status"),
        "cases": results,
        "sample_unit": "one complete workload run; pooled request TTFT is descriptive only",
        "requests": sum(len(r["requests"]) for r in raw["runs"]),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    output = {
        "engine_runs": [
            engine_summary(path) for path in sorted(args.evidence.glob("engine-*.json"))
        ],
        "limitations": [
            "Four measured runs per arm per process are descriptive, not a statistical confidence guarantee.",
            "Output length one measures prefill/TTFT; do not market it as a general decode-throughput gain.",
            "NPU event spans include gaps caused by host dispatch; native profiler kernel sums are separate.",
        ],
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    for run in output["engine_runs"]:
        for case in run["cases"]:
            ttft = case["ttft_reduction_percent"]
            print(
                run["file"],
                case["input_tokens"],
                case["concurrency"],
                case["output_tokens_per_request"],
                f"throughput {case['throughput_gain_percent']:+.3f}%",
                "TTFT unavailable" if ttft is None else f"TTFT reduction {ttft:+.3f}%",
                "token_differences",
                {
                    n: len(a["output_differences_vs_first_stock"])
                    for n, a in case["arms"].items()
                },
            )


if __name__ == "__main__":
    main()
