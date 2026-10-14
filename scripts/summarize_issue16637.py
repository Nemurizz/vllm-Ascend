"""Summarize measured throughput, decode dispatch fractions and patch equality."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = {"static_graph": "static-graph", "dynamic_graph": "dynamic_graph",
             "eager_dynamic": "eager_dynamic", "dynamic_patched": "dynamic_patched"}
    data = {name: json.loads((args.input_dir / f"issue16637-{file}.json").read_text())
            for name, file in names.items()}
    assert all(d["status"] == "completed" for d in data.values())
    assert len({d["workload_sha256"] for d in data.values()}) == 1
    rows = []
    for variant, document in data.items():
        for concurrency in sorted({r["concurrency"] for r in document["runs"]}):
            runs = [r for r in document["runs"] if r["concurrency"] == concurrency]
            counts = defaultdict(int)
            for run in runs:
                for worker in run["worker_observations"]:
                    for dispatch in worker["dispatch"]:
                        if dispatch["phase"] == "decode":
                            counts[dispatch["mode"]] += dispatch["count"]
            total = sum(counts.values())
            samples = [r["output_tokens_per_s"] for r in runs]
            rows.append({"variant": variant, "concurrency": concurrency, "runs": len(runs),
                         "median_output_tokens_per_s": statistics.median(samples),
                         "min_output_tokens_per_s": min(samples), "max_output_tokens_per_s": max(samples),
                         "decode_dispatches": total, "full_fraction": counts["FULL"] / total,
                         "piecewise_fraction": counts["PIECEWISE"] / total, "none_fraction": counts["NONE"] / total,
                         "median_ttft_p50_ms": statistics.median(r["ttft_ms"]["p50"] for r in runs),
                         "median_request_average_tpot_p50_ms": statistics.median(r["request_average_tpot_ms"]["p50"] for r in runs)})
    original = {(r["repeat"], r["concurrency"]): r for r in data["dynamic_graph"]["runs"]}
    matched, compared, mismatches = 0, 0, []
    for patched in data["dynamic_patched"]["runs"]:
        baseline = original[(patched["repeat"], patched["concurrency"])]
        for request_id, record in patched["request_records"].items():
            other = baseline["request_records"][request_id]
            compared += 1
            if record["token_ids"] == other["token_ids"]:
                matched += 1
            else:
                mismatches.append({"repeat": patched["repeat"], "concurrency": patched["concurrency"], "request_id": request_id})
    changes = []
    for row in [r for r in rows if r["variant"] == "dynamic_patched"]:
        baseline = next(r for r in rows if r["variant"] == "dynamic_graph" and r["concurrency"] == row["concurrency"])
        changes.append({"concurrency": row["concurrency"],
                        "median_throughput_change_percent": (row["median_output_tokens_per_s"] / baseline["median_output_tokens_per_s"] - 1) * 100})
    result = {"workload_sha256": data["static_graph"]["workload_sha256"], "rows": rows,
              "patch_vs_dynamic": changes,
              "patch_output_equality": {"matched_requests": matched, "compared_requests": compared, "mismatches": mismatches},
              "limitations": ["graph fractions are measured runtime dispatches; successful replays are separate raw counters",
                              "three sequential repetitions do not establish a small statistically reliable gain",
                              "TPOT is each request's average, not a per-token distribution"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
