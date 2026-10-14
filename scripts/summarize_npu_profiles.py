"""Summarize CANN 9.1 torch_npu exports without equating task sums to wall time."""
import argparse
from collections import defaultdict
import csv
from decimal import Decimal
import json
from pathlib import Path


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        yield from csv.DictReader(stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for folder in sorted(args.profile_root.rglob("ASCEND_PROFILER_OUTPUT")):
        kernel_file = folder / "kernel_details.csv"
        if not kernel_file.exists():
            continue
        intervals, types, attention = [], defaultdict(lambda: [0, 0.0]), defaultdict(lambda: [0, 0.0])
        for row in read_rows(kernel_file):
            duration = float(row["Duration(us)"])
            start = int(Decimal(row["Start Time(us)"].strip()) * 1000)
            end = start + round(duration * 1000)
            intervals.append((start, end))
            types[row["Type"]][0] += 1
            types[row["Type"]][1] += duration
            if "attention" in (row["Name"] + row["Type"]).lower():
                attention[row["Type"]][0] += 1
                attention[row["Type"]][1] += duration
        intervals.sort()
        union_ns = 0
        if intervals:
            begin, end = intervals[0]
            for next_begin, next_end in intervals[1:]:
                if next_begin > end:
                    union_ns += end - begin
                    begin, end = next_begin, next_end
                else:
                    end = max(end, next_end)
            union_ns += end - begin
        span_ns = max(e for _, e in intervals) - intervals[0][0] if intervals else 0
        apis = []
        api_file = folder / "api_statistic.csv"
        if api_file.exists():
            for row in read_rows(api_file):
                apis.append({"name": row["API Name"], "level": row["Level"],
                             "count": int(row["Count"]), "total_us": float(row["Time(us)"])})
        results.append({
            "path": str(folder.relative_to(args.profile_root)), "kernel_tasks": len(intervals),
            "summed_kernel_duration_us": sum(t[1] for t in types.values()),
            "kernel_interval_union_us": union_ns / 1000, "first_to_last_kernel_span_us": span_ns / 1000,
            "kernel_union_fraction_within_span": union_ns / span_ns if span_ns else None,
            "top_types": [{"type": name, "count": value[0], "duration_us": value[1]}
                          for name, value in sorted(types.items(), key=lambda item: item[1][1], reverse=True)[:15]],
            "attention_named_device_tasks": [{"type": name, "count": value[0], "duration_us": value[1]}
                                             for name, value in attention.items()],
            "top_host_apis": sorted(apis, key=lambda row: row["total_us"], reverse=True)[:15],
            "pipe_metrics_available": "aic_mac_ratio" in next(read_rows(kernel_file), {}),
        })
    result = {"profiles": results,
              "limitations": ["task-duration sums can overlap; kernel union is bounded by first/last device task",
                              "kernel union fraction is not overall NPU utilization or proof of a CPU bottleneck",
                              "attention-name matching is not exhaustive; memory profiling can contain incomplete records"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Summarized {len(results)} profiles")


if __name__ == "__main__":
    main()
