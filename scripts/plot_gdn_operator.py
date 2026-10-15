"""Export a static figure with ablations and all measured E2E regression points."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    pipeline = json.loads((args.evidence / "gdn-pipeline.json").read_text())["cases"]
    summary = json.loads((args.evidence / "summary.json").read_text())["engine_runs"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    colors = ["#515e6c", "#8e9eae", "#59a5aa", "#177b80"]
    x = np.arange(len(pipeline))
    for n, (name, color) in enumerate(
        zip(["stock", "fixed", "skip_outputs", "head_first"], colors)
    ):
        axes[0].bar(
            x + (n - 1.5) * 0.19,
            [r["median_ms"][name] for r in pipeline],
            0.19,
            label=name,
            color=color,
        )
    axes[0].set(
        xticks=x,
        xticklabels=[f"{r['length'] // 1024}K" for r in pipeline],
        ylabel="Median NPU event span (ms)",
        title="GDN pipeline: paired ablations",
    )
    axes[0].legend(fontsize=8)
    labels = [
        "2K c1 / 1 out",
        "8K c1 / 1 out",
        "32K c1 / 1 out",
        "8K c4 / 1 out",
        "2K c4 / 128 out",
    ]
    width = 0.24
    for index, run in enumerate(summary):
        points = run["cases"]
        if len(points) != 5:
            continue
        offset = (index - (len(summary) - 1) / 2) * width
        label = run["file"].removeprefix("engine-").removesuffix(".json")
        bars = axes[1].bar(
            np.arange(5) + offset,
            [r["throughput_gain_percent"] for r in points],
            width,
            label=label,
        )
        for bar, row in zip(bars, points):
            if row["arms"]["head_first"]["output_differences_vs_first_stock"]:
                bar.set_hatch("///")
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].axhline(
        3, color="#b75c27", linestyle="--", linewidth=1, label="3% project threshold"
    )
    axes[1].set(
        xticks=np.arange(5),
        xticklabels=labels,
        ylabel="Input-token throughput change (%)",
        title="Model E2E: Qwen3.5-4B, 910B2",
    )
    axes[1].tick_params(axis="x", rotation=18, labelsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle(
        "Community PR reproduction on pinned v0.26 / CANN 9.1; four runs per arm",
        fontsize=12,
    )
    fig.text(
        0.01,
        -0.035,
        "Hatched bar: generated-token mismatch; stock-only async graph controls also vary. Not an accuracy acceptance claim.",
        fontsize=8,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    main()
