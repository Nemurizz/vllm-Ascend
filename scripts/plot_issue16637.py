"""Plot cold block medians/ranges and warm ABBA results; not a speedup claim."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    summary = json.loads(args.summary.read_text())
    paired = json.loads(args.paired.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1.6, 1]})
    styles = [("static_graph", "Static MTP / FULL", "#2864A4"),
              ("dynamic_graph", "Dynamic MTP / PIECEWISE", "#D9633D"),
              ("eager_dynamic", "Dynamic MTP / eager", "#8A939D"),
              ("dynamic_patched", "Padding candidate / PIECEWISE", "#8269AD")]
    x = np.arange(3)
    for index, (variant, label, color) in enumerate(styles):
        rows = [next(r for r in summary["rows"] if r["variant"] == variant and r["concurrency"] == c)
                for c in [4, 8, 16]]
        y = np.array([r["median_output_tokens_per_s"] for r in rows])
        error = np.array([[y[i] - r["min_output_tokens_per_s"] for i, r in enumerate(rows)],
                          [r["max_output_tokens_per_s"] - y[i] for i, r in enumerate(rows)]])
        axes[0].bar(x + (index - 1.5) * .19, y, width=.18, yerr=error, capsize=2, label=label, color=color)
    axes[0].set(xticks=x, xticklabels=["c4", "c8", "c16"], ylabel="Output tokens / second",
                title="Sequential groups: median, min/max of 3 runs")
    axes[0].legend(fontsize=7.5, loc="upper left")
    values = [r["change_percent"] for r in paired["comparisons"]]
    axes[1].bar(x, values, color=["#AD955E" if v > 0 else "#D9633D" for v in values], width=.6)
    axes[1].axhline(0, color="#333333", linewidth=.8)
    axes[1].axhline(3, color="#8A939D", linestyle="--", linewidth=.8)
    axes[1].set(xticks=x, xticklabels=["c4", "c8", "c16"], ylabel="Candidate throughput change (%)",
                title="Warm ABBA: 2 runs per variant")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.15)
        axis.set_axisbelow(True)
    fig.suptitle("Ascend 910B2 x1 | Qwen3.5-4B BF16 | vLLM 0.26 / CANN 9.1", fontsize=12)
    fig.text(.5, .01, "Measured padding events: 0. FULL was not restored. A stable improvement is not established.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .055, 1, .94])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)


if __name__ == "__main__":
    main()
