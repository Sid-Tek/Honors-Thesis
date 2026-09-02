"""
Plots the fixed-window/variable-position sweep from position_sweep.py:
one small subplot per n-gram category (+ combined), x-axis = starting
offset, one colored line per window size tested.

A flat line within a subplot = that category, at that size, doesn't care
where in the book you sampled from -- position-invariant, the signature
of a fast-acting feature. A line that climbs with offset even at a fixed
size = genuine narrative-time emergence, not just a sample-size artifact.
Flat lines (per flatness_test) are marked with a star at their last point.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ngram_categories import CATEGORIES
from position_sweep import flatness_test

ALL_LINES = CATEGORIES + ["combined"]


def plot_position_sweep(results_by_size, offsets, alpha=0.05,
                         title="Accuracy vs. starting position, by window size"):
    """
    results_by_size: dict {window_size: {line: [accuracy per offset]}} --
                      exactly what run_position_sweep_multi_size() returns.
    offsets: same offsets list passed into the sweep.
    alpha: significance threshold used to decide which lines get a
           flatness star (same meaning as position_sweep.flatness_test).

    Returns a matplotlib Figure with one subplot per line, in a grid.
    """
    window_sizes = sorted(results_by_size.keys())
    lines = ALL_LINES

    n_cols = 3
    n_rows = int(np.ceil(len(lines) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.2 * n_rows),
                              sharey=True, squeeze=False)
    axes = axes.reshape(-1)

    cmap = plt.get_cmap("viridis")
    handles_for_legend, labels_for_legend = None, None

    for idx, line in enumerate(lines):
        ax = axes[idx]
        for size_idx, w in enumerate(window_sizes):
            accuracies = results_by_size[w][line]
            color = cmap(size_idx / max(1, len(window_sizes) - 1))
            ax.plot(offsets, accuracies, marker="o", markersize=3,
                    color=color, linewidth=1.4, label=f"{w} chars")

            ft = flatness_test(offsets, accuracies)
            if ft is not None and ft["p_value"] >= alpha:
                valid = [(o, a) for o, a in zip(offsets, accuracies)
                         if a is not None and not np.isnan(a)]
                if valid:
                    ax.scatter([valid[-1][0]], [valid[-1][1]], color=color,
                               marker="*", s=140, edgecolor="black",
                               linewidth=0.6, zorder=5)

        if idx == 0:
            handles_for_legend, labels_for_legend = ax.get_legend_handles_labels()

        ax.set_title(line, fontsize=10)
        ax.set_xlabel("Offset (chars)", fontsize=8)
        if idx % n_cols == 0:
            ax.set_ylabel("Accuracy", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.tick_params(labelsize=7)

    for j in range(len(lines), len(axes)):
        axes[j].axis("off")

    if handles_for_legend:
        fig.legend(handles_for_legend, labels_for_legend, loc="upper center",
                   ncol=min(len(window_sizes), 8), bbox_to_anchor=(0.5, 1.04),
                   fontsize=8, title="Window size (\u2605 = position-flat)")
    fig.suptitle(title, y=1.09, fontsize=12)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # Smoke test against the same synthetic corpus used in position_sweep.py.
    from position_sweep import run_position_sweep_multi_size

    rng = np.random.default_rng(0)

    def make_novel(style, length=6000):
        chunks = {
            "A": ["the cat sat, quietly", "and then it ran", "she said, 'no'"],
            "B": ["birds fly-high above", "he ran-fast today", "we saw-it clearly"],
        }[style]
        text = ""
        while len(text) < length:
            text += rng.choice(chunks) + " "
        return text[:length]

    documents, labels = [], []
    for author in ["A", "B"]:
        for _ in range(6):
            documents.append(make_novel(author))
            labels.append(author)

    window_sizes = [200, 500]
    offsets = [0, 1000, 2000, 3000]

    results_by_size = run_position_sweep_multi_size(
        documents, labels, window_sizes, offsets,
        n=3, min_count=2, n_folds=3, seed=0, verbose=False,
    )

    fig = plot_position_sweep(results_by_size, offsets)
    fig.savefig("/tmp/position_sweep_test.png", dpi=100, bbox_inches="tight")
    print("Saved test plot to /tmp/position_sweep_test.png")