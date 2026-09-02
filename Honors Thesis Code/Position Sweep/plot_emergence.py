"""
Plots the 11 emergence curves (10 n-gram categories + combined) produced
by window_sweep.py's run_window_sweep(), marks each line's plateau point
(using emergence_plateau.py's detect_plateau_slope), and draws a
horizontal chance-level reference line.

Chance level uses the same formula as your earlier k-NN pipeline: the
probability that a randomly paired novel shares the same author,
weighted by actual author group sizes (fairer than a flat 1/n_authors
when group sizes are uneven).
"""

import matplotlib
matplotlib.use("Agg")  # headless-safe backend; still works fine for saving PNGs
import matplotlib.pyplot as plt
import numpy as np

from ngram_categories import CATEGORIES
from emergence_plateau import detect_plateau_slope

ALL_LINES = CATEGORIES + ["combined"]


def chance_level(labels):
    """
    Same chance-level formula used in your k-NN pipeline: probability a
    randomly paired novel shares the same author, weighted by actual
    author group sizes.
    """
    labels = np.asarray(labels)
    n = len(labels)
    if n <= 1:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    return sum((c / n) * ((c - 1) / (n - 1)) for c in counts)


def plot_emergence(results, window_lengths, labels, epsilon=0.01, consecutive=3,
                    title="Feature emergence across window length"):
    """
    results: dict {line_name: [accuracy at each window_length]} -- exactly
             what run_window_sweep() returns. NaN entries (windows where
             a category/window had to be skipped) are handled gracefully.
    window_lengths: same list passed into run_window_sweep().
    labels: the author labels used for the sweep (needed for the chance line).

    Returns a matplotlib Figure; caller decides whether to show/save it.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    chance = chance_level(labels)
    ax.axhline(chance, color="gray", linestyle="--", linewidth=1,
               label=f"chance level ({chance:.2f})")

    cmap = plt.get_cmap("tab20")
    for idx, line in enumerate(ALL_LINES):
        accuracies = results[line]
        is_combined = (line == "combined")
        color = "black" if is_combined else cmap(idx % 20)
        linewidth = 2.5 if is_combined else 1.3

        ax.plot(window_lengths, accuracies, label=line, color=color,
                linewidth=linewidth, marker="o", markersize=3)

        # Mark this line's plateau point, skipping any NaN windows first
        clean = [(w, a) for w, a in zip(window_lengths, accuracies)
                 if a is not None and not np.isnan(a)]
        if len(clean) >= consecutive + 1:
            w_clean, a_clean = zip(*clean)
            plateau_w = detect_plateau_slope(list(w_clean), list(a_clean),
                                              epsilon=epsilon, consecutive=consecutive)
            if plateau_w is not None:
                plateau_acc = a_clean[list(w_clean).index(plateau_w)]
                ax.scatter([plateau_w], [plateau_acc], color=color,
                           edgecolor="black", zorder=5, s=60, marker="D")

    ax.set_xlabel("Window length (characters)")
    ax.set_ylabel("Classification accuracy")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # Smoke test: same synthetic corpus used in window_sweep.py's own demo.
    # This only confirms the plotting code runs end to end -- the
    # near-ceiling accuracies come from deliberately cartoonish synthetic
    # "authors" and are not evidence about anything real.
    from window_sweep import run_window_sweep

    rng = np.random.default_rng(0)

    def make_novel(style, length=300):
        chunks = {
            "A": ["the cat sat, quietly", "and then it ran", "she said, 'no'"],
            "B": ["birds fly-high above", "he ran-fast today", "we saw-it clearly"],
            "C": ["it was raining hard.", "the sky was grey.", "no one spoke then."],
        }[style]
        text = ""
        while len(text) < length:
            text += rng.choice(chunks) + " "
        return text[:length]

    documents, labels = [], []
    for author in ["A", "B", "C"]:
        for _ in range(6):
            documents.append(make_novel(author))
            labels.append(author)

    window_lengths = [50, 100, 150, 250]
    results = run_window_sweep(documents, labels, window_lengths,
                                n=3, min_count=2, n_folds=3, seed=0, verbose=False)

    fig = plot_emergence(results, window_lengths, labels)
    fig.savefig("/tmp/emergence_plot_test.png", dpi=100)
    print("Saved test plot to /tmp/emergence_plot_test.png")