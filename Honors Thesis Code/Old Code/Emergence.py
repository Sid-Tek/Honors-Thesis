"""
EmergenceSweep.py

Runs the "does this feature emerge" question across a sweep of window
lengths instead of a single one. For each window length in --lengths:

  1. Extracts the voice vector for every file in the corpus, truncated to
     that many characters (reuses Vector.py's extractor).
  2. For each INDIVIDUAL feature, and for the FULL combined vector, runs
     the same 1-NN + permutation test as Analysis.py -- but scoped to that
     one column (or all columns) -- to get an accuracy and a p-value.

Output:
  - emergence_results.csv   one row per (window_length, feature) with accuracy + p_value
  - emergence_curve.png     accuracy vs. window length, one line per feature,
                             plus one line for the combined vector

This answers: at what length does each feature start beating chance, and
does the combined vector emerge faster than any single feature alone?

IMPORTANT: must sit in the same folder as Vector.py and Test.py, since it
imports from both.

Usage:
    python Emergence.py 100_english_novels-master/corpus
    python Emergence.py 100_english_novels-master/corpus --lengths 100,250,500,1000,2000,4000,8000
    python Emergence.py 100_english_novels-master/corpus --lengths 100,250,500,1000,2000,4000,8000 --permutations 500
    python Emergence.py 100_english_novels-master/corpus --lengths 100,250,500,1000,2000,4000,8000,10000 --permutations 3000 --model en_core_web_lg --out results.csv --plot emergence_curve1B.png
    python Emergence.py 100_english_novels-master/corpus --lengths 3900,4000,4100,4200,4300,4400,4500 --permutations 1000 --model en_core_web_lg --out results.csv --plot emergence_curve3.png

"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from Vector import load_text, get_nlp, extract_voice_vector
from Test import discover_files


# ---------------------------------------------------------------------------
# Vector extraction at a given window length
# ---------------------------------------------------------------------------

def extract_at_length(pairs, n_chars, nlp):
    """Run the full corpus through Vector.py's extractor at one window length.
    Returns a DataFrame with 'filename', 'author', and one column per feature.

    NOTE: this re-parses every file at every window length (fresh spaCy call
    each time), which is the simplest correct thing to do but not the fastest.
    If the sweep gets too slow, the optimization is to parse each file once
    at the largest length and take sentence-boundary-respecting prefixes of
    that single parse for the shorter lengths instead -- left out here to
    keep this version straightforward and easy to trust the numbers from.
    """
    rows = []
    for f, author in pairs:
        text = load_text(f, n_chars)
        vector = extract_voice_vector(text, nlp)
        rows.append({"filename": f.name, "author": author, **vector})
    return pd.DataFrame(rows)


def feature_columns(df: pd.DataFrame):
    exclude = {"filename", "author"}
    return [c for c in df.columns if c not in exclude and not c.startswith("_")]


def standardized_matrix(df: pd.DataFrame, cols):
    X = df[cols].to_numpy(dtype=float)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1.0  # avoid divide-by-zero on constant columns
    return (X - mu) / sigma


# ---------------------------------------------------------------------------
# 1-NN + permutation test, scoped to an arbitrary column subset
# ---------------------------------------------------------------------------

def nn_accuracy_and_pvalue(X: np.ndarray, authors: np.ndarray,
                            n_permutations: int, seed: int):
    """Same test as Analysis.py's nearest_neighbor_check, but returns numbers
    instead of printing, and works on whatever columns X already contains --
    a single feature, or the full vector."""
    n = len(authors)
    if n < 3:
        return float("nan"), float("nan")

    nearest_idx = np.empty(n, dtype=int)
    for i in range(n):
        dists = np.linalg.norm(X - X[i], axis=1)
        dists[i] = np.inf  # exclude itself
        nearest_idx[i] = np.argmin(dists)

    matches = authors[nearest_idx] == authors
    accuracy = float(matches.mean())

    rng = np.random.default_rng(seed)
    perm_accs = np.empty(n_permutations)
    for k in range(n_permutations):
        shuffled = rng.permutation(authors)
        perm_accs[k] = np.mean(shuffled[nearest_idx] == shuffled)
    p_value = float(np.mean(perm_accs >= accuracy - 1e-12))

    return accuracy, p_value


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(corpus_dir: Path, lengths, model: str,
              n_permutations: int, seed: int) -> pd.DataFrame:
    pairs = discover_files(corpus_dir)
    n_authors = len(set(a for _, a in pairs))
    print(f"Found {len(pairs)} files across {n_authors} authors.\n")

    print(f"Loading spaCy model '{model}'... (one-time cost)")
    nlp = get_nlp(model)

    results = []
    for n_chars in lengths:
        print(f"\n=== Window length: {n_chars} chars ===")
        df = extract_at_length(pairs, n_chars, nlp)
        cols = feature_columns(df)
        authors = df["author"].to_numpy()

        # Each feature tested on its own
        for col in cols:
            X = standardized_matrix(df, [col])
            acc, p = nn_accuracy_and_pvalue(X, authors, n_permutations, seed)
            results.append({"length": n_chars, "feature": col,
                             "accuracy": acc, "p_value": p})
            print(f"  {col:28s} acc={acc:.1%}  p={p:.3f}")

        # Full combined vector
        X_all = standardized_matrix(df, cols)
        acc, p = nn_accuracy_and_pvalue(X_all, authors, n_permutations, seed)
        results.append({"length": n_chars, "feature": "ALL_COMBINED",
                         "accuracy": acc, "p_value": p})
        print(f"  {'ALL_COMBINED':28s} acc={acc:.1%}  p={p:.3f}")

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_emergence(results: pd.DataFrame, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 7))

    for feat in results["feature"].unique():
        sub = results[results["feature"] == feat].sort_values("length")
        if feat == "ALL_COMBINED":
            style = dict(linewidth=3.2, linestyle="-", alpha=1.0, zorder=5, color="black")
        else:
            style = dict(linewidth=1.2, linestyle="--", alpha=0.75)
        ax.plot(sub["length"], sub["accuracy"], marker="o", markersize=4,
                label=feat, **style)

    #ax.set_xscale("log")
    ax.set_xscale("linear")
    ax.set_xlabel("Window length (characters, linear scale)")
    ax.set_ylabel("Nearest-neighbor accuracy")
    ax.set_title("Feature emergence: accuracy vs. window length")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.axhline(0, color="gray", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Sweep window length to find each feature's emergence rate."
    )
    ap.add_argument("corpus_dir", type=Path)
    ap.add_argument("--lengths", default="100,250,500,1000,2000,4000,8000",
                     help="Comma-separated window lengths in characters")
    ap.add_argument("--model", default="en_core_web_lg")
    ap.add_argument("--permutations", type=int, default=1000,
                     help="Lower this (e.g. 200-500) for faster iteration while testing")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("emergence_results.csv"))
    ap.add_argument("--plot", type=Path, default=Path("emergence_curve.png"))
    args = ap.parse_args()

    if not args.corpus_dir.exists():
        sys.exit(f"Folder not found: {args.corpus_dir}")

    lengths = sorted(int(x) for x in args.lengths.split(","))

    results = run_sweep(args.corpus_dir, lengths, args.model,
                         args.permutations, args.seed)
    results.to_csv(args.out, index=False)
    print(f"\nSaved results to {args.out}")

    plot_emergence(results, args.plot)


if __name__ == "__main__":
    main()