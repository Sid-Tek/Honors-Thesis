"""
AnalysisKNN.py

Same as Analysis.py, but replaces the single-nearest-neighbor test with a
k-nearest-neighbors majority vote: for each novel, look at its k closest
neighbors (not just 1) and predict its author as whichever author is most
common among them. This is more forgiving of one weird neighbor, which
matters when the signal is faint (e.g. short text windows).

Everything else -- data sanity check, distance check, permutation test,
PCA plot -- works exactly like Analysis.py, just generalized from k=1 to
whatever k you choose.

Usage:
    pip install pandas numpy scikit-learn matplotlib --break-system-packages
    python AnalysisKNN.py vectors.csv
    python AnalysisKNN.py vectors.csv --k 5
    python AnalysisKNN.py vectors.csv --k 5 --permutations 2000

    python AnalysisKNN.py vectors.csv --k 2  --permutations 1000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. Load + sanity check
# ---------------------------------------------------------------------------

def load_and_check(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path.name}\n")

    print("--- 1. Data sanity check ---")
    counts = df["author"].value_counts()
    print(f"{len(counts)} unique authors.")
    singletons = counts[counts < 2]
    if len(singletons):
        print(f"WARNING: {len(singletons)} author(s) have only 1 novel -- "
              f"can't test 'closeness to same author' for these:")
        print(singletons.to_string())
    else:
        print("Every author has 2+ novels -- good, same-author comparisons are possible for all.")

    nan_counts = df.isna().sum()
    bad_cols = nan_counts[nan_counts > 0]
    if len(bad_cols):
        print(f"\nWARNING: missing values found in: {bad_cols.to_dict()}")
    else:
        print("No missing values.")
    print()
    return df, counts


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
# 2. Distance check (unchanged from Analysis.py)
# ---------------------------------------------------------------------------

def distance_check(df: pd.DataFrame, X: np.ndarray):
    print("--- 2. Distance check (mean same-author vs. different-author) ---")
    authors = df["author"].to_numpy()
    n = len(df)
    same, diff = [], []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(X[i] - X[j])
            (same if authors[i] == authors[j] else diff).append(d)

    same_mean = np.mean(same) if same else float("nan")
    diff_mean = np.mean(diff) if diff else float("nan")
    print(f"Same-author pairs:      {len(same):5d}   mean distance: {same_mean:.3f}")
    print(f"Different-author pairs: {len(diff):5d}   mean distance: {diff_mean:.3f}")
    if same:
        ratio = diff_mean / same_mean
        verdict = "same-author novels ARE closer on average -- good sign" if ratio > 1 else "no separation on average"
        print(f"Ratio (different / same): {ratio:.2f}  ({verdict})")
    print()


# ---------------------------------------------------------------------------
# 3. k-NN majority-vote test
# ---------------------------------------------------------------------------

def compute_neighbor_matrix(X: np.ndarray, k: int) -> np.ndarray:
    """For each row, the indices of its k closest other rows, ordered
    nearest-first. Depends only on X (the vectors), never on author labels --
    computed once, reused for every permutation below."""
    n = len(X)
    neighbor_idx = np.empty((n, k), dtype=int)
    for i in range(n):
        dists = np.linalg.norm(X - X[i], axis=1)
        dists[i] = np.inf  # exclude itself
        order = np.argsort(dists)
        neighbor_idx[i] = order[:k]
    return neighbor_idx


def _majority_vote(neighbor_labels_ordered):
    """Most common label among the neighbors. Ties are broken by distance:
    whichever tied label belongs to the closer neighbor wins, since
    neighbor_labels_ordered is already sorted nearest-first."""
    counts = {}
    for lbl in neighbor_labels_ordered:
        counts[lbl] = counts.get(lbl, 0) + 1
    max_count = max(counts.values())
    for lbl in neighbor_labels_ordered:
        if counts[lbl] == max_count:
            return lbl
    return neighbor_labels_ordered[0]  # unreachable, safety only


def knn_predict_all(authors: np.ndarray, neighbor_idx: np.ndarray) -> np.ndarray:
    n = len(authors)
    preds = np.empty(n, dtype=authors.dtype)
    for i in range(n):
        neighbor_labels = authors[neighbor_idx[i]]
        preds[i] = _majority_vote(list(neighbor_labels))
    return preds


def knn_check(df: pd.DataFrame, X: np.ndarray, author_counts: pd.Series,
              k: int, n_permutations: int = 2000, seed: int = 0):
    print(f"--- 3. k-NN majority-vote test (k={k}) ---")
    authors = df["author"].to_numpy()
    n = len(df)

    min_group = author_counts.min()
    if k > min_group - 1:
        print(f"NOTE: k={k} is larger than (smallest author group size - 1) = {min_group - 1}. "
              f"Authors with fewer than {k + 1} novels can never have all {k} neighbors be "
              f"same-author, even in a perfect model. Consider a smaller k, or read per-author "
              f"results below with that in mind.\n")

    neighbor_idx = compute_neighbor_matrix(X, k)

    preds = knn_predict_all(authors, neighbor_idx)
    matches = preds == authors
    correct = int(matches.sum())
    accuracy = correct / n

    chance = sum((size / n) * ((size - 1) / (n - 1)) for size in author_counts) if n > 1 else 0.0

    print(f"k-NN (k={k}) matched the same author: {correct}/{n} = {accuracy:.1%}")
    print(f"Naive chance level given your author group sizes: {chance:.1%}")

    # Permutation test, generalized from Analysis.py's 1-NN version: reshuffle
    # author labels (geometry / neighbor_idx stays fixed), recompute the
    # k-NN majority-vote accuracy each time, see how often random labels do
    # this well on their own.
    rng = np.random.default_rng(seed)
    perm_accs = np.empty(n_permutations)
    for p in range(n_permutations):
        shuffled = rng.permutation(authors)
        shuffled_preds = knn_predict_all(shuffled, neighbor_idx)
        perm_accs[p] = np.mean(shuffled_preds == shuffled)
    p_value = float(np.mean(perm_accs >= accuracy - 1e-12))

    print(f"Permutation test p-value: {p_value:.3f}  "
          f"(probability of matching this well under random author labels)")
    if p_value < 0.05:
        print("-> statistically significant: the vectors are picking up real authorial signal\n")
    else:
        print("-> NOT statistically significant at this sample size -- the result could plausibly "
              "be chance. More novels per author would make this test more trustworthy.\n")

    print("Per-author k-NN accuracy (worst offenders first):")
    per_author_acc = {a: matches[authors == a].mean() for a in author_counts.index}
    for a, acc in sorted(per_author_acc.items(), key=lambda r: r[1])[:10]:
        print(f"  {a:20s} {acc:.0%}  (n={author_counts[a]})")
    print()

    return accuracy, p_value


# ---------------------------------------------------------------------------
# 4. 2D visualization (unchanged from Analysis.py)
# ---------------------------------------------------------------------------

def plot_pca(df: pd.DataFrame, X: np.ndarray, out_path: Path):
    print("--- 4. 2D visualization ---")
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("scikit-learn not installed -- skipping PCA plot. "
              "Install with: pip install scikit-learn --break-system-packages")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coords = PCA(n_components=2, random_state=0).fit_transform(X)
    authors = df["author"].to_numpy()
    unique_authors = sorted(set(authors))

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("tab20", max(len(unique_authors), 1))
    for idx, a in enumerate(unique_authors):
        mask = authors == a
        ax.scatter(coords[mask, 0], coords[mask, 1], label=a, color=cmap(idx),
                   s=60, edgecolor="black", linewidth=0.3)

    ax.set_title("Voice vectors, reduced to 2D (PCA) -- colored by author")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    if len(unique_authors) <= 30:
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="k-NN majority-vote authorship check for a voice-vector CSV.")
    ap.add_argument("csv", type=Path, help="Path to vectors.csv (from Test.py)")
    ap.add_argument("--k", type=int, default=3, help="Number of nearest neighbors to vote with (default: 3)")
    ap.add_argument("--permutations", type=int, default=2000, help="Number of permutation-test shuffles (default: 2000)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for the permutation test (default: 0)")
    args = ap.parse_args()

    if not args.csv.exists():
        sys.exit(f"File not found: {args.csv}")
    if args.k < 1:
        sys.exit("--k must be at least 1")

    df, author_counts = load_and_check(args.csv)
    cols = feature_columns(df)
    print(f"Using {len(cols)} numeric features: {cols}\n")

    X = standardized_matrix(df, cols)

    distance_check(df, X)
    knn_check(df, X, author_counts, k=args.k, n_permutations=args.permutations, seed=args.seed)
    plot_pca(df, X, args.csv.with_name(args.csv.stem + "_pca.png"))


if __name__ == "__main__":
    main()