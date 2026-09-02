"""
analyze_vectors.py


Loads vectors.csv (produced by batch_vectors.py) and checks whether the
voice vectors actually cluster by author -- the core hypothesis of the
Variables in Voice Emergence project.

Four checks, from basic to rigorous:

  1. Data sanity check       -- enough novels per author? any missing values?
  2. Distance check          -- are same-author pairs closer than different-author
                                 pairs, on average?
  3. Nearest-neighbor test   -- for each novel, is its single closest neighbor
                                 (excluding itself) by the SAME author? This is a
                                 much stronger test than average distance, and
                                 gives you a reportable accuracy number vs. a
                                 chance baseline -- this is the number you'd
                                 actually put in a thesis.
  4. 2D visualization (PCA)  -- a scatter plot colored by author, saved as a PNG,
                                 so you can literally see whether authors cluster.

Usage:
    pip install pandas numpy scikit-learn matplotlib --break-system-packages
    python Analysis.py vectors.csv
"""

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
    return df


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
# 2. Distance check
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
# 3. Nearest-neighbor test
# ---------------------------------------------------------------------------

def nearest_neighbor_check(df: pd.DataFrame, X: np.ndarray, n_permutations: int = 2000, seed: int = 0):
    print("--- 3. Nearest-neighbor test (the real test) ---")
    authors = df["author"].to_numpy()
    n = len(df)

    # Nearest neighbor depends only on X (the feature vectors), never on the
    # author labels -- so we compute it once, then can cheaply re-check it
    # against many reshuffled label assignments below.
    nearest_idx = np.empty(n, dtype=int)
    for i in range(n):
        dists = np.linalg.norm(X - X[i], axis=1)
        dists[i] = np.inf  # exclude itself
        nearest_idx[i] = np.argmin(dists)

    matches = authors[nearest_idx] == authors
    correct = int(matches.sum())
    accuracy = correct / n

    author_sizes = df["author"].value_counts()
    chance = sum((size / n) * ((size - 1) / (n - 1)) for size in author_sizes) if n > 1 else 0.0

    print(f"Nearest neighbor matched the same author: {correct}/{n} = {accuracy:.1%}")
    print(f"Naive chance level given your author group sizes: {chance:.1%}")

    # Permutation test: shuffle which author label goes with which novel
    # (keeping the actual geometry / nearest-neighbor pairs fixed) and see how
    # often random labeling alone produces an accuracy this high. This is the
    # statistically honest check -- comparing directly to the naive chance
    # number above is unreliable for small corpora and can look "significant"
    # on pure noise more than half the time at n~12.
    rng = np.random.default_rng(seed)
    perm_accs = np.empty(n_permutations)
    for k in range(n_permutations):
        shuffled = rng.permutation(authors)
        perm_accs[k] = np.mean(shuffled[nearest_idx] == shuffled)
    p_value = float(np.mean(perm_accs >= accuracy - 1e-12))

    print(f"Permutation test p-value: {p_value:.3f}  "
          f"(probability of matching this well under random author labels)")
    if p_value < 0.05:
        print("-> statistically significant: the vectors are picking up real authorial signal\n")
    else:
        print("-> NOT statistically significant at this sample size -- the result could plausibly "
              "be chance. More novels per author would make this test more trustworthy.\n")

    print("Per-author nearest-neighbor accuracy (worst offenders first):")
    per_author_acc = {a: matches[authors == a].mean() for a in author_sizes.index}
    for a, acc in sorted(per_author_acc.items(), key=lambda r: r[1])[:10]:
        print(f"  {a:20s} {acc:.0%}  (n={author_sizes[a]})")
    print()


# ---------------------------------------------------------------------------
# 4. 2D visualization
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
    if len(sys.argv) < 2:
        sys.exit("Usage: python analyze_vectors.py vectors.csv")
    path = Path(sys.argv[1])
    if not path.exists():
        sys.exit(f"File not found: {path}")

    df = load_and_check(path)
    cols = feature_columns(df)
    print(f"Using {len(cols)} numeric features: {cols}\n")

    X = standardized_matrix(df, cols)

    distance_check(df, X)
    nearest_neighbor_check(df, X)
    plot_pca(df, X, path.with_name(path.stem + "_pca.png"))


if __name__ == "__main__":
    main()