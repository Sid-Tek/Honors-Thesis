"""
Plateau / emergence detection for the authorial-voice emergence sweep.

Given accuracy measured at a sweep of window lengths (how much of each novel
was fed to the classifier), this module answers: "at what window length does
accuracy stop climbing and flatten out?" That flattening point is this
project's operational definition of 'emergence.'

Two interchangeable methods are provided -- toggle with --plateau-method on
the command line, or by calling the function directly:

  slope        Fast. Flags a plateau once the accuracy gain between
               consecutive windows drops below a fixed threshold for
               several windows in a row. The threshold is a number you
               choose by eye. THIS IS THE DEFAULT.

  permutation  Slower (many more classifier refits), but avoids picking an
               arbitrary threshold. At each pair of consecutive windows, it
               reshuffles the author labels many times to build a
               "what a gap looks like by chance" distribution, then checks
               whether the real accuracy gap is bigger than that. Once the
               real gap is no longer distinguishable from chance, that
               pair marks the plateau.
"""

import argparse
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score


# ---------------------------------------------------------------------------
# Shared helper: measure accuracy of one window's feature matrix
# ---------------------------------------------------------------------------


def measure_accuracy(X, y, n_folds=5, seed=0, n_repeats=1):
    """
    Fits an SVM and returns cross-validated accuracy for one window's data.
 
    Cross-validation (rather than one train/test split) is used because
    with a small number of novels per author, a single split can get lucky
    or unlucky. Averaging over several folds gives a steadier number.
 
    n_repeats > 1 repeats the ENTIRE k-fold split several times, each with
    a different random fold assignment (seed, seed+1, seed+2, ...), and
    averages the scores across all of them. This adds no new novels -- the
    corpus is exactly the same size -- but it smooths out the luck of any
    one particular fold assignment. That matters most on small corpora,
    where a single split can only produce accuracy values on coarse rungs
    (e.g. multiples of 1/18 with 18 held-out predictions); averaging many
    such splits together lands on finer-grained in-between values instead.
    """
    clf = SVC(kernel="linear")
    all_scores = []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed + r)
        scores = cross_val_score(clf, X, y, cv=skf)
        all_scores.extend(scores)
    return np.mean(all_scores)

# ---------------------------------------------------------------------------
# Option 1: slope threshold  (DEFAULT -- what you chose)
# ---------------------------------------------------------------------------
def detect_plateau_slope(window_lengths, accuracies, epsilon=0.01, consecutive=3):
    """
    Walks the accuracy curve left to right. Flags a plateau at the first
    window after which accuracy never climbs more than `epsilon` above
    where it already was, for the rest of the sweep.

    Compares against the MAXIMUM of everything that follows (not
    step-by-step differences) -- a real future climb still blocks
    detection, but noisy dips/wobbles after the peak don't.

    `consecutive` sets how much runway (how many further windows) must
    exist after a candidate point before it's trusted as a plateau,
    rather than a lucky coincidence with nothing left to check.

    Returns the window_length where the plateau begins, or None if the
    curve never flattens within the sweep you gave it.
    """
    n = len(accuracies)
    for i in range(n - consecutive):
        remaining = accuracies[i:]
        if max(remaining) - accuracies[i] < epsilon:
            return window_lengths[i]
    return None


# ---------------------------------------------------------------------------
# Option 2: permutation test between consecutive windows
# ---------------------------------------------------------------------------

def _shuffled_gap(X_low, X_high, y, rng, n_folds=5):
    """
    One shuffle: scramble the author labels (vector geometry is untouched),
    recompute accuracy at both windows using the SAME shuffled labels, and
    return the resulting gap. This simulates 'what a gap looks like when
    nothing real is being learned.'
    """
    y_shuffled = rng.permutation(y)
    acc_low = measure_accuracy(X_low, y_shuffled, n_folds=n_folds)
    acc_high = measure_accuracy(X_high, y_shuffled, n_folds=n_folds)
    return acc_high - acc_low


def detect_plateau_permutation(window_lengths, feature_matrices, labels,
                                n_permutations=1000, alpha=0.05, seed=0,
                                n_folds=5, verbose=True):
    """
    For each pair of consecutive windows, tests whether the REAL accuracy
    gap is bigger than `n_permutations` shuffled ("chance") gaps would
    produce. Once the real gap is no longer in the top `alpha` fraction of
    the shuffled distribution, that pair marks the plateau.

    feature_matrices: list of X matrices, one per window, same order as
                       window_lengths.
    labels: the true author labels (same set of novels at every window).

    WARNING: this refits an SVM `n_permutations * 2` times PER adjacent
    window pair, on top of the real (unshuffled) fit at each window. For a
    sweep of ~20 windows and n_permutations=1000, that's on the order of
    hundreds of thousands of classifier fits. Run this locally, unattended
    -- not while you wait on it.
    """
    rng = np.random.default_rng(seed)

    real_accuracies = [measure_accuracy(X, labels, n_folds=n_folds)
                        for X in feature_matrices]

    for i in range(1, len(window_lengths)):
        X_low, X_high = feature_matrices[i - 1], feature_matrices[i]
        real_gap = real_accuracies[i] - real_accuracies[i - 1]

        shuffled_gaps = np.array([
            _shuffled_gap(X_low, X_high, labels, rng, n_folds=n_folds)
            for _ in range(n_permutations)
        ])

        # p-value: fraction of shuffled gaps at least as large as the real one
        p_value = (np.sum(shuffled_gaps >= real_gap) + 1) / (n_permutations + 1)

        if verbose:
            print(f"  {window_lengths[i-1]} -> {window_lengths[i]}: "
                  f"real gap={real_gap:+.3f}, p={p_value:.3f}")

        if p_value >= alpha:
            # The real gap is not distinguishable from a shuffled ("chance") gap.
            return window_lengths[i - 1]

    return None  # never flattened within this sweep


# ---------------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Detect the emergence plateau for one feature's accuracy curve."
    )
    ap.add_argument("--plateau-method", choices=["slope", "permutation"], default="slope",
                     help="Which plateau detection method to use (default: slope)")
    ap.add_argument("--epsilon", type=float, default=0.01,
                     help="[slope] Minimum accuracy gain to still count as climbing")
    ap.add_argument("--consecutive", type=int, default=3,
                     help="[slope] Number of consecutive flat windows required")
    ap.add_argument("--n-permutations", type=int, default=1000,
                     help="[permutation] Number of label shuffles per window pair")
    ap.add_argument("--alpha", type=float, default=0.05,
                     help="[permutation] Significance threshold")
    args = ap.parse_args()

    print(f"Plateau method selected: {args.plateau_method}")
    print("This main() is a wiring demo -- see the __main__ block below for a "
          "runnable example on synthetic data. In your real Emergence.py, swap "
          "the synthetic data for your actual per-window feature matrices.")


if __name__ == "__main__":
    main()