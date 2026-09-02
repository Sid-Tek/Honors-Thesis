"""
Fixed-window, variable-POSITION sweep -- the position-controlled companion
to window_sweep.py's cumulative-from-start sweep.

window_sweep.py answers: "as I feed the classifier more and more text
from the start of the novel, how does accuracy change?" That conflates
two effects that grow together as the window grows: (1) the narrative
genuinely developing more distinctive style as it goes, and (2) the
n-gram frequency estimates simply getting less noisy because there are
more tokens to count, regardless of where they came from. Effect (2)
never truly flattens -- it just keeps making smaller and smaller gains
as the sample grows -- which is the likely reason the cumulative curves
never plateau even out to 30,000 characters.

This module holds window SIZE fixed and instead sweeps the STARTING
POSITION (character offset) within each novel. That isolates the two
effects:

  - If accuracy is flat across offset (at a fixed, small size), the
    style is already fully present from the very first window -- what
    looked like slow "emergence" in the cumulative sweep was actually
    just the noise-reduction effect of accumulating more characters,
    not the narrative developing anything new. This is the "fast-acting"
    result: the feature was there all along, it just needed enough
    tokens to detect reliably.

  - If accuracy climbs with offset even at a fixed size, that's real
    narrative-time emergence: the feature genuinely needs the story to
    progress, not just more raw characters, to show itself. Not
    "fast-acting" in this project's sense.

Same fixed-vocabulary discipline as window_sweep.py: vocabulary is built
ONCE from the full, untruncated novels, and reused at every offset/size,
so a shrinking vocabulary at short windows never gets confused with a
real absence of signal.
"""

import numpy as np
from scipy import stats

from vectorizer import build_vocabulary, vectorize_document
from ngram_categories import CATEGORIES
from emergence_plateau import measure_accuracy

ALL_LINES = CATEGORIES + ["combined"]


# ---------------------------------------------------------------------------
# Core sweep
# ---------------------------------------------------------------------------

def run_position_sweep(documents, labels, window_size, offsets, n=3, min_count=5,
                        n_folds=5, seed=0, n_repeats=1, verbose=True, vocabulary=None):
    """
    documents, labels: same as run_window_sweep.
    window_size: a SINGLE fixed window length in characters (e.g. 1000).
    offsets: list of starting character positions to test, e.g.
             [0, 5000, 10000, 15000, 20000, 25000]. A novel is only
             usable at a given offset if len(novel) >= offset + window_size
             -- exactly the same "drop, don't truncate short" policy
             window_sweep.py uses, just keyed on offset+size instead of
             just size.
    vocabulary: pass in a pre-built vocabulary (from build_vocabulary) to
                skip rebuilding it -- used by run_position_sweep_multi_size
                so the SAME vocabulary is shared across every window size,
                keeping sizes comparable to each other.

    Returns: dict {line_name: [accuracy_at_offset_0, accuracy_at_offset_1, ...]}
             NaN where too few novels/authors survived at that offset.
    """
    if vocabulary is None:
        if verbose:
            print("Building vocabulary from full, untruncated novels...")
        vocabulary = build_vocabulary(documents, n=n, min_count=min_count)
        if verbose:
            for cat in CATEGORIES:
                print(f"  {cat:14s} {len(vocabulary[cat])} n-grams kept "
                      f"(>= {min_count} occurrences)")

    documents = list(documents)
    labels = np.array(labels)
    accuracies = {line: [] for line in ALL_LINES}

    for offset in offsets:
        if verbose:
            print(f"\nOffset: {offset} chars (window size {window_size})")

        # Keep only novels long enough to reach offset + window_size --
        # same "drop, don't truncate short" policy as window_sweep.py.
        keep_idx = [i for i, doc in enumerate(documents)
                    if len(doc) >= offset + window_size]
        dropped = len(documents) - len(keep_idx)
        if dropped and verbose:
            print(f"  (dropping {dropped} novel(s) too short to reach "
                  f"offset {offset} + window {window_size})")

        offset_labels = labels[keep_idx]
        unique, counts = np.unique(offset_labels, return_counts=True)
        min_class_size = counts.min() if len(counts) else 0

        if len(unique) < 2 or min_class_size < n_folds:
            if verbose:
                print(f"  Not enough data left at this offset "
                      f"(authors={len(unique)}, smallest class="
                      f"{min_class_size}, need >= {n_folds}) -- recording NaN")
            for line in ALL_LINES:
                accuracies[line].append(np.nan)
            continue

        windows = [documents[i][offset:offset + window_size] for i in keep_idx]
        per_doc_vectors = [vectorize_document(text, vocabulary, n=n)
                            for text in windows]

        for line in ALL_LINES:
            X = np.array([v[line] for v in per_doc_vectors])
            if X.shape[1] == 0:
                accuracies[line].append(np.nan)
                if verbose:
                    print(f"  {line:14s} -- empty vocabulary, skipping")
                continue
            acc = measure_accuracy(X, offset_labels, n_folds=n_folds,
                                    seed=seed, n_repeats=n_repeats)
            accuracies[line].append(acc)
            if verbose:
                print(f"  {line:14s} accuracy={acc:.3f}")

    return accuracies


def run_position_sweep_multi_size(documents, labels, window_sizes, offsets, n=3,
                                   min_count=5, n_folds=5, seed=0, n_repeats=1,
                                   verbose=True):
    """
    Runs run_position_sweep once per size in `window_sizes`, sharing ONE
    vocabulary (built once, from full documents) across all of them, so
    accuracy is comparable across sizes and not an artifact of the
    vocabulary itself shifting.

    Returns: dict {window_size: {line: [accuracy_at_offset_0, ...]}}
    """
    if verbose:
        print("Building vocabulary from full, untruncated novels "
              "(shared across all window sizes)...")
    vocabulary = build_vocabulary(documents, n=n, min_count=min_count)
    if verbose:
        for cat in CATEGORIES:
            print(f"  {cat:14s} {len(vocabulary[cat])} n-grams kept "
                  f"(>= {min_count} occurrences)")

    results_by_size = {}
    for w in window_sizes:
        if verbose:
            print(f"\n=== Window size: {w} characters ===")
        results_by_size[w] = run_position_sweep(
            documents, labels, w, offsets, n=n, min_count=min_count,
            n_folds=n_folds, seed=seed, n_repeats=n_repeats, verbose=verbose,
            vocabulary=vocabulary,
        )
    return results_by_size


# ---------------------------------------------------------------------------
# Flatness testing -- turns "eyeballing a flat line" into a real test
# ---------------------------------------------------------------------------

def flatness_test(offsets, accuracies):
    """
    Tests whether one line's accuracy is flat across position, at one
    window size. Fits accuracy ~ offset by simple linear regression
    (after dropping NaN offsets) and reports the slope and its p-value.

    "Flat" in this project's sense = the slope is NOT significantly
    different from zero, i.e. position doesn't detectably predict
    accuracy. A HIGH p-value is evidence FOR flatness here (same logic
    as the permutation-based plateau test elsewhere in this project:
    failing to detect a slope is the claim, not a failure of the test).

    Returns None if fewer than 3 valid (offset, accuracy) points remain
    -- not enough points to fit a meaningful line.

    Returns dict: {slope, p_value, mean_accuracy, n_points}
    """
    pairs = [(o, a) for o, a in zip(offsets, accuracies)
             if a is not None and not np.isnan(a)]
    if len(pairs) < 3:
        return None

    o_clean, a_clean = zip(*pairs)
    a_arr = np.array(a_clean, dtype=float)

    # Degenerate case: accuracy is EXACTLY constant across every offset
    # (e.g. a feature saturated at 100% everywhere). linregress can't
    # compute a p-value with zero residual variance and returns nan,
    # which would otherwise get read as "not flat" by an >= alpha check
    # -- backwards, since zero variance is the most flat a line can be.
    # Treat it as maximal evidence for flatness instead.
    if np.allclose(a_arr, a_arr[0]):
        return {
            "slope": 0.0,
            "p_value": 1.0,
            "mean_accuracy": float(a_arr.mean()),
            "n_points": len(pairs),
        }

    result = stats.linregress(o_clean, a_clean)
    return {
        "slope": result.slope,
        "p_value": result.pvalue,
        "mean_accuracy": float(a_arr.mean()),
        "n_points": len(pairs),
    }


def find_fast_acting_size(window_sizes, offsets, results_by_size, alpha=0.05,
                           ceiling=None, ceiling_fraction=0.9):
    """
    For each line, scans window_sizes smallest -> largest and returns the
    first size at which:

      (a) accuracy-vs-offset is flat (flatness_test p_value >= alpha), AND
      (b) [if `ceiling` given] mean accuracy at that size is at least
          `ceiling_fraction` of that line's ceiling accuracy.

    ceiling: optional dict {line: accuracy}, e.g. that line's accuracy at
             a long cumulative-from-start window (see --ceiling-length in
             run_pipeline.py). Without it, only condition (a) is checked,
             which can call a line "fast-acting" even if it's flat at a
             uselessly low accuracy -- pass a ceiling whenever you have
             one available.

    Returns: dict {line: window_size_or_None}. None = this line never
    satisfied the condition(s) within the sizes you tested -- no evidence
    of fast-acting, position-invariant detectability found (yet).
    """
    lines = list(results_by_size[window_sizes[0]].keys())
    fast_acting = {}
    for line in lines:
        found = None
        for w in sorted(window_sizes):
            ft = flatness_test(offsets, results_by_size[w][line])
            if ft is None:
                continue
            is_flat = ft["p_value"] >= alpha
            meets_ceiling = True
            if ceiling is not None and line in ceiling and not np.isnan(ceiling.get(line, np.nan)):
                meets_ceiling = ft["mean_accuracy"] >= ceiling_fraction * ceiling[line]
            if is_flat and meets_ceiling:
                found = w
                break
        fast_acting[line] = found
    return fast_acting


def print_fast_acting_summary(fast_acting, ceiling=None):
    """Pretty-prints find_fast_acting_size()'s result."""
    label = "position-flat" + (" and near-ceiling" if ceiling else "")
    print(f"\nFast-acting size (smallest window that is {label}):")
    for line, size in sorted(fast_acting.items(), key=lambda kv: (kv[1] is None, kv[1] or 0)):
        if size is None:
            print(f"  {line:14s} -- not reached within tested sizes")
        else:
            print(f"  {line:14s} {size} characters")


if __name__ == "__main__":
    # Smoke test with the same synthetic corpus style used elsewhere in
    # this project. "A" and "B" are given a real, POSITION-INDEPENDENT
    # habit (their punctuation choice never changes), so this should
    # recover flat lines even at the smallest size tested -- confirming
    # the plumbing works end to end before running on real novels.
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

    fast_acting = find_fast_acting_size(window_sizes, offsets, results_by_size, alpha=0.05)
    print_fast_acting_summary(fast_acting)