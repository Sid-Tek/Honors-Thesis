"""
Runs the emergence sweep: for a corpus of novels (with author labels),
vectorizes each novel at a sweep of window lengths (characters), using a
FIXED vocabulary built once from the full novels, and measures SVM
accuracy at each window length -- separately for each of the 10 n-gram
categories, plus the combined vector.

This produces the 11 accuracy curves (10 categories + combined) that
emergence_plateau.py then reads to find where each curve flattens.

DESIGN NOTE -- short novels at large windows:
At a given window length w, a novel shorter than w characters can't be
truncated to w without giving it less real evidence than its longer
siblings -- which would silently make that window's comparison unfair
right at the point being measured. This module's default behavior is to
DROP novels shorter than w from that window's test entirely (with a
warning), rather than truncate them short. Practical consequence: your
largest window lengths will only be testable on your longest novels --
your shortest novel in the corpus effectively caps how far the sweep can
meaningfully reach.
"""

import numpy as np
from vectorizer import build_vocabulary, vectorize_document
from ngram_categories import CATEGORIES
from emergence_plateau import measure_accuracy

ALL_LINES = CATEGORIES + ["combined"]


def run_window_sweep(documents, labels, window_lengths, n=3, min_count=5,
                      n_folds=5, seed=0, n_repeats=1, verbose=True):
    """
    documents: list of full raw text strings (one per novel)
    labels:    list/array of author labels, same order as documents
    window_lengths: list of character counts to test, e.g. [500, 1000, ...]

    Returns: dict {line_name: [accuracy_at_window_0, accuracy_at_window_1, ...]}
             line_name is one of the 10 categories or "combined".
             A value is np.nan for a window where too few novels/authors
             remained to test (see design note above), or where a
             category's vocabulary was empty.
    """
    if verbose:
        print("Building vocabulary from full, untruncated novels...")
    vocabulary = build_vocabulary(documents, n=n, min_count=min_count)
    for cat in CATEGORIES:
        if verbose:
            print(f"  {cat:14s} {len(vocabulary[cat])} n-grams kept "
                  f"(>= {min_count} occurrences)")

    documents = list(documents)
    labels = np.array(labels)
    accuracies = {line: [] for line in ALL_LINES}

    for w in window_lengths:
        if verbose:
            print(f"\nWindow length: {w} characters")

        # Keep only novels long enough to actually reach this window --
        # see module docstring for why we don't truncate short ones.
        keep_idx = [i for i, doc in enumerate(documents) if len(doc) >= w]
        dropped = len(documents) - len(keep_idx)
        if dropped and verbose:
            print(f"  (dropping {dropped} novel(s) shorter than {w} chars)")

        window_labels = labels[keep_idx]
        unique, counts = np.unique(window_labels, return_counts=True)
        min_class_size = counts.min() if len(counts) else 0

        if len(unique) < 2 or min_class_size < n_folds:
            # Not enough authors, or not enough novels for the smallest
            # author, to run n_folds-fold cross-validation meaningfully.
            if verbose:
                print(f"  Not enough data left at this window "
                      f"(authors={len(unique)}, smallest class="
                      f"{min_class_size}, need >= {n_folds}) -- recording NaN")
            for line in ALL_LINES:
                accuracies[line].append(np.nan)
            continue

        windows = [documents[i][:w] for i in keep_idx]
        per_doc_vectors = [vectorize_document(text, vocabulary, n=n)
                            for text in windows]

        for line in ALL_LINES:
            X = np.array([v[line] for v in per_doc_vectors])
            if X.shape[1] == 0:
                accuracies[line].append(np.nan)
                if verbose:
                    print(f"  {line:14s} -- empty vocabulary, skipping")
                continue
            acc = measure_accuracy(X, window_labels, n_folds=n_folds, seed=seed, n_repeats=n_repeats)
            accuracies[line].append(acc)
            if verbose:
                print(f"  {line:14s} accuracy={acc:.3f}")

    return accuracies


if __name__ == "__main__":
    # Small synthetic smoke test: 3 "authors", 6 short synthetic "novels"
    # each, with deliberately different punctuation/word habits so the
    # classifier has something real to find. This only tests that the
    # plumbing runs end to end -- it is NOT evidence about real novels.
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
                                n=3, min_count=2, n_folds=3, seed=0)

    print("\nFinal accuracy table (rows=lines, cols=window lengths):")
    header = "line".ljust(14) + "".join(f"{w:>8}" for w in window_lengths)
    print(header)
    for line in ALL_LINES:
        row = line.ljust(14) + "".join(
            f"{v:8.3f}" if not np.isnan(v) else f"{'nan':>8}"
            for v in results[line]
        )
        print(row)