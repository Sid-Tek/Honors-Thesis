"""
Turns the raw category counts from ngram_categories.py into fixed-length
numeric vectors an SVM can actually train on.

Two steps, matching the paper's method:

  1. build_vocabulary(...)   -- decide ONCE which n-grams count as
                                 features, using the full (untruncated)
                                 training novels. Only n-grams occurring
                                 at least `min_count` times survive
                                 (the paper used min_count=5).

  2. vectorize_document(...) -- represent any document (full novel, or a
                                 truncated window of one, for the
                                 emergence sweep) as counts against that
                                 SAME fixed vocabulary, in the same
                                 order every time.

Building the vocabulary once from full novels and reusing it at every
window length is a deliberate choice: rebuilding it fresh at each short
window would make the feature list itself shrink/shift at exactly the
short windows you're trying to measure, confounding "voice hasn't
emerged yet" with "vocabulary barely exists yet at this length."
"""

from collections import Counter
from ngram_categories import extract_ngram_categories, CATEGORIES


def build_vocabulary(documents, n=3, min_count=5):
    """
    documents: list of full, untruncated raw text strings (your training
               novels).
    Returns: dict {category: [ngram, ngram, ...]}
             One fixed, sorted list of n-grams per category, kept only
             if its TOTAL count across all documents is >= min_count.
             Sorting makes the vector order reproducible across runs.
    """
    total_counts = {cat: Counter() for cat in CATEGORIES}
    for text in documents:
        doc_counts = extract_ngram_categories(text, n=n)
        for cat in CATEGORIES:
            total_counts[cat].update(doc_counts[cat])

    vocabulary = {}
    for cat in CATEGORIES:
        kept = [ngram for ngram, count in total_counts[cat].items()
                if count >= min_count]
        vocabulary[cat] = sorted(kept)
    return vocabulary


def vectorize_document(text, vocabulary, n=3):
    """
    Represents one document (could be a full novel, or a truncated
    window of one) as fixed-length count vectors against `vocabulary`.

    Returns dict {category: [count, count, ...]}, one vector per
    category (same length and order as vocabulary[category], every
    time), PLUS a "combined" key: all ten categories' vectors
    concatenated into one long vector -- this is the single reference
    line alongside your 10 per-category lines.
    """
    doc_counts = extract_ngram_categories(text, n=n)
    vectors = {}
    combined = []
    for cat in CATEGORIES:
        vec = [doc_counts[cat].get(ngram, 0) for ngram in vocabulary[cat]]
        vectors[cat] = vec
        combined.extend(vec)
    vectors["combined"] = combined
    return vectors


if __name__ == "__main__":
    # Tiny smoke test with toy "novels" standing in for full training texts
    training_docs = [
        "The actors wanted to see if the pact seemed like an old-fashioned one.",
        "The actors wanted to act, and acting was all they wanted.",
    ]

    vocab = build_vocabulary(training_docs, n=3, min_count=2)
    print("Vocabulary sizes per category (min_count=2):")
    for cat in CATEGORIES:
        print(f"  {cat:14s} {len(vocab[cat])} n-grams: {vocab[cat][:5]}")

    # Vectorize a short "window" -- a truncated slice of one novel
    window_text = training_docs[0][:30]
    vectors = vectorize_document(window_text, vocab, n=3)
    print(f"\nVectorizing a {len(window_text)}-character window:")
    for cat in CATEGORIES:
        print(f"  {cat:14s} length={len(vectors[cat])}  vector={vectors[cat]}")
    print(f"  {'combined':14s} length={len(vectors['combined'])}")