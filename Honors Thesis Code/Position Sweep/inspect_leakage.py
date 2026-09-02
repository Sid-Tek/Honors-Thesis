"""
Diagnostic: is a window's classification signal coming from genuine
authorial style, or from book-specific content (e.g. a character's name)
that happens to correlate with author because that author only wrote it
into one book?

For a handful of example novels, this looks at the n-grams that
dominate a SMALL, early window (default: the first 500 characters --
where the position sweep showed the strongest offset-dependence for
word/affix categories) and checks how often each one shows up ANYWHERE
else in the corpus:

  - repeats in OTHER books by the SAME author  -> consistent with a
    genuine authorial habit, used across their body of work.
  - barely appears anywhere else (not even in this author's other
    books) -> consistent with book-specific content -- a name, a place,
    a one-off word tied to that story's premise -- accidentally
    correlating with authorship because only one author happened to
    write that one book.

This doesn't change or fix anything in the pipeline. It's a look-under-
the-hood check to confirm (or rule out) the leakage hypothesis before
spending time on a code fix (a document-frequency filter in
vectorizer.py's build_vocabulary()).

Usage:
    python inspect_leakage.py /path/to/corpus_folder
    python inspect_leakage.py /path/to/corpus_folder --window-size 500 --n-examples 5
    python inspect_leakage.py /path/to/corpus_folder --categories prefix,suffix,whole-word,mid-word,beg-punct

python inspect_leakage.py 100_english_novels-master/corpus --window-size 500 --n-examples 5


"""

import argparse
from pathlib import Path

from ngram_categories import extract_ngram_categories, CATEGORIES

# The categories the position sweep implicated, plus one punctuation
# category shown alongside for contrast (punctuation structurally can't
# encode word identity, so it shouldn't show the same effect).
WORD_AFFIX_CATEGORIES_OF_INTEREST = ["prefix", "suffix", "whole-word", "mid-word", "multi-word"]
CONTRAST_CATEGORY = "beg-punct"


def load_corpus_with_titles(folder):
    """
    Same Author_Title.txt convention as run_pipeline.load_corpus(), but
    also keeps each file's stem for readable diagnostic output.
    """
    folder = Path(folder)
    documents, labels, titles = [], [], []
    for path in sorted(folder.glob("*.txt")):
        author = path.stem.split("_")[0]
        text = path.read_text(encoding="utf-8", errors="replace")
        documents.append(text)
        labels.append(author)
        titles.append(path.stem)
    return documents, labels, titles


def build_reverse_index(documents, n=3, verbose=True):
    """
    For every n-gram in every category, which documents (by index)
    contain it at least once, ANYWHERE in their full text?

    Returns: dict {category: {ngram: set(doc_indices)}}

    This is the same full-corpus scan vectorizer.build_vocabulary()
    already does -- just keeping WHICH documents each n-gram came from,
    instead of collapsing straight to a total count.
    """
    reverse_index = {cat: {} for cat in CATEGORIES}
    for i, text in enumerate(documents):
        if verbose:
            print(f"  Indexing document {i + 1}/{len(documents)}...", end="\r")
        doc_counts = extract_ngram_categories(text, n=n)
        for cat in CATEGORIES:
            for ngram in doc_counts[cat]:
                reverse_index[cat].setdefault(ngram, set()).add(i)
    if verbose:
        print()
    return reverse_index


def pick_example_indices(labels, n_examples, seed=0):
    """
    One novel per distinct author, in corpus order, up to n_examples --
    gives a spread across authors instead of several examples that are
    all from the same one.
    """
    seen_authors = set()
    picked = []
    for i, label in enumerate(labels):
        if label not in seen_authors:
            picked.append(i)
            seen_authors.add(label)
        if len(picked) >= n_examples:
            break
    return picked


def inspect_window(doc_index, documents, labels, titles, reverse_index,
                    window_size, offset, n, categories, top_k):
    """
    Prints the top n-grams (by in-window count) for one example novel's
    window, per requested category, alongside how many OTHER books --
    by the same author, and by other authors -- each n-gram appears in
    anywhere in the corpus.

    Returns (flagged_count, shown_count) for this novel, so the caller
    can roll up a corpus-wide summary.
    """
    text = documents[doc_index]
    window = text[offset:offset + window_size]
    author = labels[doc_index]
    title = titles[doc_index] if titles else f"doc {doc_index}"

    print(f"\n=== {title}  (author: {author})  "
          f"window [{offset}:{offset + window_size}] ===")

    if len(window) < window_size:
        print(f"  (novel too short for this window -- only {len(window)} "
              f"chars available)")
        return 0, 0

    window_counts = extract_ngram_categories(window, n=n)

    flagged_total = 0
    shown_total = 0

    for cat in categories:
        tag = "  (contrast category -- shouldn't show the effect)" \
            if cat == CONTRAST_CATEGORY else ""
        print(f"\n  {cat}{tag}")
        top = window_counts[cat].most_common(top_k)
        if not top:
            print("    (no n-grams of this category in this window)")
            continue

        for ngram, in_window_count in top:
            containing_docs = reverse_index[cat].get(ngram, set()) - {doc_index}
            same_author = sum(1 for j in containing_docs if labels[j] == author)
            other_author = len(containing_docs) - same_author

            flag = ""
            if same_author == 0 and other_author <= 1:
                flag = "  <-- barely appears anywhere else in the corpus"
                flagged_total += 1
            shown_total += 1

            print(f"    {ngram!r:10s} in-window={in_window_count:<4d} "
                  f"same-author other books={same_author:<3d} "
                  f"other-author books={other_author:<3d}{flag}")

    return flagged_total, shown_total


def run_leakage_inspection(documents, labels, titles=None, window_size=500,
                            offset=0, n=3, n_examples=4, top_k=6,
                            categories=None, seed=0):
    if categories is None:
        categories = WORD_AFFIX_CATEGORIES_OF_INTEREST + [CONTRAST_CATEGORY]

    print(f"Building corpus-wide n-gram index ({len(documents)} documents, n={n})...")
    reverse_index = build_reverse_index(documents, n=n)

    example_indices = pick_example_indices(labels, n_examples, seed=seed)

    total_flagged, total_shown = 0, 0
    for idx in example_indices:
        flagged, shown = inspect_window(idx, documents, labels, titles,
                                         reverse_index, window_size, offset,
                                         n, categories, top_k)
        total_flagged += flagged
        total_shown += shown

    if total_shown:
        print(f"\n\nSummary: {total_flagged}/{total_shown} top n-grams shown "
              f"across all examples appear in 0 other books by the same "
              f"author and at most 1 book by any other author -- i.e. are "
              f"essentially confined to a single novel.")
        print(f"If those flags cluster in {WORD_AFFIX_CATEGORIES_OF_INTEREST} "
              f"and are rare/absent in '{CONTRAST_CATEGORY}', that supports "
              f"the content-leakage hypothesis. If they're scattered evenly "
              f"across all categories instead, something else is going on.")


def main():
    ap = argparse.ArgumentParser(
        description="Diagnostic: check whether small, early-novel windows lean "
                    "on book-specific content (e.g. character names) rather "
                    "than general authorial style."
    )
    ap.add_argument("corpus_folder", help="Folder of Author_Title.txt novel files")
    ap.add_argument("--window-size", type=int, default=500,
                     help="Window size in characters (default 500 -- matches "
                          "the size where the effect showed up strongest)")
    ap.add_argument("--offset", type=int, default=0,
                     help="Starting character offset (default 0)")
    ap.add_argument("--n", type=int, default=3, help="n-gram length (default 3)")
    ap.add_argument("--n-examples", type=int, default=4,
                     help="Number of example novels to inspect, one per "
                          "distinct author (default 4)")
    ap.add_argument("--top-k", type=int, default=6,
                     help="Top-K n-grams to show per category (default 6)")
    ap.add_argument("--categories", default=None,
                     help="Comma-separated category names to inspect (default: "
                          "prefix,suffix,whole-word,mid-word,multi-word,beg-punct)")
    ap.add_argument("--seed", type=int, default=0,
                     help="Seed for picking example novels (default 0)")
    args = ap.parse_args()

    documents, labels, titles = load_corpus_with_titles(args.corpus_folder)
    print(f"Loaded {len(documents)} novels across {len(set(labels))} authors.")

    categories = args.categories.split(",") if args.categories else None

    run_leakage_inspection(documents, labels, titles=titles,
                            window_size=args.window_size, offset=args.offset,
                            n=args.n, n_examples=args.n_examples,
                            top_k=args.top_k, categories=categories,
                            seed=args.seed)


if __name__ == "__main__":
    main()