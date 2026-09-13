"""
Diagnostic: what does the actual distribution of raw dash counts per local
window look like across the whole corpus? Settles, directly, whether
dash_rate's floor value (3 vs 5 vs anything else) is the bottleneck, or
whether the corpus itself just doesn't use dashes densely enough at
WINDOW_SIZE words per window for this feature to be testable at all.

Run from the same folder as run_full_corpus.py (needs windowing.py and
features.py importable next to it):

    python diagnose_dash_counts.py
"""
import os
import re
from collections import Counter

from windowing import load_text, parse_sentences, local_windows
from features import extract_features

CORPUS_DIR = 'real_corpus'
WINDOW_SIZE = 300  # must match run_full_corpus.py's WINDOW_SIZE
FNAME_RE = re.compile(r'^([A-Za-z]+)_(.+)_(\d{4})\.txt$')


def main():
    all_counts = []
    per_author_max = {}

    for fn in sorted(os.listdir(CORPUS_DIR)):
        m = FNAME_RE.match(fn)
        if not m:
            continue
        author = m.group(1)
        text = load_text(os.path.join(CORPUS_DIR, fn))
        sents = parse_sentences(text)
        windows = local_windows(sents, WINDOW_SIZE)
        counts = [extract_features(w).get('dash_count', 0) for w in windows]
        if not counts:
            continue
        all_counts.extend(counts)
        per_author_max[author] = max(per_author_max.get(author, 0), max(counts))

    if not all_counts:
        print("No windows found -- check CORPUS_DIR / filename pattern match run_full_corpus.py.")
        return

    all_counts.sort()
    n = len(all_counts)
    print(f"Total local windows across corpus: {n}")
    print(f"Min / median / max dash_count per window: {all_counts[0]} / {all_counts[n // 2]} / {all_counts[-1]}")

    dist = Counter(all_counts)
    print("\nExact dash count -> number of windows with that many dashes:")
    for k in range(0, 11):
        print(f"  {k:2d} dashes: {dist.get(k, 0)} windows")
    print(f"  11+ dashes: {sum(v for k, v in dist.items() if k > 10)} windows")

    print("\nWhat % of all windows would pass each candidate floor:")
    for thresh in [1, 2, 3, 4, 5, 10]:
        pct = sum(1 for c in all_counts if c >= thresh) / n * 100
        print(f"  >= {thresh} dashes: {pct:.3f}% of windows ({sum(1 for c in all_counts if c >= thresh)} windows)")

    print("\nSingle highest dash_count seen in any one window, per author (sorted):")
    for author, m in sorted(per_author_max.items(), key=lambda kv: -kv[1]):
        print(f"  {author:15s} {m}")


if __name__ == '__main__':
    main()