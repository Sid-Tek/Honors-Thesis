"""
Follow-up diagnostic: for each of the 6 authors who use dashes at all, does
their REFERENCE book (earliest-published -- the one run_full_corpus.py uses
to build their own_profile) actually contain any dash-heavy windows, or do
their dashes live entirely in the LATER books used as test material?

This directly tests the hypothesis that dash_rate's own_profile is NaN for
every author because the specific held-out reference book happens to be
dash-light, even for authors who use dashes elsewhere in their career --
not because the window-level floor itself is too strict.

Run from the same folder as run_full_corpus.py:
    python diagnose_dash_reference_vs_test.py
"""
import os
import re
from collections import defaultdict

from windowing import load_text, parse_sentences, local_windows
from features import extract_features

CORPUS_DIR = 'real_corpus'
WINDOW_SIZE = 300  # must match run_full_corpus.py
FNAME_RE = re.compile(r'^([A-Za-z]+)_(.+)_(\d{4})\.txt$')

DASH_USERS = {'Corelli', 'Ward', 'Trollope', 'Chesterton', 'Ford', 'Doyle'}


def discover_authors():
    by_author = defaultdict(list)
    for fn in sorted(os.listdir(CORPUS_DIR)):
        m = FNAME_RE.match(fn)
        if not m:
            continue
        author, title, year = m.groups()
        by_author[author].append((int(year), fn))
    for author in by_author:
        by_author[author].sort()
    return by_author


def dash_stats(filename):
    text = load_text(os.path.join(CORPUS_DIR, filename))
    sents = parse_sentences(text)
    windows = local_windows(sents, WINDOW_SIZE)
    counts = [extract_features(w).get('dash_count', 0) for w in windows]
    if not counts:
        return {'n_windows': 0, 'max': 0, 'n_ge3': 0, 'n_ge5': 0}
    return {
        'n_windows': len(counts),
        'max': max(counts),
        'n_ge3': sum(1 for c in counts if c >= 3),
        'n_ge5': sum(1 for c in counts if c >= 5),
    }


def main():
    by_author = discover_authors()
    for author in sorted(DASH_USERS):
        if author not in by_author:
            print(f"{author}: not found in corpus")
            continue
        books = by_author[author]
        ref_year, ref_file = books[0]
        print(f"\n=== {author} ===")
        ref = dash_stats(ref_file)
        print(f"  REFERENCE ({ref_file}, {ref_year}): {ref['n_windows']} windows, "
              f"max={ref['max']}, windows>=3={ref['n_ge3']}, windows>=5={ref['n_ge5']}")
        for year, fn in books[1:]:
            t = dash_stats(fn)
            print(f"  test book  ({fn}, {year}): {t['n_windows']} windows, "
                  f"max={t['max']}, windows>=3={t['n_ge3']}, windows>=5={t['n_ge5']}")


if __name__ == '__main__':
    main()