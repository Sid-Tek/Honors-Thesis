"""
Per-book consistency check for author-level bursts (the red rings on
output_full/plots/master_scatter_all_points.png).

The rings come from each author's curve AVERAGED across their two test
books. A ring can therefore mean either
  (a) the author habitually opens this way -- both books burst, or
  (b) one book has an unusual opening that pulls the average up.
This script scores each test book on its own (full length, not truncated
to the shorter book) and asks, for every author x feature pair: where does
THIS book's opening rank among every same-length stretch of THIS book?
(stats_tests.opening_rank_p -- detrended, autocorrelation-safe.)

Verdict for each pair, using a 5% cut per book (opening in the top 5%):
  both books      -- consistent: likely the author's habit
  one book        -- driven by a single book
  neither         -- the average-level ring doesn't survive splitting

Chance baseline: the same test is run on ALL author x feature pairs, so
"how many pairs burst in both books" can be compared with what the whole
corpus shows (under no opening effect: ~0.05 x 0.05 = 0.25% of pairs).

Output: output_full/per_book_bursts.csv   (every pair, both books)
        output_full/per_book_rings.md     (the ringed pairs, readable)
"""
import os

import numpy as np
import pandas as pd

from features import extract_features, FEATURE_NAMES
from scoring import corpus_stats, author_profile, distinctiveness_scores
from stats_tests import opening_rank_p
from run_full_corpus import discover_authors, book_local_windows, average_profiles, OUT_DIR

BOOK_CUT = 0.05   # opening in the top 5% of that book


def main():
    by_author = discover_authors()
    ref_feats, test_files, all_ref = {}, {}, []
    for author, books in by_author.items():
        test_files[author] = [fn for (_, fn) in books[1:]]
        feats = [extract_features(w) for w in book_local_windows(books[0][1])]
        ref_feats[author] = feats
        all_ref.extend(feats)
    stats = corpus_stats(all_ref)
    profiles = {a: author_profile(f, stats) for a, f in ref_feats.items()}

    rows = []
    for author in [a for a in by_author if test_files[a]]:
        background = average_profiles([p for a, p in profiles.items() if a != author])
        for fn in test_files[author]:
            scores = [distinctiveness_scores(extract_features(w), profiles[author], background, stats)
                      for w in book_local_windows(fn)]
            for feat in FEATURE_NAMES:
                curve = np.array([s.get(feat, np.nan) for s in scores])
                rows.append({'author': author, 'feature': feat, 'book': fn,
                             'n_windows': len(curve), 'rank_p': opening_rank_p(curve)})
        print(f"  {author} done")

    books = pd.DataFrame(rows)
    books['book_burst'] = books['rank_p'] <= BOOK_CUT
    wide = (books.groupby(['author', 'feature'])
                 .agg(n_books=('book', 'size'), books=('book', ' / '.join),
                      rank_p_each=('rank_p', lambda s: ' / '.join(f'{v:.3f}' for v in s)),
                      n_books_burst=('book_burst', 'sum'),
                      n_books_tested=('rank_p', lambda s: int(s.notna().sum())))
                 .reset_index())

    res = pd.read_csv(os.path.join(OUT_DIR, 'author_feature_results.csv'))
    wide = wide.merge(res[['author', 'feature', 'opening_rank_p', 'author_level_burst', 'burst_p']],
                      on=['author', 'feature'], how='left')

    def verdict(r):
        if r['n_books_tested'] < 2:
            return 'not testable'
        return {2: 'both books', 1: 'one book', 0: 'neither'}[int(r['n_books_burst'])]
    wide['per_book_verdict'] = wide.apply(verdict, axis=1)
    wide.to_csv(os.path.join(OUT_DIR, 'per_book_bursts.csv'), index=False)

    testable = wide[wide['per_book_verdict'] != 'not testable']
    rings = wide[wide['author_level_burst'] == True].copy()
    sep18 = pd.read_csv(os.path.join('past_outputs', '9_23_before_chunk_null', 'author_feature_results.csv')) \
        if os.path.exists(os.path.join('past_outputs', '9_23_before_chunk_null', 'author_feature_results.csv')) else None
    if sep18 is not None:
        orig = set(zip(sep18.loc[sep18['burst_q'] < 0.05, 'author'], sep18.loc[sep18['burst_q'] < 0.05, 'feature']))
        rings['original_7'] = [(a, f) in orig for a, f in zip(rings['author'], rings['feature'])]
    else:
        rings['original_7'] = False
    rings = rings.sort_values(['per_book_verdict', 'opening_rank_p'])

    all_both = int((testable['per_book_verdict'] == 'both books').sum())
    lines = [
        '# Per-book check of the 22 author-level bursts',
        '',
        f'Each ringed pair re-scored on each test book separately. "Burst in a book" = that '
        f"book's opening is in the top {BOOK_CUT:.0%} of every same-length stretch of that book.",
        '',
        '| verdict | ringed pairs |',
        '|---|---:|',
    ]
    for v in ('both books', 'one book', 'neither', 'not testable'):
        lines.append(f'| {v} | {int((rings["per_book_verdict"] == v).sum())} |')
    lines += [
        '',
        f'Chance baseline: across ALL {len(testable)} testable author x feature pairs, '
        f'{all_both} burst in both books (~{0.0025 * len(testable):.0f} expected if openings were random).',
        '',
        '| author | feature | verdict | opening rank p, each book | averaged rank p | in original 7 |',
        '|---|---|---|---|---:|:---:|',
    ]
    for _, r in rings.iterrows():
        lines.append(f"| {r['author']} | {r['feature']} | {r['per_book_verdict']} | {r['rank_p_each']} | "
                     f"{r['opening_rank_p']:.4f} | {'yes' if r['original_7'] else ''} |")
    with open(os.path.join(OUT_DIR, 'per_book_rings.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
