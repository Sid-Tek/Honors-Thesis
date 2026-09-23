"""
Book-by-book burst map.

Step 1 -- every test book x every feature: is that book's opening a burst?
  Each book is scored on its own (full length). For each feature, the
  opening (first 2 windows, ~600 words -- same as burst_test) is ranked
  against every same-length stretch of that book, on the detrended curve
  (stats_tests.opening_rank_p). Burst = opening in the top 5% of its book.

Step 2 -- pair each author's two test books.
  For every author: which burst features the two books SHARE, and which
  appear in only one of them.

Step 3 -- consistent across the corpus, or author-dependent?
  Per feature:
    * how many of the 66 books burst on it, vs ~5% (3.3) by chance
      (one-sided binomial test, BH-FDR across the 24 features) -> does
      this feature tend to burst at openings in general?
    * in how many authors BOTH books burst on it -> author habit
  Per author:
    * shared bursts vs the number expected if the two books picked their
      burst features independently (hypergeometric: a*b/m) -> is this
      author's opening consistent from book to book?
  Overall: total shared bursts vs expected, with a permutation test that
  keeps every book's own burst count fixed but breaks the link between
  an author's two books (re-pairs books across authors).

Outputs (output_full/burst_map/):
  book_feature_bursts.csv   every book x feature: rank p, percentile, burst
  author_pairs.csv          every author: shared / only-book-A / only-book-B
  feature_consistency.csv   every feature: #books, #authors both, test
  burst_map.png             heatmap: books (grouped by author) x features
  burst_map_report.md       readable summary of all of the above

Run: python book_burst_map.py   (~1 min)
"""
import math
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from features import extract_features, FEATURE_NAMES
from scoring import corpus_stats, author_profile, distinctiveness_scores
from stats_tests import opening_rank_p, bh_fdr
from run_full_corpus import discover_authors, book_local_windows, average_profiles, OUT_DIR, FNAME_RE

BOOK_CUT = 0.05          # burst = opening in the top 5% of its own book
N_BURST_WINDOWS = 2      # ~600 words, same opening as burst_test
MAP_DIR = os.path.join(OUT_DIR, 'burst_map')
N_PERM = 20000

INK = '#333333'
MUTED = '#8a8a8a'
C_NONE = '#f1f1ef'
C_ONE = '#9ec5f4'
C_BOTH = '#1c5cab'
C_NA = '#ffffff'


def book_label(fn):
    m = FNAME_RE.match(fn)
    return f"{m.group(2)} ({m.group(3)})" if m else fn


def binom_sf(k, n, p):
    """P(X >= k) for X ~ Binomial(n, p)."""
    return float(sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def score_books():
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
        for slot, fn in enumerate(test_files[author]):
            scores = [distinctiveness_scores(extract_features(w), profiles[author], background, stats)
                      for w in book_local_windows(fn)]
            for feat in FEATURE_NAMES:
                curve = np.array([s.get(feat, np.nan) for s in scores])
                p = opening_rank_p(curve, N_BURST_WINDOWS)
                rows.append({'author': author, 'book_slot': 'A' if slot == 0 else 'B',
                             'book': fn, 'title': book_label(fn), 'feature': feat,
                             'n_windows': len(curve), 'rank_p': p,
                             'opening_percentile': np.nan if np.isnan(p) else 1 - p,
                             'burst': bool(p <= BOOK_CUT) if not np.isnan(p) else np.nan})
        print(f"  {author} scored")
    return pd.DataFrame(rows)


def author_pairs(bk):
    rows = []
    for author, g in bk.groupby('author'):
        a = g[g['book_slot'] == 'A']
        b = g[g['book_slot'] == 'B']
        testable = set(a.loc[a['rank_p'].notna(), 'feature']) & set(b.loc[b['rank_p'].notna(), 'feature'])
        fa = set(a.loc[a['burst'] == True, 'feature']) & testable
        fb = set(b.loc[b['burst'] == True, 'feature']) & testable
        m = len(testable)
        shared, only_a, only_b = fa & fb, fa - fb, fb - fa
        expected = len(fa) * len(fb) / m if m else np.nan
        # P(overlap >= observed) under hypergeometric (book B's bursts drawn at random)
        k, na, nb = len(shared), len(fa), len(fb)
        p_hyper = sum(math.comb(na, i) * math.comb(m - na, nb - i) for i in range(k, min(na, nb) + 1)) \
            / math.comb(m, nb) if m and nb <= m else np.nan
        rows.append({'author': author,
                     'book_A': a['title'].iloc[0], 'book_B': b['title'].iloc[0],
                     'n_features_tested': m,
                     'n_bursts_A': na, 'n_bursts_B': nb,
                     'n_shared': k, 'expected_shared_by_chance': round(expected, 2),
                     'p_overlap': p_hyper,
                     'jaccard': (k / len(fa | fb)) if (fa | fb) else np.nan,
                     'shared': ', '.join(sorted(shared)),
                     'only_A': ', '.join(sorted(only_a)),
                     'only_B': ', '.join(sorted(only_b))})
    return pd.DataFrame(rows).sort_values(['n_shared', 'n_bursts_A'], ascending=False)


def feature_consistency(bk, pairs):
    n_authors = bk['author'].nunique()
    rows = []
    for feat, g in bk.groupby('feature'):
        t = g[g['rank_p'].notna()]
        n_books, k_books = len(t), int((t['burst'] == True).sum())
        per_author = t.groupby('author')['burst'].agg(lambda s: int((s == True).sum()))
        both = sorted(per_author[per_author == 2].index)
        one = sorted(per_author[per_author == 1].index)
        rows.append({'feature': feat, 'n_books_tested': n_books, 'n_books_burst': k_books,
                     'pct_books_burst': 100 * k_books / n_books if n_books else np.nan,
                     'expected_books_by_chance': BOOK_CUT * n_books,
                     'p_more_than_chance': binom_sf(k_books, n_books, BOOK_CUT) if n_books else np.nan,
                     'n_authors_both_books': len(both), 'n_authors_one_book': len(one),
                     'authors_both_books': ', '.join(both),
                     'authors_one_book': ', '.join(one)})
    df = pd.DataFrame(rows)
    df['q_more_than_chance'] = bh_fdr(df['p_more_than_chance'].values)
    return df.sort_values(['n_books_burst', 'n_authors_both_books'], ascending=False)


def overlap_permutation(bk, seed=0):
    """Keep every book's burst set; re-pair books ACROSS authors at random.
    If an author's two books share more bursts than random book pairs do,
    burst features are an author habit rather than one-off book quirks."""
    feats = list(FEATURE_NAMES)
    authors = sorted(bk['author'].unique())
    mats = {}
    for slot in ('A', 'B'):
        sub = bk[bk['book_slot'] == slot].pivot(index='author', columns='feature', values='burst')
        mats[slot] = sub.reindex(index=authors, columns=feats).fillna(False).astype(bool).values
    A, B = mats['A'], mats['B']
    observed = int((A & B).sum())
    rng = np.random.default_rng(seed)
    null = np.empty(N_PERM, dtype=int)
    for i in range(N_PERM):
        null[i] = int((A & B[rng.permutation(len(authors))]).sum())
    p = (1 + np.sum(null >= observed)) / (N_PERM + 1)
    return observed, float(null.mean()), float(np.percentile(null, 95)), float(p)


def plot_map(bk, pairs, fc, path):
    feats = fc['feature'].tolist()                    # most-bursting features first
    authors = pairs['author'].tolist()                # most-consistent authors first
    row_keys, row_labels, author_edges = [], [], []
    for a in authors:
        for slot in ('A', 'B'):
            t = bk[(bk['author'] == a) & (bk['book_slot'] == slot)]['title'].iloc[0]
            row_keys.append((a, slot))
            row_labels.append(f"{a} — {t}")
        author_edges.append(len(row_keys))
    lookup = bk.set_index(['author', 'book_slot', 'feature'])
    grid = np.zeros((len(row_keys), len(feats)))
    for i, (a, slot) in enumerate(row_keys):
        other = 'B' if slot == 'A' else 'A'
        for j, f in enumerate(feats):
            here = lookup.loc[(a, slot, f), 'burst']
            there = lookup.loc[(a, other, f), 'burst']
            if pd.isna(here):
                grid[i, j] = 3
            elif here is True or here == True:
                grid[i, j] = 2 if there == True else 1
            else:
                grid[i, j] = 0
    cmap = ListedColormap([C_NONE, C_ONE, C_BOTH, C_NA])
    fig, ax = plt.subplots(figsize=(11, 17))
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=3, aspect='auto', interpolation='nearest')
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=60, ha='left', rotation_mode='anchor', fontsize=8, color=INK)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=6.5, color=INK)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.set_xticks(np.arange(-0.5, len(feats), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_keys), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.2)
    for e in author_edges[:-1]:
        ax.axhline(e - 0.5, color=MUTED, linewidth=0.9)
    ax.tick_params(which='both', length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    counts = fc.set_index('feature')
    for j, f in enumerate(feats):
        ax.text(j, len(row_keys) - 0.1, f"{int(counts.loc[f, 'n_books_burst'])}", ha='center', va='top',
                fontsize=7, color=INK)
    ax.text(-0.7, len(row_keys) - 0.1, 'books bursting (of 66; ~3 by chance) →', ha='right', va='top',
            fontsize=7, color=INK)
    ax.legend(handles=[Patch(color=C_BOTH, label="burst in BOTH of the author's books"),
                       Patch(color=C_ONE, label='burst in this book only'),
                       Patch(facecolor=C_NONE, edgecolor=MUTED, label='no burst'),
                       Patch(facecolor=C_NA, edgecolor=MUTED, label='not testable')],
              loc='upper left', bbox_to_anchor=(0, -0.035), ncol=4, frameon=False, fontsize=8)
    ax.set_title('Opening bursts, book by book  (burst = opening in the top 5% of its own book, ~600 words)\n'
                 'Features: most-bursting first.  Authors: most shared bursts first.',
                 loc='left', fontsize=10, color=INK, pad=70)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def write_report(bk, pairs, fc, perm, path):
    n_books = bk['book'].nunique()
    tested = bk[bk['rank_p'].notna()]
    total_bursts = int((tested['burst'] == True).sum())
    observed, null_mean, null95, p_perm = perm
    L = []
    L.append('# Book-by-book burst map')
    L.append('')
    L.append(f'Every test book ({n_books} books: 33 authors × 2) was checked on every feature (24). '
             f'A book "bursts" on a feature when its opening ~600 words rank in the top 5% of every '
             f'same-length stretch of that book (detrended; see `stats_tests.opening_rank_p`).')
    L.append('')
    L.append('## Headline numbers')
    L.append('')
    L.append(f'- Book-level bursts: **{total_bursts}** of {len(tested)} book × feature checks '
             f'(~{BOOK_CUT * len(tested):.0f} expected by chance).')
    L.append(f'- Shared by both of an author\'s books: **{observed}** author × feature pairs. '
             f'If books were paired at random across authors, the average would be {null_mean:.1f} '
             f'(95th percentile {null95:.0f}); p = {p_perm:.4f}.')
    n_auth_shared = int((pairs['n_shared'] > 0).sum())
    L.append(f'- Authors with at least one shared burst: **{n_auth_shared} of {len(pairs)}**. '
             f'Authors with no burst in either book: {int(((pairs["n_bursts_A"] + pairs["n_bursts_B"]) == 0).sum())}.')
    sig = fc[fc['q_more_than_chance'] < 0.05]
    L.append(f'- Features that burst in more books than chance across the corpus (q < 0.05): '
             f'**{len(sig)}**' + (f' ({", ".join(sig["feature"])})' if len(sig) else '') + '.')
    L.append('')
    L.append('## By feature: does any feature burst consistently?')
    L.append('')
    L.append('| feature | books bursting (of ~66; ~3 by chance) | q (more than chance) | authors: both books | authors: one book |')
    L.append('|---|---:|---:|---|---|')
    for _, r in fc.iterrows():
        both = r['authors_both_books'] or '–'
        L.append(f"| {r['feature']} | {int(r['n_books_burst'])} | {r['q_more_than_chance']:.3f} | "
                 f"{both} | {r['authors_one_book'] or '–'} |")
    L.append('')
    L.append('## By author: do the two books agree?')
    L.append('')
    L.append('"Expected" = shared bursts if the two books picked their burst features independently. '
             'p = chance of sharing at least this many.')
    L.append('')
    L.append('| author | book A | book B | bursts A | bursts B | shared | expected | p | shared features | only A | only B |')
    L.append('|---|---|---|---:|---:|---:|---:|---:|---|---|---|')
    for _, r in pairs.iterrows():
        L.append(f"| {r['author']} | {r['book_A']} | {r['book_B']} | {r['n_bursts_A']} | {r['n_bursts_B']} | "
                 f"{r['n_shared']} | {r['expected_shared_by_chance']:.2f} | {r['p_overlap']:.3f} | "
                 f"{r['shared'] or '–'} | {r['only_A'] or '–'} | {r['only_B'] or '–'} |")
    L.append('')
    L.append('Files: `book_feature_bursts.csv` (every book × feature, with rank p and percentile, so a '
             'different cut-off can be applied), `author_pairs.csv`, `feature_consistency.csv`, `burst_map.png`.')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')


def main():
    os.makedirs(MAP_DIR, exist_ok=True)
    bk = score_books()
    bk.to_csv(os.path.join(MAP_DIR, 'book_feature_bursts.csv'), index=False)
    pairs = author_pairs(bk)
    pairs.to_csv(os.path.join(MAP_DIR, 'author_pairs.csv'), index=False)
    fc = feature_consistency(bk, pairs)
    fc.to_csv(os.path.join(MAP_DIR, 'feature_consistency.csv'), index=False)
    perm = overlap_permutation(bk)
    plot_map(bk, pairs, fc, os.path.join(MAP_DIR, 'burst_map.png'))
    write_report(bk, pairs, fc, perm, os.path.join(MAP_DIR, 'burst_map_report.md'))
    pd.set_option('display.width', 220)
    print(fc[['feature', 'n_books_burst', 'q_more_than_chance', 'n_authors_both_books', 'n_authors_one_book']].to_string(index=False))
    print(pairs[['author', 'n_bursts_A', 'n_bursts_B', 'n_shared', 'expected_shared_by_chance', 'p_overlap', 'shared']].to_string(index=False))
    print('overlap permutation (observed, null mean, null 95th, p):', perm)


if __name__ == '__main__':
    main()
