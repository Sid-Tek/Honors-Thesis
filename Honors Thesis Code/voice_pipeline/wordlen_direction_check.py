"""
Long-word vs short-word check: are word-length bursts at openings a
DESCRIPTIVE-OPENING convention, or each author's VOICE turned up?

Background: a "burst" (book_burst_map.py) means the opening leans harder
toward the author's OWN habit than the rest of the book does. For word
length, the author's habit can point either way:
  long-word authors  -- use longer words than the other authors
  short-word authors -- use shorter words than the other authors

Two explanations make opposite predictions:
  A. Descriptive openings. Openings describe places/people, and description
     uses longer words for everyone. -> In RAW terms, openings have longer
     words in BOTH groups. In "burst" terms, long-word authors look like
     they burst; short-word authors look like the opposite (their opening
     moves AWAY from their habit).
  B. Voice amplified. Each author leans harder into their own habit.
     -> Long-word authors open with even longer words, short-word authors
     with even shorter words. Both groups burst; in raw terms the groups
     move in opposite directions.

Features checked: avg_word_len, syll_per_word, pct_monosyllabic (all word
length -- for pct_monosyllabic, FEWER one-syllable words = longer words),
plus fw_of (the other feature that bursts often; "more 'of'" plays the role
of "longer words").

How: each author's direction on each feature (own profile vs. background,
exactly as scoring.py uses it) is attached to book_feature_bursts.csv. The
opening percentile there is author-signed; flipping it for authors whose
direction is negative gives the RAW percentile (0.9 = the opening has
longer words than 90% of same-length stretches in that book). This is
exact: the distinctiveness score is a linear function of the raw feature
value with sign = the author's direction, and the detrending is linear.

Tests (authors, not books, are the unit -- each author's two books are
averaged first so one author can't count twice):
  1. Raw direction, all authors: is the mean raw opening percentile above
     0.5? (Wilcoxon signed-rank vs 0.5, one-sided.) A predicts yes.
  2. Group difference: do long-word and short-word authors differ in raw
     opening percentile? (Mann-Whitney, two-sided.) B predicts yes, with
     long-word > 0.5 and short-word < 0.5; A predicts no difference.
  3. Burst counts per group (signed percentile >= 0.95) and "anti-bursts"
     (signed percentile <= 0.05, i.e. the opening moves away from the
     author's habit).
  4. Title/heading check: the first window of each book contains the title
     and chapter heading (front matter isn't trimmed), and many titles
     contain "of". So the raw test is repeated for the ~600-word stretch
     starting at window 0, 1, 2 and 5 (0 / 300 / 600 / 1,500 words in). If
     the effect survives skipping the first window, titles aren't causing it.

Run book_burst_map.py first. Then: python wordlen_direction_check.py

Outputs (output_full/wordlen_check/):
  book_directions.csv    every book x checked feature: group, signed and raw percentile
  group_summary.csv      per feature x group: authors, mean raw percentile, bursts, anti-bursts
  tests.csv              tests 1 and 2 per feature
  skip_check.csv         test 4: effect when the "opening" starts later
  wordlen_check.png      raw opening percentile per book, coloured by group
  wordlen_check_report.md
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, mannwhitneyu

from features import extract_features, FEATURE_NAMES
from scoring import corpus_stats, author_profile
from run_full_corpus import discover_authors, book_local_windows, average_profiles, OUT_DIR

MAP_CSV = os.path.join(OUT_DIR, 'burst_map', 'book_feature_bursts.csv')
CHECK_DIR = os.path.join(OUT_DIR, 'wordlen_check')

# feature -> +1 if a HIGHER raw value means "longer words" / "more of"
CHECKED = {'avg_word_len': +1, 'syll_per_word': +1, 'pct_monosyllabic': -1, 'fw_of': +1}
LABELS = {'avg_word_len': 'average word length', 'syll_per_word': 'syllables per word',
          'pct_monosyllabic': 'share of one-syllable words', 'fw_of': '"of"'}
BURST = 0.95

INK = '#333333'
MUTED = '#8a8a8a'
C_LONG = '#2a78d6'
C_SHORT = '#eb6834'


def author_directions():
    """sign(own_profile - background) per author per feature -- the same
    'direction' scoring.distinctiveness_scores uses."""
    by_author = discover_authors()
    ref, all_ref = {}, []
    for author, books in by_author.items():
        feats = [extract_features(w) for w in book_local_windows(books[0][1])]
        ref[author] = feats
        all_ref.extend(feats)
    stats = corpus_stats(all_ref)
    prof = {a: author_profile(f, stats) for a, f in ref.items()}
    rows = []
    for author in by_author:
        if len(by_author[author]) < 2:
            continue
        bg = average_profiles([p for a, p in prof.items() if a != author])
        for f in CHECKED:
            gap = prof[author][f] - bg[f]
            rows.append({'author': author, 'feature': f, 'profile_gap_z': gap,
                         'direction': 0 if abs(gap) < 1e-9 else int(np.sign(gap))})
    return pd.DataFrame(rows)


def main():
    if not os.path.exists(MAP_CSV):
        raise SystemExit('Run book_burst_map.py first (needs output_full/burst_map/book_feature_bursts.csv).')
    os.makedirs(CHECK_DIR, exist_ok=True)
    books = pd.read_csv(MAP_CSV)
    books = books[books['feature'].isin(CHECKED)].copy()
    d = author_directions()
    books = books.merge(d, on=['author', 'feature'], how='left')

    # Group = the author's word-length habit, defined once from avg_word_len
    # so every feature uses the same two groups.
    wl = d[d['feature'] == 'avg_word_len'].set_index('author')['direction']
    books['group'] = books['author'].map({a: ('long-word' if v > 0 else 'short-word') for a, v in wl.items()})

    books['signed_pct'] = books['opening_percentile']
    # raw percentile of the raw feature value; flip if author direction < 0
    books['raw_pct'] = np.where(books['direction'] >= 0, books['signed_pct'], 1 - books['signed_pct'])
    # re-orient so 1.0 always means "longer words" / "more of"
    books['longer_pct'] = np.where(books['feature'].map(CHECKED) > 0, books['raw_pct'], 1 - books['raw_pct'])
    books['burst'] = books['signed_pct'] >= BURST
    books['anti_burst'] = books['signed_pct'] <= 1 - BURST
    books.to_csv(os.path.join(CHECK_DIR, 'book_directions.csv'), index=False)

    per_author = (books.groupby(['feature', 'author', 'group'])
                       .agg(longer_pct=('longer_pct', 'mean'), signed_pct=('signed_pct', 'mean'))
                       .reset_index())

    summ, tests = [], []
    for f in CHECKED:
        pa = per_author[(per_author['feature'] == f) & per_author['longer_pct'].notna()]
        bk = books[(books['feature'] == f) & books['longer_pct'].notna()]
        for g in ('long-word', 'short-word'):
            s, b = pa[pa['group'] == g], bk[bk['group'] == g]
            summ.append({'feature': f, 'group': g, 'n_authors': len(s), 'n_books': len(b),
                         'mean_longer_pct': s['longer_pct'].mean(),
                         'mean_signed_pct': s['signed_pct'].mean(),
                         'n_books_burst': int(b['burst'].sum()),
                         'n_books_anti_burst': int(b['anti_burst'].sum()),
                         'expected_each_by_chance': round(0.05 * len(b), 1)})
        x = pa['longer_pct'].values
        long_x = pa.loc[pa['group'] == 'long-word', 'longer_pct'].values
        short_x = pa.loc[pa['group'] == 'short-word', 'longer_pct'].values
        tests.append({'feature': f, 'n_authors': len(x),
                      'mean_longer_pct_all': x.mean(),
                      'p_openings_longer_all_authors': wilcoxon(x - 0.5, alternative='greater').pvalue,
                      'mean_longer_pct_long_word': long_x.mean(),
                      'mean_longer_pct_short_word': short_x.mean(),
                      'p_groups_differ': mannwhitneyu(long_x, short_x, alternative='two-sided').pvalue})
    summ = pd.DataFrame(summ)
    tests = pd.DataFrame(tests)
    summ.to_csv(os.path.join(CHECK_DIR, 'group_summary.csv'), index=False)
    tests.to_csv(os.path.join(CHECK_DIR, 'tests.csv'), index=False)

    sk = skip_check()
    sk.to_csv(os.path.join(CHECK_DIR, 'skip_check.csv'), index=False)
    plot(books, os.path.join(CHECK_DIR, 'wordlen_check.png'))
    report(summ, tests, wl, sk, os.path.join(CHECK_DIR, 'wordlen_check_report.md'))
    pd.set_option('display.width', 200)
    print(summ.to_string(index=False))
    print(tests.to_string(index=False))
    print(sk.to_string(index=False))


SKIP_STARTS = (0, 1, 2, 5)


def skip_check():
    """Raw 'longer words' percentile of the 2-window stretch starting at
    window s, per author (two test books averaged), on raw feature values."""
    per = {(f, s): [] for f in CHECKED for s in SKIP_STARTS}
    for author, books in discover_authors().items():
        if len(books) < 2:
            continue
        acc = {(f, s): [] for f in CHECKED for s in SKIP_STARTS}
        for _, fn in books[1:]:
            fe = [extract_features(w) for w in book_local_windows(fn)]
            for f, sign in CHECKED.items():
                y = np.array([d[f] for d in fe]) * sign
                x = np.arange(len(y))
                r = y - np.polyval(np.polyfit(x, y, 1), x)
                c = np.concatenate([[0.0], np.cumsum(r)])
                m = (c[2:] - c[:-2]) / 2
                for s in SKIP_STARTS:
                    acc[(f, s)].append(float(np.mean(m < m[s] - 1e-9)))
        for k, v in acc.items():
            per[k].append(np.mean(v))
    rows = []
    for (f, s), v in per.items():
        v = np.array(v)
        rows.append({'feature': f, 'start_window': s, 'start_word': s * 300, 'n_authors': len(v),
                     'mean_longer_pct': v.mean(),
                     'p_longer_than_typical': wilcoxon(v - 0.5, alternative='greater').pvalue})
    return pd.DataFrame(rows)


def plot(books, path):
    feats = list(CHECKED)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    rng = np.random.default_rng(1)
    for i, f in enumerate(feats):
        for g, col, off in (('long-word', C_LONG, -0.15), ('short-word', C_SHORT, 0.15)):
            b = books[(books['feature'] == f) & (books['group'] == g) & books['longer_pct'].notna()]
            y = i + off + rng.uniform(-0.07, 0.07, len(b))
            ax.scatter(b['longer_pct'], y, s=26, color=col, edgecolor='white', linewidth=0.8, zorder=3,
                       label=f'{g} authors' if i == 0 else None)
            ax.plot([b['longer_pct'].mean()] * 2, [i + off - 0.13, i + off + 0.13], color=col, lw=2.5, zorder=4)
    ax.axvline(0.5, color=MUTED, lw=1, ls='--')
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels([LABELS[f] for f in feats], color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel('Opening percentile within its own book  (1 = the opening has the LONGEST words / most "of")\n'
                  'dots = books, bars = group means', color=INK, fontsize=9)
    ax.set_title('Do openings use longer words for everyone, or exaggerate each author\'s habit?',
                 loc='left', color=INK, fontsize=11)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.xaxis.grid(True, color='#e6e6e6', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc='lower left', bbox_to_anchor=(0, -0.42), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def report(summ, tests, wl, sk, path):
    n_long, n_short = int((wl > 0).sum()), int((wl <= 0).sum())
    L = ['# Long-word vs short-word check', '',
         f'Authors split by their usual word length (from their reference book, vs. the other authors): '
         f'**{n_long} long-word**, **{n_short} short-word**.', '',
         '"Longer percentile" = where the opening ranks among every same-length stretch of its own book '
         '(1.0 = the opening has the longest words / most "of"; 0.5 = typical). Authors are the unit: '
         'each author\'s two books are averaged.', '',
         '## Tests', '',
         '| feature | mean, all authors | p (openings longer, all) | long-word authors | short-word authors | p (groups differ) |',
         '|---|---:|---:|---:|---:|---:|']
    for _, r in tests.iterrows():
        L.append(f"| {LABELS[r['feature']]} | {r['mean_longer_pct_all']:.2f} | {r['p_openings_longer_all_authors']:.3f} | "
                 f"{r['mean_longer_pct_long_word']:.2f} | {r['mean_longer_pct_short_word']:.2f} | {r['p_groups_differ']:.3f} |")
    L += ['', 'Prediction A (descriptive openings): both groups above 0.5, no group difference.',
          'Prediction B (voice amplified): long-word above 0.5, short-word below 0.5, groups differ.', '',
          '## Bursts and anti-bursts per group (books)', '',
          'Burst = opening leans toward the author\'s own habit (top 5% of its book). Anti-burst = opening '
          'leans away from it (bottom 5%).', '',
          '| feature | group | books | bursts | anti-bursts | ~by chance (each) |',
          '|---|---|---:|---:|---:|---:|']
    for _, r in summ.iterrows():
        L.append(f"| {LABELS[r['feature']]} | {r['group']} | {r['n_books']} | {r['n_books_burst']} | "
                 f"{r['n_books_anti_burst']} | {r['expected_each_by_chance']} |")
    L += ['', '## Title/heading check', '',
          'The first window includes each book\'s title and chapter heading. Same raw test, but the '
          '~600-word "opening" starts later:', '',
          '| feature | ' + ' | '.join(f'starts at word {s * 300:,}' for s in SKIP_STARTS) + ' |',
          '|---|' + '---:|' * len(SKIP_STARTS)]
    for f in CHECKED:
        cells = []
        for s in SKIP_STARTS:
            r = sk[(sk['feature'] == f) & (sk['start_window'] == s)].iloc[0]
            cells.append(f"{r['mean_longer_pct']:.2f} (p={r['p_longer_than_typical']:.3f})")
        L.append(f'| {LABELS[f]} | ' + ' | '.join(cells) + ' |')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
