"""
Burst-window sweep: did the Sep 18 run miss bursts because of WHERE and
HOW WIDE it looked?

The original burst test (stats_tests.burst_test) asks one question: "are
the first 2 windows (~600 words) more author-distinctive than the rest of
the book?" This script widens that, on the exact same curves
run_full_corpus.py tests (cached by build_curves.py).

PART A -- WIDER OPENINGS  (opening = first k windows, k = 1,2,3,5,8,13,
          i.e. ~300 to ~3,900 words)
  A1  Original method: single-window shuffle null, BH-FDR within each k
      across 33 authors x 24 features. k = 2 reproduces the Sep 18 run.
  A2  Position rank: of every k-window stretch in the book (every start
      position), where does the opening rank? p = share of stretches at
      least as distinctive as the opening. This needs no independence
      assumption: it asks directly "is the opening more distinctive than
      the same-length stretch almost anywhere else in this book?"
  A3  Corpus level, per feature: across the 33 authors, do openings rank
      higher than chance? Under no opening effect each author's rank is
      roughly uniform, so the mean rank-p is ~0.5. Authors are separate
      books, so they ARE independent -- which is what gives this test real
      power where single-author tests cannot have it (see A2 note below).

PART B -- SLIDING SCAN  (the most distinctive block ANYWHERE in the book)
  Block lengths L = 2, 5, 10 windows (~600 / 1,500 / 3,000 words). Take the
  max over every start position; the null takes the max the same way over
  permuted curves, so scanning many positions does not by itself create
  "significant" peaks. Curves are linearly detrended first so a steady
  accumulator cannot pose as a burst at the end of the book.

WHY TWO KINDS OF NULL
  Neighbouring 300-word windows are not independent: a scene or a dialogue-
  heavy chapter spans several windows (median lag-1 autocorrelation of the
  detrended curves is ~0.13, gone by lag ~5). Shuffling single windows --
  the original method -- ignores that, so short stretches look more
  unusual than they really are. The calibration run below measures how
  much: on curves with NO real position effect, the single-window null
  flags roughly 4x as many as it should. So:
    - the scan uses a 5-window CHUNK shuffle as its primary null, and
    - the opening uses the position-rank test (A2/A3) as its robust check.
  Honest-null consequence (A2): a single book of n windows only has about
  n distinct stretches to compare the opening against, so its smallest
  possible p is ~1/n (e.g. 1/300). Study-wide FDR over ~780 tests needs
  p ~ 1e-4, so NO single author x feature opening can clear study-wide
  correction under a null that respects autocorrelation. That is a property
  of the question, not a bug -- and it is why A3 (pooling across authors)
  is the test that can actually answer "do openings burst?"

TIES
  For discrete-valued features (question_rate, exclaim_rate, ...) permuted
  statistics often exactly equal the observed one, and `perm >= observed`
  then depends on floating-point rounding (it moved p by up to 0.6 between
  your Windows run and a Linux run of the same curve). Ties count as "at
  least as extreme" here (1e-9 tolerance, the conservative convention). The
  Sep 18 burst verdicts are identical either way.

Outputs (output_sweep/):
  opening_sweep_results.csv    (author, feature, k): A1 + A2 p/q values
  opening_sweep_summary.csv    counts per k
  opening_corpus_level.csv     A3, per (feature, k)
  scan_results.csv             (author, feature, L): peak location + p/q
  scan_summary.csv             counts per L + where significant peaks fall
  scan_calibration.csv         false-positive and detection rates
  plots/opening_sweep.png, plots/opening_corpus_level.png,
  plots/scan_peak_positions.png

Run:  python burst_sweep.py            (cached curves; ~5 min)
      python burst_sweep.py --rebuild  (re-parse corpus first, +1 min)
      python burst_sweep.py --quick    (2,000 permutations; p floor 0.0005,
                                        so FDR counts are NOT trustworthy)
"""
import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from build_curves import load_curves, OUT_DIR
from stats_tests import bh_fdr

OPENING_KS = (1, 2, 3, 5, 8, 13)
SCAN_LS = (2, 5, 10)
CHUNK = 5            # chunk-shuffle null: chunk length, in windows
WINDOW_WORDS = 300   # run_full_corpus.WINDOW_SIZE
TIE_TOL = 1e-9
ALPHA = 0.05
_ROWS = 1000         # permutations scored per batch (bounds memory)

BLUE = '#2a78d6'
ORANGE = '#eb6834'
PALE_BLUE = '#cde2fb'
INK = '#333333'
MUTED = '#8a8a8a'


# --------------------------------------------------------------------------
# Permutation indices (cached per curve length)
# --------------------------------------------------------------------------
_cache = {}


def perm_index(n, n_perm, kind, seed=0):
    """(n_perm, n) int32 index matrix.
    'perm'  -- each row shuffles single windows; identical RNG stream to
               stats_tests._perm_index, so k=2 reproduces Sep 18 exactly.
    'block' -- each row shuffles consecutive CHUNK-window pieces (the last
               may be shorter) and concatenates them, keeping short-range
               autocorrelation intact."""
    key = (n, n_perm, kind, seed)
    if key in _cache:
        return _cache[key]
    if len(_cache) > 4:
        _cache.clear()
    idx = np.empty((n_perm, n), dtype=np.int32)
    if kind == 'perm':
        rng = np.random.default_rng(seed)
        for s in range(0, n_perm, 2000):
            e = min(s + 2000, n_perm)
            idx[s:e] = np.argsort(rng.random((e - s, n)), axis=1)
    else:
        rng = np.random.default_rng(seed + 7919)
        pieces = [np.arange(i, min(i + CHUNK, n)) for i in range(0, n, CHUNK)]
        for r in range(n_perm):
            idx[r] = np.concatenate([pieces[j] for j in rng.permutation(len(pieces))])
    _cache[key] = idx
    return idx


def detrend(y):
    x = np.arange(len(y), dtype=float)
    return y - np.polyval(np.polyfit(x, y, 1), x)


def norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


# --------------------------------------------------------------------------
# Part A
# --------------------------------------------------------------------------
def opening_perm_pvalues(curve, ks, n_perm):
    """A1: original single-window-shuffle burst test, every k at once."""
    n = len(curve)
    out = {k: (np.nan, np.nan) for k in ks}
    valid = [k for k in ks if n >= k + 2]
    if np.isnan(curve).any() or not valid:
        return out
    idx = perm_index(n, n_perm, 'perm')
    total = curve.sum()
    obs = {k: curve[:k].mean() - (total - curve[:k].sum()) / (n - k) for k in valid}
    exceed = dict.fromkeys(valid, 0)
    kmax = max(valid)
    for s in range(0, n_perm, _ROWS):
        csum = np.cumsum(curve[idx[s:s + _ROWS, :kmax]], axis=1)
        for k in valid:
            hk = csum[:, k - 1]
            exceed[k] += int(np.sum(hk / k - (total - hk) / (n - k) >= obs[k] - TIE_TOL))
    for k in valid:
        out[k] = (float(obs[k]), (1 + exceed[k]) / (n_perm + 1))
    return out


def opening_rank_pvalues(curve, ks):
    """A2: share of k-window stretches (any start) at least as distinctive as
    the opening, on the detrended curve. Also returns the opening's
    percentile (1.0 = most distinctive stretch in the book)."""
    n = len(curve)
    out = {k: (np.nan, np.nan) for k in ks}
    if np.isnan(curve).any():
        return out
    r = detrend(curve)
    c = np.concatenate([[0.0], np.cumsum(r)])
    for k in ks:
        if n < 4 * k:
            continue
        means = (c[k:] - c[:-k]) / k          # every start position 0..n-k
        p = float(np.mean(means >= means[0] - TIE_TOL))
        pct = float(np.mean(means < means[0] - TIE_TOL))
        out[k] = (p, pct)
    return out


def run_opening_sweep(curves, n_perm):
    rows = []
    for key, curve in curves.items():
        author, feat = key.split('|')
        a1 = opening_perm_pvalues(curve, OPENING_KS, n_perm)
        a2 = opening_rank_pvalues(curve, OPENING_KS)
        for k in OPENING_KS:
            rows.append({'author': author, 'feature': feat, 'k_windows': k,
                         'approx_words': k * WINDOW_WORDS, 'n_windows': len(curve),
                         'burst_stat': a1[k][0], 'p_perm': a1[k][1],
                         'p_rank': a2[k][0], 'opening_percentile': a2[k][1]})
    df = pd.DataFrame(rows)
    for col in ('perm', 'rank'):
        df[f'q_{col}'] = np.nan
        for k in OPENING_KS:
            m = df['k_windows'] == k
            df.loc[m, f'q_{col}'] = bh_fdr(df.loc[m, f'p_{col}'].values)
        df[f'sig_{col}'] = df[f'q_{col}'] < ALPHA
    return df


def opening_corpus_level(op):
    """A3: per (feature, k), pool the 33 authors' position-rank p values.
    Null: each ~Uniform(0,1) => mean ~ N(0.5, 1/(12m)). One-sided toward
    'openings more distinctive'. Also reports the opposite direction, since
    an opening that is distinctively LESS author-like is also informative."""
    rows = []
    for (feat, k), g in op.groupby(['feature', 'k_windows']):
        p = g['p_rank'].dropna().values
        m = len(p)
        if m == 0:
            continue
        mean_p = float(p.mean())
        z = (0.5 - mean_p) / math.sqrt(1 / (12 * m))
        rows.append({'feature': feat, 'k_windows': k, 'approx_words': k * WINDOW_WORDS,
                     'n_authors': m, 'mean_rank_p': mean_p,
                     'mean_opening_percentile': float(g['opening_percentile'].mean()),
                     'n_authors_top10pct': int((p <= 0.10).sum()),
                     'expected_top10pct': 0.10 * m,
                     'z_opening_high': z,
                     'p_opening_high': norm_sf(z),
                     'p_opening_low': norm_sf(-z)})
    df = pd.DataFrame(rows)
    df['q_opening_high'] = np.nan
    df['q_opening_low'] = np.nan
    for k in OPENING_KS:
        m = df['k_windows'] == k
        df.loc[m, 'q_opening_high'] = bh_fdr(df.loc[m, 'p_opening_high'].values)
        df.loc[m, 'q_opening_low'] = bh_fdr(df.loc[m, 'p_opening_low'].values)
    return df.sort_values(['k_windows', 'p_opening_high'])


# --------------------------------------------------------------------------
# Part B
# --------------------------------------------------------------------------
def max_block_sums(rows, Ls):
    c = np.cumsum(rows, axis=1)
    c = np.concatenate([np.zeros((c.shape[0], 1)), c], axis=1)
    return {L: (c[:, L:] - c[:, :-L]).max(axis=1) for L in Ls}


def scan_curve(curve, Ls, n_perm, kinds=('block', 'perm'), detrended=False):
    """Max block (sum == contrast, since detrended residuals sum to 0) over
    every start position, vs. the same max over permuted curves."""
    n = len(curve)
    Ls = [L for L in Ls if n >= 4 * L]
    if np.isnan(curve).any() or not Ls:
        return {}
    r = curve if detrended else detrend(curve)
    obs = max_block_sums(r[None, :], Ls)
    c = np.concatenate([[0.0], np.cumsum(r)])
    out = {}
    for L in Ls:
        sums = c[L:] - c[:-L]
        start = int(np.argmax(sums))
        out[L] = {'peak_start_window': start,
                  'peak_position_frac': (start + L / 2) / n,
                  'peak_contrast': float(sums[start] / L - (r.sum() - sums[start]) / (n - L))}
    for kind in kinds:
        idx = perm_index(n, n_perm, kind)
        exceed = dict.fromkeys(Ls, 0)
        for s in range(0, n_perm, _ROWS):
            null = max_block_sums(r[idx[s:s + _ROWS]], Ls)
            for L in Ls:
                exceed[L] += int(np.sum(null[L] >= obs[L][0] - TIE_TOL))
        for L in Ls:
            out[L][f'p_{kind}'] = (1 + exceed[L]) / (n_perm + 1)
    return out


def run_scan(curves, n_perm):
    rows = []
    for key, curve in curves.items():
        author, feat = key.split('|')
        res = scan_curve(curve, SCAN_LS, n_perm)
        for L in SCAN_LS:
            row = {'author': author, 'feature': feat, 'L_windows': L,
                   'approx_words': L * WINDOW_WORDS, 'n_windows': len(curve)}
            if L in res:
                row.update(res[L])
                row['peak_start_word'] = res[L]['peak_start_window'] * WINDOW_WORDS
            rows.append(row)
    df = pd.DataFrame(rows)
    for col in ('block', 'perm'):
        df[f'q_{col}'] = np.nan
        for L in SCAN_LS:
            m = df['L_windows'] == L
            df.loc[m, f'q_{col}'] = bh_fdr(df.loc[m, f'p_{col}'].values)
        df[f'sig_{col}'] = df[f'q_{col}'] < ALPHA
    return df


def calibrate_scan(curves, n_perm, n_curves=120, seed=11):
    """Uses real curves (real noise, real autocorrelation) with any real
    position structure destroyed by a chunk shuffle.
      null          -- should be flagged ~5% of the time at p < .05
      inject_Xsd    -- a 5-window (~1,500-word) burst of X standard
                       deviations added 20-80% of the way through; how
                       often is it found, and found in the right place?"""
    rng = np.random.default_rng(seed)
    keys = [k for k, c in curves.items() if not np.isnan(c).any() and len(c) >= 80]
    keys = [keys[i] for i in rng.choice(len(keys), size=min(n_curves, len(keys)), replace=False)]
    rows = []
    for key in keys:
        y0 = detrend(curves[key])
        n, sd = len(y0), y0.std()
        pieces = [y0[i:i + CHUNK] for i in range(0, n, CHUNK)]
        shuffled = np.concatenate([pieces[j] for j in rng.permutation(len(pieces))])
        start = int(rng.integers(int(0.2 * n), int(0.8 * n) - 5))
        for label, size in [('null', 0.0), ('inject_0.5sd', 0.5), ('inject_1sd', 1.0), ('inject_2sd', 2.0)]:
            y = shuffled.copy()
            y[start:start + 5] += size * sd
            res = scan_curve(detrend(y), (5,), n_perm, detrended=True)[5]
            for kind in ('perm', 'block'):
                flagged = res[f'p_{kind}'] < ALPHA
                right = abs(res['peak_start_window'] - start) <= 5
                rows.append({'curve': key, 'condition': label, 'null': kind,
                             'flagged': flagged,
                             'flagged_right_place': bool(flagged and size > 0 and right)})
    df = pd.DataFrame(rows)
    return (df.groupby(['condition', 'null'])
              .agg(n_curves=('flagged', 'size'), rate_flagged=('flagged', 'mean'),
                   rate_found_right_place=('flagged_right_place', 'mean'))
              .reset_index())


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def _style(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.yaxis.grid(True, color='#e6e6e6', linewidth=0.8)
    ax.set_axisbelow(True)


def _word_axis(ax, words):
    ax.set_xscale('log')
    ax.set_xticks(list(words))
    ax.set_xticklabels([f'{w:,}' for w in words])
    ax.minorticks_off()


def plot_opening(summary, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = summary['approx_words']
    n_top5 = summary['pct_nominal_p05_rank'] * summary['n_tests'] / 100
    chance = 0.05 * summary['n_tests'].iloc[0]
    ax.axhline(chance, color=MUTED, lw=1, ls=':')
    ax.text(x.iloc[0], chance, 'expected by chance (5%)', fontsize=8, color=MUTED, va='bottom')
    ax.plot(x, summary['n_sig_perm'], color=BLUE, lw=2, marker='o', ms=8,
            markeredgecolor='white', markeredgewidth=2)
    ax.plot(x, n_top5, color=ORANGE, lw=2, marker='s', ms=8,
            markeredgecolor='white', markeredgewidth=2)
    ax.annotate('bursts, original test\n(q < 0.05)', (x.iloc[-1], summary['n_sig_perm'].iloc[-1]),
                xytext=(8, 0), textcoords='offset points', fontsize=8, color=INK, va='center')
    ax.annotate('opening in the top 5%\nof its own book', (x.iloc[-1], n_top5.iloc[-1]),
                xytext=(8, 0), textcoords='offset points', fontsize=8, color=INK, va='center')
    ax.axvline(600, color=MUTED, lw=1, ls='--')
    ax.set_ylim(0, max(n_top5.max(), summary['n_sig_perm'].max()) * 1.15)
    ax.text(600, ax.get_ylim()[1], ' Sep 18 setting', fontsize=8, color=MUTED, va='top')
    _word_axis(ax, x)
    ax.set_xlabel('Opening length tested (words)', color=INK)
    ax.set_ylabel('Author × feature curves (of 780)', color=INK)
    ax.set_title('Wider openings: more bursts, but only a small excess over chance',
                 color=INK, loc='left', fontsize=11)
    _style(ax)
    ax.set_xlim(x.min() * 0.85, x.max() * 2.6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_corpus_level(cl, path):
    """Mean opening percentile per feature per k, as a dot strip."""
    feats = (cl[cl['k_windows'] == 2].sort_values('mean_opening_percentile')['feature'].tolist())
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ypos = {f: i for i, f in enumerate(feats)}
    ks = sorted(cl['k_windows'].unique())
    shades = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#104281'][:len(ks)]
    for k, col in zip(ks, shades):
        sub = cl[cl['k_windows'] == k]
        ax.scatter(sub['mean_opening_percentile'], [ypos[f] for f in sub['feature']],
                   s=40, color=col, edgecolor='white', linewidth=1.2, zorder=3,
                   label=f'{k * WINDOW_WORDS:,} words')
        sig = sub[sub['q_opening_high'] < ALPHA]
        ax.scatter(sig['mean_opening_percentile'], [ypos[f] for f in sig['feature']],
                   s=110, facecolor='none', edgecolor=INK, linewidth=1.2, zorder=4)
    ax.axvline(0.5, color=MUTED, lw=1, ls='--')
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=8)
    ax.set_xlim(0.2, 0.8)
    ax.set_xlabel('Mean percentile of the opening among every same-length stretch in the book\n'
                  '(0.5 = the opening is typical; ringed = corpus-level burst, q < 0.05)',
                  color=INK, fontsize=9)
    ax.set_title('Are openings more author-distinctive than the rest of the book?',
                 loc='left', color=INK, fontsize=11)
    _style(ax)
    ax.xaxis.grid(True, color='#e6e6e6', linewidth=0.8)
    ax.yaxis.grid(False)
    ax.legend(title='Opening length', fontsize=7, title_fontsize=8, frameon=False,
              loc='lower right')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scan_positions(scan, path):
    fig, axes = plt.subplots(1, len(SCAN_LS), figsize=(10.5, 3.6), sharey=True)
    bins = np.linspace(0, 1, 11)
    for i, (ax, L) in enumerate(zip(axes, SCAN_LS)):
        sub = scan[(scan['L_windows'] == L) & scan['peak_position_frac'].notna()]
        sig = sub[sub['sig_block']]
        ax.hist(sub['peak_position_frac'], bins=bins, color=PALE_BLUE, edgecolor='white',
                linewidth=2, label='peak of every curve')
        ax.hist(sig['peak_position_frac'], bins=bins, color=BLUE, edgecolor='white',
                linewidth=2, label='significant peaks')
        ax.set_title(f'~{L * WINDOW_WORDS:,}-word blocks: {len(sig)} significant',
                     fontsize=9, color=INK, loc='left')
        ax.set_xlabel('Position in book (0 = start, 1 = end)', fontsize=8, color=INK)
        _style(ax)
        if i == 0:
            ax.set_ylabel('Author × feature curves', color=INK)
            ax.legend(frameon=False, fontsize=7, loc='upper center')
    fig.suptitle('Where is each curve\'s most author-distinctive block? (detrended; chunk-shuffle null)',
                 x=0.01, ha='left', fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rebuild', action='store_true')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--n-perm', type=int, default=None)
    ap.add_argument('--skip-calibration', action='store_true')
    args = ap.parse_args()
    n_perm = args.n_perm or (2000 if args.quick else 20000)

    t0 = time.time()
    curves = load_curves(rebuild=args.rebuild)
    os.makedirs(os.path.join(OUT_DIR, 'plots'), exist_ok=True)
    print(f"{len(curves)} curves loaded. n_perm = {n_perm}.")
    pd.set_option('display.width', 200)

    # ---------------- Part A ----------------
    op = run_opening_sweep(curves, n_perm)
    op.to_csv(os.path.join(OUT_DIR, 'opening_sweep_results.csv'), index=False)
    op_sum = []
    for k, g in op.groupby('k_windows'):
        op_sum.append({'k_windows': k, 'approx_words': k * WINDOW_WORDS,
                       'n_tests': int(g['p_perm'].notna().sum()),
                       'n_sig_perm': int(g['sig_perm'].sum()),
                       'n_authors_sig_perm': g.loc[g['sig_perm'], 'author'].nunique(),
                       'n_sig_rank': int(g['sig_rank'].sum()),
                       'pct_nominal_p05_perm': 100 * float((g['p_perm'] < 0.05).mean()),
                       'pct_nominal_p05_rank': 100 * float((g['p_rank'] < 0.05).mean()),
                       'sig_perm_list': '; '.join(g.loc[g['sig_perm'], 'author'] + '/' + g.loc[g['sig_perm'], 'feature'])})
    op_sum = pd.DataFrame(op_sum)
    op_sum.to_csv(os.path.join(OUT_DIR, 'opening_sweep_summary.csv'), index=False)
    plot_opening(op_sum, os.path.join(OUT_DIR, 'plots', 'opening_sweep.png'))

    cl = opening_corpus_level(op)
    cl.to_csv(os.path.join(OUT_DIR, 'opening_corpus_level.csv'), index=False)
    plot_corpus_level(cl, os.path.join(OUT_DIR, 'plots', 'opening_corpus_level.png'))
    print(f"Part A done ({time.time()-t0:.0f}s)")
    print(op_sum.drop(columns='sig_perm_list').to_string(index=False))
    print(cl[(cl['q_opening_high'] < ALPHA) | (cl['q_opening_low'] < ALPHA)].to_string(index=False))

    # ---------------- Part B ----------------
    sc = run_scan(curves, n_perm)
    sc.to_csv(os.path.join(OUT_DIR, 'scan_results.csv'), index=False)
    rows = []
    for L in SCAN_LS:
        sub = sc[(sc['L_windows'] == L) & sc['p_block'].notna()]
        sig = sub[sub['sig_block']]
        frac = sig['peak_position_frac']
        rows.append({'L_windows': L, 'approx_words': L * WINDOW_WORDS, 'n_tests': len(sub),
                     'n_sig_block': len(sig), 'n_authors_block': sig['author'].nunique(),
                     'n_sig_perm': int(sub['sig_perm'].sum()),
                     'n_sig_first_third': int((frac < 1 / 3).sum()),
                     'n_sig_middle_third': int(((frac >= 1 / 3) & (frac < 2 / 3)).sum()),
                     'n_sig_last_third': int((frac >= 2 / 3).sum()),
                     'sig_block_list': '; '.join(sig['author'] + '/' + sig['feature'] + '@'
                                                 + (100 * frac).round().astype(int).astype(str) + '%')})
    sc_sum = pd.DataFrame(rows)
    sc_sum.to_csv(os.path.join(OUT_DIR, 'scan_summary.csv'), index=False)
    plot_scan_positions(sc, os.path.join(OUT_DIR, 'plots', 'scan_peak_positions.png'))
    print(f"Part B done ({time.time()-t0:.0f}s)")
    print(sc_sum.drop(columns='sig_block_list').to_string(index=False))

    # ---------------- Calibration ----------------
    if not args.skip_calibration:
        cal = calibrate_scan(curves, min(n_perm, 5000))
        cal.to_csv(os.path.join(OUT_DIR, 'scan_calibration.csv'), index=False)
        print(f"Calibration done ({time.time()-t0:.0f}s)")
        print(cal.to_string(index=False))

    print(f"\nTotal {time.time()-t0:.0f}s. Outputs in {OUT_DIR}/")


if __name__ == '__main__':
    main()
