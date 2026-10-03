"""
Step 3 of the convention -> voice experiment: WHERE DOES ACCUMULATION BEGIN?

Steps 1-2 showed: novels open in a shared register that fades within the
first ~1,500-2,500 words, and most accumulators are real voice building
(not the register fading). The open question: does an author's voice start
building only AFTER the opening register fades, or from page one,
underneath it?

Method
------
1. Curves: the same front-matter-trimmed distinctiveness curves as step 2
   (accumulator_retest.py; its cached features are reused).
2. Which features: each author's accumulators from step 2 (whole book,
   trimmed -- the 69). Authors with none are skipped.
3. One curve per author: each accumulating feature's curve is standardised
   (mean 0, SD 1 over the book) and the author's features are averaged.
   One feature's curve is too noisy to place an onset; pooling helps.
4. Two versions of every curve:
     raw        as measured
     corrected  the shared opening register subtracted first. For each
                feature, the corpus-average opening shift (first 40
                windows, in the same z units as the curves, leave-this-
                author-out) is multiplied by the author's habit direction
                and removed. This is what the opening would do to ANY
                author with this habit, so what's left is the author's own
                trajectory.
5. Onset: a "flat, then rising" line (hinge) is fitted to the pooled curve.
   The hinge point T is where accumulation begins. T = 0 means voice
   builds from the first word. Uncertainty: a block bootstrap of the
   residuals (5-window blocks, matching the chunk null used elsewhere),
   2,000 draws, 95% interval.
6. Corpus-level onset: all authors' pooled curves are averaged over the
   first 30,000 words and the same hinge is fitted; the interval comes from
   resampling authors. This is the headline number -- single-author
   onsets are noisy.
7. Compare with step 1's register end points (output_register/
   register_boundaries.csv).

Per-author verdict (corrected curve):
  from the start     95% interval reaches 0 and ends before 3,000 words
  after the opening  95% interval starts at 1,500 words or later
  can't tell         anything else

Run after accumulator_retest.py:  python accumulation_onset.py
Outputs (output_onset/):
  author_onsets.csv, grand_onset.csv, onset_report.md
  plots/onset_grand_curve.png, plots/onset_by_author.png
"""
import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from features import FEATURE_NAMES
from scoring import corpus_stats
from run_full_corpus import discover_authors, WINDOW_SIZE
from accumulator_retest import load_features, build_curves, ALPHA
from accumulator_retest import OUT as RETEST_OUT

OUT = 'output_onset'
COMPARISON_CSV = os.path.join(RETEST_OUT, 'accum_comparison.csv')
BOUNDARY_CSV = os.path.join('output_register', 'register_boundaries.csv')
ACCUM_SET = 'trimmed_full'   # which step-2 condition picks each author's accumulators
CONV_POS = 40                # convention profile length: 40 windows = 12,000 words
BASE_START = 20              # baseline for the convention profile: windows >= 6,000 words
GRAND_LEN = 100              # corpus-level curve: first 100 windows = 30,000 words
GRAND_MAX_T = 60             # corpus-level onset searched up to 18,000 words
AUTHOR_MAX_T_FRAC = 0.5      # per-author onset searched over the first half of the book
BLOCK = 5
N_BOOT = 2000
SEED = 20261003

INK, MUTED, GRID = '#333333', '#8a8a8a', '#e6e6e6'
C_RAW, C_COR, C_BAND = '#b5b5b5', '#2a78d6', '#eb6834'

warnings.filterwarnings('ignore', category=RuntimeWarning)


# ------------------------------------------------- convention profiles ----
def convention_residuals(by_author, feats):
    """author -> (CONV_POS x n_features): the author's average opening shift,
    in the same corpus-z units the distinctiveness curves use, relative to a
    straight-line baseline fitted from window BASE_START on (as in step 1)."""
    ref = [d for books in by_author.values() for d in feats[books[0][1]]]
    stats = corpus_stats(ref)
    mu = np.array([stats[f][0] for f in FEATURE_NAMES])
    sd = np.array([stats[f][1] for f in FEATURE_NAMES])
    out = {}
    for a, books in by_author.items():
        if len(books) < 2:
            continue
        per = []
        for _, fn in books:
            Z = (np.array([[d[f] for f in FEATURE_NAMES] for d in feats[fn]]) - mu) / sd
            x = np.arange(len(Z))
            use = x >= BASE_START
            res = np.empty((CONV_POS, len(FEATURE_NAMES)))
            for j in range(Z.shape[1]):
                coef = np.polyfit(x[use], Z[use, j], 1)
                res[:, j] = Z[:CONV_POS, j] - np.polyval(coef, x[:CONV_POS])
            per.append(res)
        out[a] = np.mean(per, axis=0)
    return out


# --------------------------------------------------------- hinge model ----
def hinge_design(n, t_grid):
    p = np.arange(n, dtype=float)
    H = np.maximum(0.0, p[None, :] - t_grid[:, None])
    Hc = H - H.mean(axis=1, keepdims=True)
    return Hc, (Hc ** 2).sum(axis=1)


def fit_hinge(Y, t_grid, design):
    """Y: (B x n). Best hinge point per row (rising slopes only).
    Returns (T in windows, slope); T = nan if no rising fit exists."""
    Hc, Sxx = design
    Yc = Y - np.nanmean(Y, axis=1, keepdims=True)
    Sxy = Yc @ Hc.T
    gain = np.where(Sxy > 0, Sxy ** 2 / Sxx, -np.inf)
    k = np.argmax(gain, axis=1)
    ok = np.isfinite(gain[np.arange(len(Y)), k])
    T = np.where(ok, t_grid[k], np.nan)
    slope = np.where(ok, Sxy[np.arange(len(Y)), k] / Sxx[k], np.nan)
    return T, slope


def hinge_fitted(y, T):
    n = len(y)
    h = np.maximum(0.0, np.arange(n) - T)
    X = np.column_stack([np.ones(n), h])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return X @ coef, coef


def block_bootstrap_T(y, t_grid, design, rng):
    T0, _ = fit_hinge(y[None, :], t_grid, design)
    T0 = T0[0]
    if np.isnan(T0):
        return np.nan, np.nan, np.nan
    fitted, _ = hinge_fitted(y, T0)
    r = y - fitted
    n = len(y)
    nb = int(np.ceil(n / BLOCK))
    starts = rng.integers(0, n - BLOCK + 1, (N_BOOT, nb))
    idx = (starts[:, :, None] + np.arange(BLOCK)[None, None, :]).reshape(N_BOOT, -1)[:, :n]
    Ts, _ = fit_hinge(fitted[None, :] + r[idx], t_grid, design)
    lo, hi = np.nanpercentile(Ts, [2.5, 97.5])
    return T0, lo, hi


def verdict(lo_w, hi_w):
    if np.isnan(lo_w):
        return 'no rise found'
    if lo_w <= 0 and hi_w < 3000:
        return 'from the start'
    if lo_w >= 1500:
        return 'after the opening'
    return "can't tell"


# ------------------------------------------------------------ pooling ----
def pooled_curves(curves_a, feats_a, habit_a, conv_a):
    raw, cor, used = [], [], []
    for f in feats_a:
        c = curves_a[f]
        if np.isnan(c).any():
            continue
        m, s = c.mean(), c.std()
        if s < 1e-12:
            continue
        cc = c.copy()
        h = habit_a[f]
        if not np.isnan(h):
            k = min(CONV_POS, len(c))
            cc[:k] -= h * conv_a[:k, FEATURE_NAMES.index(f)]
        raw.append((c - m) / s)
        cor.append((cc - m) / s)
        used.append(f)
    if not used:
        return None, None, []
    return np.mean(raw, axis=0), np.mean(cor, axis=0), used


# --------------------------------------------------------------- plots ----
def _style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.set_axisbelow(True)


def _smooth(y, k=5):
    out = np.full(len(y), np.nan)
    for i in range(len(y)):
        lo, hi = max(0, i - k // 2), min(len(y), i + k // 2 + 1)
        out[i] = np.nanmean(y[lo:hi])
    return out


def plot_grand(G_raw, G_cor, grand, band, path):
    x = (np.arange(GRAND_LEN) + 0.5) * WINDOW_SIZE / 1000
    yr, yc = np.nanmean(G_raw, axis=0), np.nanmean(G_cor, axis=0)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    if band is not None:
        ax.axvspan(band[0] / 1000, band[1] / 1000, color=C_BAND, alpha=0.12, linewidth=0)
        ax.text(band[1] / 1000 + 0.2, 0.97, 'opening register ends\n(step 1, middle half of features)',
                transform=ax.get_xaxis_transform(), fontsize=7, color=C_BAND, va='top')
    ax.plot(x, _smooth(yr), color=C_RAW, lw=1.5, label='raw (5-window average)')
    ax.plot(x, _smooth(yc), color=C_COR, lw=2, label='opening register removed (5-window average)')
    g = grand.set_index('version').loc['corrected']
    if not np.isnan(g['onset_words']):
        fit, _ = hinge_fitted(yc, g['onset_words'] / WINDOW_SIZE)
        ax.plot(x, fit, color=INK, lw=1.2, ls='--', label=f"hinge fit: rise begins at {g['onset_words']:,.0f} words")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel('thousand words into the narrative', fontsize=8, color=INK)
    ax.set_ylabel('pooled voice signal (SD units)', fontsize=8, color=INK)
    ax.set_title('When does voice start to build? Average of all authors\' accumulating features',
                 loc='left', fontsize=10, color=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    _style(ax)
    ax.legend(frameon=False, fontsize=7, loc='lower right')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_authors(res, band, path):
    r = res.dropna(subset=['onset_words_corrected']).sort_values('onset_words_corrected')
    fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(r) + 1.4))
    y = np.arange(len(r))
    if band is not None:
        ax.axvspan(band[0] / 1000, band[1] / 1000, color=C_BAND, alpha=0.12, linewidth=0)
    ax.hlines(y, r['ci_lo_corrected'] / 1000, r['ci_hi_corrected'] / 1000, color=C_COR, lw=2)
    ax.scatter(r['onset_words_corrected'] / 1000, y, color=C_COR, s=30, zorder=3, edgecolor='white', linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{a} ({n})" for a, n in zip(r['author'], r['n_features'])], fontsize=7, color=INK)
    ax.set_xlabel('thousand words into the narrative where accumulation begins\n'
                  'dot = estimate, bar = 95% interval, shaded = where the opening register ends; '
                  '(n) = features pooled', fontsize=8, color=INK)
    ax.set_title('Accumulation onset per author (opening register removed)', loc='left', fontsize=10, color=INK)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -------------------------------------------------------------- report ----
def write_report(res, grand, band, path):
    L = ['# Accumulation onset (step 3)', '',
         f'Run {time.strftime("%Y-%m-%d")}. Each author\'s accumulating features (step 2, whole book, '
         'front matter trimmed) pooled into one curve; "flat, then rising" line fitted; onset = where the rise '
         'begins. "Corrected" = the shared opening register subtracted first.', '']
    if band is not None:
        L += [f'Opening register ends (step 1, middle half of features): {band[0]:,.0f}-{band[1]:,.0f} words.', '']
    L += ['## Corpus level (all authors averaged, first 30,000 words)', '',
          '| version | onset | 95% interval (authors resampled) |', '|---|---:|---:|']
    for _, g in grand.iterrows():
        L.append(f"| {g['version']} | {g['onset_words']:,.0f} words | {g['ci_lo']:,.0f}-{g['ci_hi']:,.0f} |")
    L += ['', '## Per author (corrected curve)', '']
    vc = res['verdict'].value_counts()
    L += ['| verdict | authors |', '|---|---:|'] + [f'| {k} | {v} |' for k, v in vc.items()]
    L += ['', '| author | features pooled | onset, raw | onset, corrected | 95% interval | verdict |',
          '|---|---:|---:|---:|---:|---|']
    for _, r in res.sort_values('onset_words_corrected').iterrows():
        L.append(f"| {r['author']} | {r['n_features']} | {r['onset_words_raw']:,.0f} | "
                 f"{r['onset_words_corrected']:,.0f} | {r['ci_lo_corrected']:,.0f}-{r['ci_hi_corrected']:,.0f} | "
                 f"{r['verdict']} |")
    L += ['', 'How to read it: an onset near 0 means the author\'s voice starts building from the first page, '
          'underneath the opening register. An onset at or after the shaded band means voice starts building '
          'once the register has faded. Single-author intervals are wide; the corpus-level row is the main result.',
          '', 'Caution: the features pooled for each author were chosen because their curve rises over the whole '
          'book, so every pooled curve rises somewhere by construction. The question answered here is where, '
          'not whether.']
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')


# ---------------------------------------------------------------- main ----
def main():
    t0 = time.time()
    os.makedirs(os.path.join(OUT, 'plots'), exist_ok=True)
    rng = np.random.default_rng(SEED)
    if not os.path.exists(COMPARISON_CSV):
        raise SystemExit(f'{COMPARISON_CSV} not found -- run accumulator_retest.py first.')

    by_author = discover_authors()
    feats = load_features(by_author)
    curves, habit = build_curves(by_author, feats)
    print('Computing the shared opening register per author (for the corrected curves)...', flush=True)
    conv_res = convention_residuals(by_author, feats)

    comp = pd.read_csv(COMPARISON_CSV)
    acc = comp[comp[f'sig_{ACCUM_SET}'].astype(bool)].groupby('author')['feature'].apply(list).to_dict()
    print(f'{sum(len(v) for v in acc.values())} accumulators across {len(acc)} authors '
          f'(step 2, {ACCUM_SET}).', flush=True)

    print('Fitting onsets per author (block bootstrap)...', flush=True)
    rows, G_raw, G_cor = [], [], []
    for i, (a, fl) in enumerate(sorted(acc.items())):
        conv_a = np.mean([conv_res[b] for b in conv_res if b != a], axis=0)
        raw, cor, used = pooled_curves(curves[a], fl, habit[a], conv_a)
        if raw is None:
            continue
        n = len(raw)
        t_grid = np.arange(0, int(n * AUTHOR_MAX_T_FRAC) + 1, dtype=float)
        design = hinge_design(n, t_grid)
        Tr, lr, hr = block_bootstrap_T(raw, t_grid, design, rng)
        Tc, lc, hc = block_bootstrap_T(cor, t_grid, design, rng)
        w = WINDOW_SIZE
        rows.append({'author': a, 'n_features': len(used), 'features': ' '.join(used), 'n_windows': n,
                     'onset_words_raw': Tr * w, 'ci_lo_raw': lr * w, 'ci_hi_raw': hr * w,
                     'onset_words_corrected': Tc * w, 'ci_lo_corrected': lc * w, 'ci_hi_corrected': hc * w,
                     'verdict': verdict(lc * w, hc * w)})
        pad = lambda y: np.concatenate([y[:GRAND_LEN], np.full(max(0, GRAND_LEN - len(y)), np.nan)])
        G_raw.append(pad(raw))
        G_cor.append(pad(cor))
        print(f'  [{i+1}/{len(acc)}] {a}: {len(used)} features, onset {Tc * w:,.0f} words '
              f'({lc * w:,.0f}-{hc * w:,.0f}) ({time.time()-t0:.0f}s)', flush=True)
    res = pd.DataFrame(rows)
    G_raw, G_cor = np.array(G_raw), np.array(G_cor)

    print('Fitting the corpus-level onset (authors resampled)...', flush=True)
    t_grid = np.arange(0, GRAND_MAX_T + 1, dtype=float)
    design = hinge_design(GRAND_LEN, t_grid)
    grand = []
    for name, G in (('raw', G_raw), ('corrected', G_cor)):
        T0, _ = fit_hinge(np.nanmean(G, axis=0)[None, :], t_grid, design)
        idx = rng.integers(0, len(G), (N_BOOT, len(G)))
        Yb = np.nanmean(G[idx], axis=1)
        Tb, _ = fit_hinge(Yb, t_grid, design)
        lo, hi = np.nanpercentile(Tb, [2.5, 97.5])
        grand.append({'version': name, 'onset_words': T0[0] * WINDOW_SIZE,
                      'ci_lo': lo * WINDOW_SIZE, 'ci_hi': hi * WINDOW_SIZE, 'n_authors': len(G)})
    grand = pd.DataFrame(grand)

    band = None
    if os.path.exists(BOUNDARY_CSV):
        b = pd.read_csv(BOUNDARY_CSV)['end_words']
        band = (float(b.quantile(0.25)), float(b.quantile(0.75)))

    res.to_csv(os.path.join(OUT, 'author_onsets.csv'), index=False)
    grand.to_csv(os.path.join(OUT, 'grand_onset.csv'), index=False)
    plot_grand(G_raw, G_cor, grand, band, os.path.join(OUT, 'plots', 'onset_grand_curve.png'))
    plot_authors(res, band, os.path.join(OUT, 'plots', 'onset_by_author.png'))
    write_report(res, grand, band, os.path.join(OUT, 'onset_report.md'))

    print('\n' + grand.to_string(index=False))
    print('\n' + res['verdict'].value_counts().to_string())
    print(f'\nOutputs written to {OUT}/ (report: onset_report.md) -- total {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
