"""
Shared helpers for the verification scripts (plan steps A1-A4):

    onset_robustness.py        A1 split-half / split-book, A2 method variants
    onset_shuffled_control.py  A3 shuffled-author control
    dialogue_share_check.py    A4 dialogue-share check

Nothing here runs on import. The helpers re-implement step 3
(accumulation_onset.py) with every window-count constant passed in through a
Cfg, so the same code can run on 300-word windows (the main analysis) and on
500-word windows (a variant), and with different accumulator sets, register
corrections and onset shapes. With the default settings it reproduces step 3's
corpus-level onset (the scripts print a check against output_onset/
grand_onset.csv so any mismatch is visible straight away).

Terms used throughout
---------------------
curve         one author x feature distinctiveness curve (accumulator_retest.
              build_curves): test books averaged position by position.
accumulator   an author x feature pair whose curve has a significant rising
              slope (chunk-shuffle permutation test, BH-FDR across all tests).
pooled curve  one author's accumulating features, each standardised, averaged.
register      the shared opening shift (longer words, more "of", ...) that
              step 1 found. "Corrected" curves have it subtracted first.
onset         where a flat-then-rising line fitted to the corpus-level pooled
              curve starts to rise.
"""
import contextlib
import os
import pickle
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import stats_tests
from features import FEATURE_NAMES, extract_features
from scoring import corpus_stats, author_profile, distinctiveness_scores
from stats_tests import accumulator_test, bh_fdr
from windowing import parse_sentences, local_windows
from front_matter import load_trimmed
from run_full_corpus import discover_authors, average_profiles, CORPUS_DIR
from accumulator_retest import load_features as _load_features_300
from accumulator_retest import register_directions, ALPHA
from accumulator_retest import OUT as RETEST_OUT

warnings.filterwarnings('ignore', category=RuntimeWarning)

OUT = 'output_verify'
COMPARISON_CSV = os.path.join(RETEST_OUT, 'accum_comparison.csv')
GRAND_CSV = os.path.join('output_onset', 'grand_onset.csv')
BOUNDARY_CSV = os.path.join('output_register', 'register_boundaries.csv')
DIALOGUE_FEATURES = ('exclaim_rate', 'question_rate', 'dash_rate')

INK, MUTED, GRID = '#333333', '#8a8a8a', '#e6e6e6'
C_RAW, C_COR, C_BAND = '#b5b5b5', '#2a78d6', '#eb6834'


# --------------------------------------------------------------- config ----
@dataclass(frozen=True)
class Cfg:
    """Window-count constants, derived from the window size so that they cover
    the same number of WORDS whatever the window size is (step 3 values for
    300-word windows in brackets)."""
    window: int = 300

    def _w(self, words):
        return max(1, int(round(words / self.window)))

    @property
    def conv_pos(self):        # register profile length: 12,000 words      (40)
        return self._w(12000)

    @property
    def base_start(self):      # register baseline starts at 6,000 words    (20)
        return self._w(6000)

    @property
    def grand_len(self):       # corpus-level curve: first 30,000 words     (100)
        return self._w(30000)

    @property
    def grand_max_t(self):     # onset searched up to 18,000 words          (60)
        return self._w(18000)

    @property
    def chunk(self):           # chunk-shuffle null piece: ~1,500 words      (5)
        return self._w(1500)

    @property
    def cut3000(self):         # "drop the first 3,000 words" in windows     (10)
        return self._w(3000)


# ------------------------------------------------------------ plumbing ----
@contextlib.contextmanager
def chunk_size(n):
    """Temporarily set stats_tests.CHUNK_SIZE (and drop its permutation cache,
    whose key does not include the chunk size)."""
    old = stats_tests.CHUNK_SIZE
    stats_tests.CHUNK_SIZE = int(n)
    stats_tests._perm_cache.clear()
    try:
        yield
    finally:
        stats_tests.CHUNK_SIZE = old
        stats_tests._perm_cache.clear()


def _extract_w(args):
    fn, window = args
    text, _, _, _ = load_trimmed(os.path.join(CORPUS_DIR, fn), fn)
    wins = local_windows(parse_sentences(text), window)
    return fn, [extract_features(w) for w in wins]


def load_features(by_author, window=300):
    """Per-window feature dicts for every book, front matter trimmed.
    300-word windows reuse step 2's cache; other sizes get their own cache in
    output_verify/ (built once)."""
    os.makedirs(OUT, exist_ok=True)
    if window == 300:
        os.makedirs(RETEST_OUT, exist_ok=True)
        return _load_features_300(by_author)
    path = os.path.join(OUT, f'trimmed_window_features_w{window}.pkl')
    if os.path.exists(path):
        print(f'Loading cached {window}-word window features from {path}', flush=True)
        with open(path, 'rb') as f:
            return pickle.load(f)
    files = [fn for books in by_author.values() for _, fn in books]
    print(f'Extracting {window}-word window features for {len(files)} books '
          f'(one-time; cached afterwards)...', flush=True)
    t0 = time.time()
    feats = {}
    with ProcessPoolExecutor() as ex:
        for i, (fn, fl) in enumerate(ex.map(_extract_w, [(fn, window) for fn in files]), 1):
            feats[fn] = fl
            print(f'  [{i}/{len(files)}] {fn} ({len(fl)} windows) {time.time()-t0:.0f}s', flush=True)
    with open(path, 'wb') as f:
        pickle.dump(feats, f)
    return feats


def acc_from_comp(comp, expr):
    """author -> [features] for rows of step 2's accum_comparison.csv where the
    boolean expression holds, e.g. comp['sig_trimmed_full']."""
    m = expr(comp) if callable(expr) else expr
    return comp[m.astype(bool)].groupby('author')['feature'].apply(list).to_dict()


def acc_dict(df):
    """author -> [features] from a test_curves() table."""
    return df[df['sig']].groupby('author')['feature'].apply(list).to_dict()


def n_pairs(acc):
    return sum(len(v) for v in acc.values())


# ----------------------------------------------------------- accumulator ----
def test_curves(curves, cfg, n_perm, sel=None, label=None, verbose=True):
    """Accumulator test (stats_tests.accumulator_test, chunk-shuffle null) on
    every author x feature curve, then BH-FDR across ALL these tests together
    (the same family size as step 2's conditions: authors x 24 features).

    sel : optional function n_windows -> (lo, hi): test only curve[lo:hi]
          (e.g. the second half of the book for the split-half check).
    label : if given, the table is cached in output_verify/ and reused.
    Returns a DataFrame: author, feature, n_windows, slope, p, q, sig."""
    path = os.path.join(OUT, f'tests_{label}_p{n_perm}.csv') if label else None
    if path and os.path.exists(path):
        if verbose:
            print(f'  (using saved tests {path})', flush=True)
        return pd.read_csv(path)
    t0 = time.time()
    rows = []
    with chunk_size(cfg.chunk):
        for i, (a, fc) in enumerate(curves.items()):
            for f in FEATURE_NAMES:
                c = np.asarray(fc[f], float)
                if sel is not None:
                    lo, hi = sel(len(c))
                    c = c[lo:hi]
                at = accumulator_test(c, n_perm=n_perm)
                rows.append({'author': a, 'feature': f, 'n_windows': len(c),
                             'slope': at['slope'], 'p': at['p_value']})
            if verbose:
                print(f'    tested [{i+1}/{len(curves)}] {a} ({time.time()-t0:.0f}s)', flush=True)
    df = pd.DataFrame(rows)
    df['q'] = bh_fdr(df['p'].values)
    df['sig'] = df['q'] < ALPHA
    if verbose:
        print(f'  -> {int(df["sig"].sum())} accumulators of {len(df)} tests', flush=True)
    if path:
        os.makedirs(OUT, exist_ok=True)
        df.to_csv(path, index=False)
    return df


# ------------------------------------------------------- per-book curves ----
def build_curves_by_book(by_author, feats):
    """Like accumulator_retest.build_curves, but keeps each test book's curve
    separate instead of averaging them: curves[author] = [ {feature: array}, ... ]
    (one dict per test book, in year order). Same reference profiles, same
    background, same scoring -- only the averaging step is skipped."""
    print('Building per-book distinctiveness curves...', flush=True)
    ref = {a: feats[books[0][1]] for a, books in by_author.items()}
    stats = corpus_stats([d for fl in ref.values() for d in fl])
    profiles = {a: author_profile(fl, stats) for a, fl in ref.items()}
    out = {}
    for a, books in by_author.items():
        if len(books) < 2:
            continue
        own = profiles[a]
        bg = average_profiles([p for b, p in profiles.items() if b != a])
        per = []
        for _, fn in books[1:]:
            sc = [distinctiveness_scores(d, own, bg, stats) for d in feats[fn]]
            per.append({f: np.array([s.get(f, np.nan) for s in sc]) for f in FEATURE_NAMES})
        out[a] = per
    return out


# ------------------------------------------------------- register profile ----
def opening_residuals(by_author, feats, cfg, which='all'):
    """author -> (conv_pos x n_features): the author's average opening shift
    (corpus-z units, relative to a straight-line baseline fitted from
    cfg.base_start on) -- step 3's convention_residuals, parametrised.
      which='all'  every book of the author (step 3)
      which='ref'  the reference (earliest) book only: text that is NOT part
                   of the curves, so it cannot be circular."""
    ref = [d for books in by_author.values() for d in feats[books[0][1]]]
    stats = corpus_stats(ref)
    mu = np.array([stats[f][0] for f in FEATURE_NAMES])
    sd = np.array([stats[f][1] for f in FEATURE_NAMES])
    out = {}
    for a, books in by_author.items():
        if len(books) < 2:
            continue
        use_books = books if which == 'all' else books[:1]
        per = []
        for _, fn in use_books:
            Z = (np.array([[d[f] for f in FEATURE_NAMES] for d in feats[fn]]) - mu) / sd
            x = np.arange(len(Z))
            use = x >= cfg.base_start
            res = np.empty((cfg.conv_pos, len(FEATURE_NAMES)))
            for j in range(Z.shape[1]):
                coef = np.polyfit(x[use], Z[use, j], 1)
                res[:, j] = Z[:cfg.conv_pos, j] - np.polyval(coef, x[:cfg.conv_pos])
            per.append(res)
        out[a] = np.mean(per, axis=0)
    return out


def correction_for(author, mode, resid_all, resid_ref):
    """The opening-register profile to subtract for this author, or None.
      corpus_loo  mean of every OTHER author's opening shift (step 3)
      own_ref     this author's own reference-book opening shift
      none        no correction"""
    if mode == 'corpus_loo':
        others = [v for b, v in resid_all.items() if b != author]
        return np.mean(others, axis=0) if others else None
    if mode == 'own_ref':
        r = resid_ref.get(author)
        return None if r is None else np.nan_to_num(r)
    if mode == 'none':
        return None
    raise ValueError(f'unknown correction mode {mode!r}')


# ------------------------------------------------------------ pooling ----
def pooled_author(curves_a, feats_a, habit_a, corr, cfg, lo, hi):
    """One author's pooled curve over windows lo:hi (raw and register-
    corrected), exactly step 3's pooled_curves except that the slice is
    explicit. The correction is applied on the full curve first (it lives at
    absolute positions 0..conv_pos), then the slice is taken and standardised
    (mean 0, SD 1 within the slice)."""
    raw, cor, used = [], [], []
    for f in feats_a:
        c = np.asarray(curves_a[f], float)
        if np.isnan(c).any():
            continue
        cc = c.copy()
        if corr is not None:
            h = habit_a[f]
            if not np.isnan(h):
                k = min(cfg.conv_pos, len(c))
                cc[:k] -= h * corr[:k, FEATURE_NAMES.index(f)]
        r, q = c[lo:hi], cc[lo:hi]
        m, s = r.mean(), r.std()
        if len(r) < 8 or s < 1e-12:
            continue
        raw.append((r - m) / s)
        cor.append((q - m) / s)
        used.append(f)
    if not used:
        return None, None, []
    return np.mean(raw, axis=0), np.mean(cor, axis=0), used


# ------------------------------------------------------------- shapes ----
SHAPES = ('hinge', 'quad', 'step')


def shape_design(n, t_grid, shape='hinge'):
    """Centred regressor for each candidate onset T (rows) over n positions.
      hinge  flat, then a straight-line rise from T       max(0, p - T)
      quad   flat, then a smooth (curved) rise from T     max(0, p - T)^2
      step   flat, then a jump to a new level at T        1[p >= T]
    A step's T is where the level changes; for a gradual rise that sits
    later than the hinge's T, so compare shapes by their own CIs, not by
    matching numbers."""
    p = np.arange(n, dtype=float)
    d = p[None, :] - t_grid[:, None]
    if shape == 'hinge':
        S = np.maximum(0.0, d)
    elif shape == 'quad':
        S = np.maximum(0.0, d) ** 2
    elif shape == 'step':
        S = (d >= 0).astype(float)
    else:
        raise ValueError(f'unknown shape {shape!r}')
    Sc = S - S.mean(axis=1, keepdims=True)
    return Sc, (Sc ** 2).sum(axis=1)


def fit_shape(Y, t_grid, design):
    """Y: (B x n). Best onset per row among rising fits (same rule as step 3's
    fit_hinge). Returns (T in windows, slope); T = nan if no rising fit."""
    Sc, Sxx = design
    Yc = Y - np.nanmean(Y, axis=1, keepdims=True)
    Sxy = Yc @ Sc.T
    ok_s = Sxx > 1e-12
    gain = np.where((Sxy > 0) & ok_s[None, :], Sxy ** 2 / np.where(ok_s, Sxx, 1.0)[None, :], -np.inf)
    k = np.argmax(gain, axis=1)
    ok = np.isfinite(gain[np.arange(len(Y)), k])
    T = np.where(ok, t_grid[k], np.nan)
    slope = np.where(ok, Sxy[np.arange(len(Y)), k] / np.where(ok_s, Sxx, 1.0)[k], np.nan)
    return T, slope


def fit_grand(G, cfg, shape, rng, n_boot, max_t=None):
    """Corpus-level onset from G (authors x positions, NaN-padded): fit the
    shape to the author-average curve; 95% interval by resampling authors.
    Returns T, lo, hi in WINDOWS, plus slope, rise (mean of last windows minus
    mean of first windows, in SD units) and the number of authors."""
    out = dict(T=np.nan, lo=np.nan, hi=np.nan, slope=np.nan, rise=np.nan, n_authors=0)
    G = np.asarray(G, float)
    if G.ndim != 2 or len(G) == 0:
        return out
    G = G[~np.all(np.isnan(G), axis=1)]
    out['n_authors'] = len(G)
    if len(G) == 0:
        return out
    mean = np.nanmean(G, axis=0)
    good = np.where(~np.isnan(mean))[0]
    if len(good) < 8:
        return out
    n = int(good.max()) + 1
    G, mean = G[:, :n], mean[:n]
    mt = cfg.grand_max_t if max_t is None else max_t
    t_lo = 1 if shape == 'step' else 0
    t_grid = np.arange(t_lo, min(mt, (6 * n) // 10) + 1, dtype=float)
    design = shape_design(n, t_grid, shape)
    T0, s0 = fit_shape(mean[None, :], t_grid, design)
    k = max(1, min(20, n // 3))
    out.update(T=float(T0[0]), slope=float(s0[0]),
               rise=float(np.nanmean(mean[-k:]) - np.nanmean(mean[:k])))
    if n_boot > 0:
        idx = rng.integers(0, len(G), (n_boot, len(G)))
        Tb, _ = fit_shape(np.nanmean(G[idx], axis=1), t_grid, design)
        if np.isfinite(Tb).any():
            out['lo'], out['hi'] = (float(v) for v in np.nanpercentile(Tb, [2.5, 97.5]))
    return out


def pad_to(y, L):
    y = np.asarray(y, float)[:L]
    return np.concatenate([y, np.full(L - len(y), np.nan)])


def estimate(curves, habit, acc, cfg, *, mode='corpus_loo', shape='hinge', resid_all=None,
             resid_ref=None, est=None, L=None, cut=0, rng=None, n_boot=2000,
             drop_feats=(), keep_feats=None):
    """Corpus-level onset for one analysis choice.

    curves, habit : accumulator_retest.build_curves output (or per-book dicts)
    acc           : author -> accumulating features to pool
    mode          : register correction ('corpus_loo' | 'own_ref' | 'none')
    shape         : 'hinge' | 'quad' | 'step'
    est           : function n_windows -> (lo, hi): window range used to
                    estimate the onset (default: the whole curve)
    L             : number of windows in the corpus-level curve
    cut           : drop this many windows from the start of the estimation
                    range (the onset is then measured from the cut)
    drop_feats / keep_feats : exclude / restrict the features pooled
    Returns {'raw': fit, 'cor': fit, 'G_raw', 'G_cor', 'authors', 'n_features'}.
    Fits are in WINDOWS from the start of the estimation range; use
    onset_words() to convert."""
    rng = rng if rng is not None else np.random.default_rng(0)
    L = L or cfg.grand_len
    rows_r, rows_c, authors, nfeat = [], [], [], 0
    for a in sorted(acc):
        if a not in curves:
            continue
        fl = [f for f in acc[a] if f not in drop_feats and (keep_feats is None or f in keep_feats)]
        if not fl:
            continue
        n = len(curves[a][FEATURE_NAMES[0]])
        lo, hi = (0, n) if est is None else est(n)
        lo += cut
        corr = correction_for(a, mode, resid_all, resid_ref)
        raw, cor, used = pooled_author(curves[a], fl, habit[a], corr, cfg, lo, hi)
        if raw is None:
            continue
        rows_r.append(pad_to(raw, L))
        rows_c.append(pad_to(cor, L))
        authors.append(a)
        nfeat += len(used)
    G_raw = np.array(rows_r) if rows_r else np.empty((0, L))
    G_cor = np.array(rows_c) if rows_c else np.empty((0, L))
    fit_raw = fit_grand(G_raw, cfg, shape, rng, n_boot)
    fit_cor = fit_raw if mode == 'none' else fit_grand(G_cor, cfg, shape, rng, n_boot)
    return {'raw': fit_raw, 'cor': fit_cor, 'G_raw': G_raw, 'G_cor': G_cor,
            'authors': authors, 'n_features': nfeat, 'cut': cut}


def onset_words(fit, cfg, cut=0):
    """(onset, lo, hi) in words from the start of the narrative."""
    w = cfg.window
    return ((fit['T'] + cut) * w, (fit['lo'] + cut) * w, (fit['hi'] + cut) * w)


# ----------------------------------------------------- fade (register end) ----
def fade_grid(n_pos):
    return np.arange(1.0, n_pos - 1 + 1e-9, 0.25)


def fade_design(n, e_grid):
    p = np.arange(n, dtype=float)
    G = np.clip(1 - (p[None, :] + 0.5) / e_grid[:, None], 0, None)
    Gc = G - G.mean(axis=1, keepdims=True)
    return Gc, (Gc ** 2).sum(axis=1)


def fit_fade(Y, e_grid, design):
    """End point (in windows) of a fade-out that starts at the beginning and
    reaches zero at E; same model as step 1's end point and onset_gap_test's.
    Y positive at the start = the thing being faded."""
    Gc, Sxx = design
    Yc = Y - np.nanmean(Y, axis=1, keepdims=True)
    Sxy = Yc @ Gc.T
    gain = np.where(Sxy > 0, Sxy ** 2 / Sxx, -np.inf)
    k = np.argmax(gain, axis=1)
    ok = np.isfinite(gain[np.arange(len(Y)), k])
    return np.where(ok, e_grid[k], np.nan)


# ---------------------------------------------------------- reporting ----
def register_band():
    """(25th, 75th percentile) of step 1's per-feature register end points."""
    if os.path.exists(BOUNDARY_CSV):
        b = pd.read_csv(BOUNDARY_CSV)['end_words']
        return float(b.quantile(0.25)), float(b.quantile(0.75))
    return None


def read_real_onsets():
    """Step 3's corpus-level onsets, or {} if not available."""
    if not os.path.exists(GRAND_CSV):
        return {}
    g = pd.read_csv(GRAND_CSV).set_index('version')
    return {v: (float(g.loc[v, 'onset_words']), float(g.loc[v, 'ci_lo']), float(g.loc[v, 'ci_hi']))
            for v in g.index}


def check_baseline(res, cfg):
    """Compare a baseline estimate() with step 3's saved corpus-level onsets."""
    real = read_real_onsets()
    if not real:
        print('  (baseline check skipped: output_onset/grand_onset.csv not found)')
        return
    ok = True
    for ver, key in (('raw', 'raw'), ('corrected', 'cor')):
        mine = onset_words(res[key], cfg)[0]
        theirs = real[ver][0]
        same = abs(mine - theirs) < 1e-6
        ok &= same
        print(f'  baseline check, {ver}: this run {mine:,.0f} words vs step 3 {theirs:,.0f} -> '
              f'{"match" if same else "DIFFERENT"}')
    if not ok:
        print('  !! The baseline does not reproduce step 3. Something upstream changed (corpus, '
              'cached features, accumulator set). Treat comparisons with caution until this is understood.')


def style_axes(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.set_axisbelow(True)


def forest_plot(rows, band, path, title, accept=(3000, 8000)):
    """rows: dicts with label, onset, lo, hi (words). One dot + bar per row."""
    r = [x for x in rows if np.isfinite(x['onset'])]
    if not r:
        return
    fig, ax = plt.subplots(figsize=(8.5, 0.34 * len(r) + 1.8))
    y = np.arange(len(r))[::-1]
    if band is not None:
        ax.axvspan(band[0] / 1000, band[1] / 1000, color=C_BAND, alpha=0.12, linewidth=0)
    if accept is not None:
        ax.axvline(accept[0] / 1000, color=MUTED, lw=0.8, ls=':')
        ax.axvline(accept[1] / 1000, color=MUTED, lw=0.8, ls=':')
    for yi, x in zip(y, r):
        if np.isfinite(x['lo']) and np.isfinite(x['hi']):
            ax.hlines(yi, x['lo'] / 1000, x['hi'] / 1000, color=C_COR, lw=2)
        ax.scatter(x['onset'] / 1000, yi, color=C_COR, s=30, zorder=3, edgecolor='white', linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([x['label'] for x in r], fontsize=7, color=INK)
    ax.set_xlabel('thousand words into the narrative where voice accumulation begins\n'
                  'dot = estimate, bar = 95% interval (authors resampled), orange band = where the opening '
                  'register ends, dotted lines = 3,000 and 8,000 words', fontsize=7.5, color=INK)
    ax.set_title(title, loc='left', fontsize=10, color=INK)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def md_table(df, cols, fmts=None):
    """Markdown table from a DataFrame. cols: list of (column, header)."""
    fmts = fmts or {}
    L = ['| ' + ' | '.join(h for _, h in cols) + ' |', '|' + '|'.join('---' for _ in cols) + '|']
    for _, r in df.iterrows():
        cells = []
        for c, _ in cols:
            v = r[c]
            fm = fmts.get(c)
            if fm and isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v):
                cells.append(format(v, fm))
            elif isinstance(v, float) and pd.isna(v):
                cells.append('-')
            else:
                cells.append(str(v))
        L.append('| ' + ' | '.join(cells) + ' |')
    return L
