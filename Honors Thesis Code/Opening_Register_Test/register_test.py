"""
Opening Register Test -- stage 3: the test, the placebo check, and the verdict.

ONE COMMAND runs everything (stage 1 loader, stage 2 window features, stage 3
statistics) on any folder of plain-text novels:

    python register_test.py --corpus-dir path\\to\\txt_folder --out output_run
    python register_test.py --corpus-dir ... --authors authors.csv --starts starts.csv

What it tests (frozen in config.py; spec in the project docs)
-------------------------------------------------------------
Claim: novels open in a shared "opening register". 13 pre-chosen features
should shift in a predicted direction in the first 1,500 words, and dialogue
share should be lower.

Method (the same as the original opening_register_scan.py)
1. Each book is cut into 300-word windows (stage 2).
2. Per book and feature, a straight line is fitted to the windows from word
   6,000 onward (the "rest of the book" baseline). Every window becomes a
   z-score against it: z = (value - baseline) / SD of the baseline residuals.
   The opening never influences its own baseline.
3. Opening score = mean z over the first 5 windows (about 1,500 words),
   averaged across an author's books. AUTHORS are the unit of analysis.
4. Per feature, a one-sided sign-flip permutation test across authors in the
   PREDICTED direction (exact when there are 16 authors or fewer, otherwise
   20,000 random flips). Benjamini-Hochberg across the 13 tested features.
   A feature PASSES at q < 0.05.
5. Dialogue (share of words inside quotation marks) is measured as the plain
   difference in share points (opening minus the same baseline line, NOT a
   z-score: books with almost no dialogue would blow a z up) and gets a
   one-sided test (lower in the opening), p < 0.05. Books without detectable
   dialogue are left out of this test only.
6. VERDICT (frozen): CONFIRMED = at least 9 of 13 pass AND dialogue is lower;
   PARTIAL = 5-8 pass, or 9+ pass but dialogue does not; NOT FOUND = 4 or fewer.
7. PLACEBO: the same test at random mid-book "openings" (50 draws). It shows
   how often the test passes features, or would say CONFIRMED, by chance.
8. The 11 other original features are reported in a separate EXPLORATORY table
   (two-sided tests); they do not count toward the verdict.

Writes, in --out (besides the stage 1-2 files):
    register_report.md        the verdict and all tables  <- read this first
    register_results.csv      the 13 tested features + dialogue
    exploratory_results.csv   the other 11 features
    author_scores.csv         each author's opening score per feature
    placebo.csv               one row per placebo draw
    plots/register_effects.png
"""
import argparse
import os
import time
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config
from loader import load_corpus, write_outputs
from windows import FEATURE_NAMES, build_windows, write_window_outputs

COLS = list(FEATURE_NAMES) + ['dialogue']
COL = {f: i for i, f in enumerate(COLS)}
TESTED = list(config.TESTED)
DIALOGUE_J = COL['dialogue']

# Original Victorian/Edwardian corpus (33 authors), first 1,500 words: mean z and the number of
# authors shifting the same way. Reference only; never used in the verdict.
# Source: voice_pipeline/output_register/opening_register_report.md (run 2026-10-03).
ORIGINAL = {
    'sent_len_mean': (0.60, 29), 'avg_word_len': (0.40, 28), 'fw_the': (0.48, 29), 'fw_of': (0.52, 28),
    'sent_len_std': (0.31, 25), 'syll_per_word': (0.35, 28), 'fw_in': (0.23, 26), 'fw_and': (0.23, 26),
    'fw_was': (0.19, 21), 'pct_monosyllabic': (-0.38, 27), 'fw_to': (-0.23, 25), 'fw_but': (-0.16, 23),
    'ttr': (-0.21, 22),
}
ORIGINAL_N_AUTHORS = 33
ORIGINAL_DIALOGUE_POINTS = 17      # original corpus: about 17 share points lower (30 of 33 authors)

INK, MUTED, GRID, BLUE = '#333333', '#8a8a8a', '#e6e6e6', '#2a78d6'


# ============================================================ z-scores per book ==
def book_matrix(res):
    """24 features + dialogue share (NaN if the book has no detectable dialogue): windows x 25."""
    d = res['dialogue'] if res['style'] != 'none' else np.full(len(res['dialogue']), np.nan)
    return np.column_stack([res['features'], d])


def book_z(M, exclude=None):
    """z of every window against a straight-line baseline fitted from window BASELINE_START onward
    (or, for the placebo, outside the window range `exclude` = (lo, hi)).
    The dialogue column is the plain difference in share (opening minus baseline), not a z."""
    n = len(M)
    x = np.arange(n)
    use = x >= config.BASELINE_START_WINDOW
    if exclude is not None:
        use = (x < exclude[0]) | (x >= exclude[1])
    Z = np.full(M.shape, np.nan)
    for j in range(M.shape[1]):
        y = M[:, j]
        ok = use & ~np.isnan(y)
        if ok.sum() < 5:
            continue
        coef = np.polyfit(x[ok], y[ok], 1)
        r = y - np.polyval(coef, x)
        if j == DIALOGUE_J:
            # Dialogue is NOT divided by the baseline SD: a book with almost no dialogue (a play, an essay)
            # has a near-zero SD and would turn a tiny difference into a huge z. It stays in share
            # (fraction of words in quotes; 0.10 = 10 share points), as in the original pipeline.
            Z[:, j] = r
            continue
        sd = r[ok].std(ddof=2)
        if sd > 1e-12:
            Z[:, j] = r / sd
    return Z


def author_opening_scores(groups):
    """groups: author -> list of window matrices. Returns authors, (authors x 25) opening scores."""
    k = config.OPENING_WINDOWS
    authors = sorted(groups)
    A = np.full((len(authors), len(COLS)), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        for i, a in enumerate(authors):
            stack = np.stack([book_z(M)[:k] for M in groups[a]])        # books x k x 25
            A[i] = np.nanmean(np.nanmean(stack, axis=0), axis=0)
    return authors, A


# ================================================================== statistics ==
_SIGNS = {}


def _all_signs(n):
    if n not in _SIGNS:
        idx = np.arange(2 ** n)[:, None]
        _SIGNS[n] = (((idx >> np.arange(n)) & 1) * 2 - 1).astype(float)
    return _SIGNS[n]


def signflip_p(v, alternative, rng, n_perm=config.N_PERM):
    """Sign-flip permutation p for mean(v) across authors. alternative: 'greater' or 'two-sided'.
    Exact (all 2^n flips) when n <= EXACT_MAX_AUTHORS, else n_perm random flips."""
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n < 2:
        return np.nan
    obs = abs(v.mean()) if alternative == 'two-sided' else v.mean()
    if n <= config.EXACT_MAX_AUTHORS:
        null, exact = _all_signs(n) @ v / n, True
    else:
        null, exact = (rng.choice([-1.0, 1.0], size=(n_perm, n)) * v).mean(axis=1), False
    stat = np.abs(null) if alternative == 'two-sided' else null
    c = int(np.sum(stat >= obs - 1e-12))
    return c / len(null) if exact else (1 + c) / (len(null) + 1)


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1)
    return out


def boot_ci(v, rng, n=config.N_BOOT):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return np.nan, np.nan
    m = v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return lo, hi


def signed(v, direction):
    return v if direction == 'higher' else -v


def pts(x):
    """dialogue difference (fraction of words) -> share points"""
    return 100.0 * x


def verdict_from(n_pass, dialogue_pass, dialogue_testable=True):
    if n_pass >= config.CONFIRMED_MIN_PASS and dialogue_pass:
        return 'CONFIRMED'
    if n_pass >= config.PARTIAL_MIN_PASS:
        return 'PARTIAL'
    return 'NOT FOUND'


def tested_table(authors, A, rng):
    rows = []
    for f in TESTED + ['dialogue']:
        d = config.TESTED.get(f, config.DIALOGUE_PREDICTION)
        v = A[:, COL[f]]
        ok = ~np.isnan(v)
        n = int(ok.sum())
        sv = signed(v, d)
        lo, hi = boot_ci(v, rng)
        o = ORIGINAL.get(f)
        rows.append({
            'feature': f, 'label': config.LABELS[f], 'predicted': d, 'n_authors': n,
            'unit': 'share (fraction of words; x100 = points)' if f == 'dialogue' else 'z',
            'mean_z': float(np.nanmean(v)) if n else np.nan, 'ci_lo': lo, 'ci_hi': hi,
            'authors_in_predicted_direction': int(np.sum(sv[ok] > 0)),
            'p_one_sided': signflip_p(sv, 'greater', rng),
            'original_mean_z': o[0] if o else np.nan,
            'original_authors_same_direction': o[1] if o else np.nan,
        })
    df = pd.DataFrame(rows)
    feat = df['feature'] != 'dialogue'
    df.loc[feat, 'q'] = bh(df.loc[feat, 'p_one_sided'].values)
    df.loc[~feat, 'q'] = df.loc[~feat, 'p_one_sided']          # single test: q = p
    df['pass'] = df['q'] < config.ALPHA
    return df


def exploratory_table(A, rng):
    rows = []
    for f in config.EXPLORATORY:
        v = A[:, COL[f]]
        ok = ~np.isnan(v)
        m = float(np.nanmean(v)) if ok.sum() else np.nan
        rows.append({'feature': f, 'label': config.LABELS[f], 'n_authors': int(ok.sum()), 'mean_z': m,
                     'authors_same_sign_as_mean': int(np.sum(np.sign(v[ok]) == np.sign(m))),
                     'p_two_sided': signflip_p(v, 'two-sided', rng)})
    df = pd.DataFrame(rows)
    df['q'] = bh(df['p_two_sided'].values)
    df['significant'] = df['q'] < config.ALPHA
    return df.sort_values('p_two_sided')


def placebo(groups, rng, n_draws=config.N_PLACEBO):
    """The same test at random mid-book pseudo-openings (baseline fitted outside the pseudo-opening region)."""
    k = config.OPENING_WINDOWS
    authors = sorted(groups)
    cols = [COL[f] for f in TESTED] + [DIALOGUE_J]
    rows = []
    for d in range(n_draws):
        V = np.full((len(authors), len(COLS)), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            for i, a in enumerate(authors):
                vals = []
                for M in groups[a]:
                    n = len(M)
                    s = int(rng.integers(int(0.3 * n), int(0.7 * n) - k))
                    Z = book_z(M, exclude=(s, s + config.BASELINE_START_WINDOW))
                    vals.append(Z[s:s + k].mean(axis=0))
                V[i] = np.nanmean(vals, axis=0)
        p = np.array([signflip_p(signed(V[:, COL[f]], config.TESTED[f]), 'greater', rng, 5000) for f in TESTED])
        pd_ = signflip_p(signed(V[:, DIALOGUE_J], config.DIALOGUE_PREDICTION), 'greater', rng, 5000)
        q = bh(p)
        n_pass = int(np.sum(q < config.ALPHA))
        dpass = bool(pd_ < config.ALPHA)
        rows.append({'draw': d, 'n_features_p_below_05': int(np.sum(p < config.ALPHA)),
                     'n_features_pass_fdr': n_pass, 'dialogue_p': pd_, 'dialogue_pass': dpass,
                     'would_be_confirmed': verdict_from(n_pass, dpass) == 'CONFIRMED',
                     'verdict': verdict_from(n_pass, dpass)})
        if (d + 1) % 10 == 0:
            print(f'  placebo draw {d + 1}/{n_draws}', flush=True)
    return pd.DataFrame(rows)


# ===================================================================== outputs ==
def style_axes(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)


def plot_effects(tt, path, n_authors):
    """Effect in the PREDICTED direction (positive = as predicted): new corpus vs original corpus."""
    rows = list(tt.itertuples())
    feats = [r for r in rows if r.feature != 'dialogue']
    dlg = [r for r in rows if r.feature == 'dialogue'][0]
    order = sorted(feats, key=lambda r: -(r.mean_z if r.predicted == 'higher' else -r.mean_z))
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8.4, 0.42 * len(order) + 2.6),
                                  gridspec_kw={'height_ratios': [len(order), 2.2]})
    ys = []
    for i, r in enumerate(order):
        y = len(order) - 1 - i
        ys.append((y, r.label + ('  ↓' if r.predicted == 'lower' else '  ↑')))
        sgn = 1 if r.predicted == 'higher' else -1
        m, lo, hi = sgn * r.mean_z, sgn * r.ci_lo, sgn * r.ci_hi
        lo, hi = min(lo, hi), max(lo, hi)
        ax.plot([lo, hi], [y, y], color=BLUE, lw=1.6, solid_capstyle='round', zorder=2)
        ax.scatter([m], [y], s=46, zorder=3, edgecolor=BLUE, linewidth=1.6,
                   facecolor=BLUE if r.pass_ else 'white')
        if not np.isnan(r.original_mean_z):
            ax.scatter([sgn * r.original_mean_z], [y], marker='D', s=26, color=INK, zorder=4)
    ax.axvline(0, color=MUTED, lw=1, ls='--')
    ax.set_yticks([y for y, _ in ys])
    ax.set_yticklabels([t for _, t in ys], fontsize=8.5, color=INK)
    ax.set_xlabel('mean z in the first 1,500 words, in the predicted direction (right = as predicted)',
                  fontsize=8.5, color=INK)
    ax.set_title(f'Opening register in the new corpus ({n_authors} authors)', loc='left', fontsize=10, color=INK)

    # dialogue: own panel, in share points (lower in the opening = right = as predicted)
    m, lo, hi = -pts(dlg.mean_z), -pts(dlg.ci_hi), -pts(dlg.ci_lo)
    ax2.plot([lo, hi], [0, 0], color=BLUE, lw=1.6, solid_capstyle='round', zorder=2)
    ax2.scatter([m], [0], s=46, zorder=3, edgecolor=BLUE, linewidth=1.6, facecolor=BLUE if dlg.pass_ else 'white')
    ax2.scatter([ORIGINAL_DIALOGUE_POINTS], [0], marker='D', s=26, color=INK, zorder=4)
    ax2.axvline(0, color=MUTED, lw=1, ls='--')
    ax2.set_yticks([0])
    ax2.set_yticklabels(['dialogue share  ↓'], fontsize=8.5, color=INK)
    ax2.set_ylim(-1, 1)
    ax2.set_xlabel('share points lower in the first 1,500 words than the book\'s own baseline (right = as predicted)',
                   fontsize=8.5, color=INK)
    style_axes(ax2)
    handles = [
        plt.Line2D([], [], marker='o', color=BLUE, mfc=BLUE, lw=0, label='passed (q < 0.05)'),
        plt.Line2D([], [], marker='o', color=BLUE, mfc='white', lw=0, label='did not pass'),
        plt.Line2D([], [], marker='D', color=INK, lw=0, markersize=5, label='original Victorian corpus (33 authors)'),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc='lower right')
    style_axes(ax)
    fig.text(0.01, 0.01, 'Bars = 95% interval from resampling authors. Arrow = predicted direction. '
             'Dialogue is in share points, not z (original corpus: about 17 points).', fontsize=7, color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fmt_p(p):
    return '-' if np.isnan(p) else (f'{p:.4f}' if p >= 0.0001 else '<0.0001')


def write_report(path, args, tt, ex, pl, authors, groups, info, verdict, n_pass, dlg_pass, dlg_n, elapsed):
    n_a = len(authors)
    n_books = sum(len(v) for v in groups.values())
    feat = tt[tt['feature'] != 'dialogue']
    dlg = tt[tt['feature'] == 'dialogue'].iloc[0]
    L = ['# Opening register test: results', '',
         f'Run {time.strftime("%Y-%m-%d")}. Corpus folder: `{args.corpus_dir}`. '
         f'Predictions and thresholds were frozen in config.py before any new data was analysed.', '']
    L += [f'## Verdict: **{verdict}**', '',
          f'* {n_pass} of {len(TESTED)} pre-chosen features shifted in the predicted direction '
          f'(one-sided, FDR q < {config.ALPHA}). CONFIRMED needs at least {config.CONFIRMED_MIN_PASS}; '
          f'PARTIAL is {config.PARTIAL_MIN_PASS}-{config.CONFIRMED_MIN_PASS - 1}; below that is NOT FOUND.',
          (f'* Dialogue share in the opening is lower: **{"yes" if dlg_pass else "no"}** '
           f'(mean {pts(dlg["mean_z"]):+.1f} share points vs the book\'s own baseline, '
           f'{int(dlg["authors_in_predicted_direction"])}/{int(dlg["n_authors"])} '
           f'authors lower, p = {fmt_p(dlg["p_one_sided"])}).') if dlg_n >= 3 else
          '* Dialogue could not be tested (fewer than 3 authors with detectable dialogue); the verdict cannot be CONFIRMED.',
          '']
    if verdict == 'NOT FOUND' and n_a < config.MIN_AUTHORS_WARN:
        L += [f'Only {n_a} authors: with this few, "not found" may simply mean low power.', '']
    elif n_a < config.MIN_AUTHORS_WARN:
        L += [f'Only {n_a} authors: power is low, read the intervals, not just the verdict.', '']

    L += ['## Corpus', '',
          f'{n_a} authors, {n_books} books (at least {args.min_books} books per author, at least '
          f'{args.min_words:,} words per book after trimming). Authors with detectable dialogue: {dlg_n}.']
    if info.get('excluded'):
        L += ['Excluded: ' + ', '.join(f'{r} ({n})' for r, n in info['excluded'].most_common())
              + '. Full list in exclusions.csv.']
    if info.get('styles'):
        L += ['Quote styles: ' + ', '.join(f'{k} ({v})' for k, v in sorted(info['styles'].items())) + '.']
    L += ['Before trusting this, open cuts.csv and check that each book starts at the story and not at a '
          'preface or contents page.', '']

    L += ['## The 13 tested features', '',
          'z = how far the first 1,500 words sit from the book\'s own baseline (from word 6,000 on), in typical '
          'window-to-window SDs; averaged within author. "Predicted direction" counts authors who moved as predicted.', '',
          '| feature | predicted | mean z (95% interval) | authors as predicted | q | pass | original corpus (mean z; authors) |',
          '|---|---|---:|---:|---:|---|---:|']
    for r in feat.itertuples():
        orig = f'{r.original_mean_z:+.2f}; {int(r.original_authors_same_direction)}/{ORIGINAL_N_AUTHORS}'
        L.append(f'| {r.label} | {r.predicted} | {r.mean_z:+.2f} ({r.ci_lo:+.2f} to {r.ci_hi:+.2f}) | '
                 f'{r.authors_in_predicted_direction}/{r.n_authors} | {fmt_p(r.q)} | {"**yes**" if r.pass_ else "no"} | {orig} |')
    L += ['', f'**Dialogue share** (predicted lower), in share points (not z; a z blows up for books with almost '
          f'no dialogue): mean {pts(dlg["mean_z"]):+.1f} ({pts(dlg["ci_lo"]):+.1f} to {pts(dlg["ci_hi"]):+.1f}), '
          f'{int(dlg["authors_in_predicted_direction"])}/{int(dlg["n_authors"])} authors lower, '
          f'p = {fmt_p(dlg["p_one_sided"])}. '
          'In the original corpus dialogue was about 17 share points lower than in words 6,000-12,000.', '',
          '![effects](plots/register_effects.png)', '']

    L += ['## Placebo check', '',
          f'The same test at {len(pl)} random mid-book "openings" (baseline fitted outside them). By chance a feature '
          f'passes one-sided at p < 0.05 about 5% of the time.', '',
          f'* Features with p < 0.05 per draw: mean {pl["n_features_p_below_05"].mean():.2f} of {len(TESTED)} '
          f'(about {0.05 * len(TESTED):.2f} expected).',
          f'* Features passing FDR per draw: mean {pl["n_features_pass_fdr"].mean():.2f}, maximum {int(pl["n_features_pass_fdr"].max())}.',
          f'* Draws that would have been called CONFIRMED: {int(pl["would_be_confirmed"].sum())} of {len(pl)}.', '',
          'If the placebo passes many features, the test is too loose and the verdict above should not be trusted.', '']

    L += ['## Exploratory: the other 11 original features', '',
          'Not part of the verdict. Two-sided tests, FDR across these 11.', '',
          '| feature | mean z | authors same sign as mean | p | q | significant |', '|---|---:|---:|---:|---:|---|']
    for r in ex.itertuples():
        L.append(f'| {r.label} | {r.mean_z:+.2f} | {r.authors_same_sign_as_mean}/{r.n_authors} | '
                 f'{fmt_p(r.p_two_sided)} | {fmt_p(r.q)} | {"yes" if r.significant else "no"} |')
    L += ['', '## Limits', '',
          '* Plain text loses italics, so italic thoughts or dialogue are not counted as dialogue.',
          '* Translated authors mix translator style into author style.',
          '* "Not found" can mean low power, not absence; compare the intervals with the original corpus.',
          f'* Sign-flip p-values are exact when there are {config.EXACT_MAX_AUTHORS} authors or fewer (p cannot go '
          f'below 1/2^n); with more authors they come from {config.N_PERM:,} random flips (p cannot go below about '
          f'{1 / (config.N_PERM + 1):.5f}).',
          f'* Settings: {config.WINDOW_SIZE}-word windows, opening = first {config.OPENING_WINDOWS} windows, '
          f'baseline from window {config.BASELINE_START_WINDOW}, seed {config.SEED}.',
          f'* Runtime {elapsed:.0f}s.', '']
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))


# ======================================================================== main ==
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('Writes, in')[0],
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--corpus-dir', required=True, help='folder of .txt novels')
    ap.add_argument('--out', default='output_run', help='output folder (default output_run)')
    ap.add_argument('--authors', default=None, help='optional authors.csv (filename,author)')
    ap.add_argument('--starts', default=None, help='optional starts.csv (filename,start_text)')
    ap.add_argument('--min-words', type=int, default=config.MIN_WORDS)
    ap.add_argument('--min-books', type=int, default=config.MIN_BOOKS)
    ap.add_argument('--placebo-draws', type=int, default=config.N_PLACEBO)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--rebuild', action='store_true', help='ignore the window-feature cache')
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(config.SEED)
    os.makedirs(os.path.join(args.out, 'plots'), exist_ok=True)

    # ---- stages 1 and 2
    books = load_corpus(args.corpus_dir, args.starts, args.authors, args.min_words, args.min_books)
    write_outputs(books, args.out, args)
    included = [b for b in books if b.included]
    print(f'Stage 1: {len(books)} files, {len(included)} included, {len(books) - len(included)} excluded.', flush=True)
    if not included:
        raise SystemExit('No books passed the inclusion rules; see exclusions.csv.')
    results = build_windows(included, args.out, args.rebuild, args.workers)
    write_window_outputs(included, results, args.out)

    # ---- books with enough windows, authors with enough books
    excluded = Counter(b.reason for b in books if not b.included)
    by_author = defaultdict(list)
    for b in included:
        if results[b.filename]['n_windows'] >= config.MIN_WINDOWS:
            by_author[b.author].append(b)
        else:
            excluded['too_few_windows'] += 1
    groups = {}
    for a, bs in by_author.items():
        if len(bs) >= args.min_books:
            groups[a] = [book_matrix(results[b.filename]) for b in bs]
        else:
            excluded['author_too_few_books'] += len(bs)
    if len(groups) < 3:
        raise SystemExit(f'Only {len(groups)} authors left; the test needs at least 3 (see exclusions.csv).')
    styles = Counter(results[b.filename]['style'] for a in groups for b in by_author[a])

    # ---- stage 3
    authors, A = author_opening_scores(groups)
    pd.DataFrame(A, index=authors, columns=COLS).rename_axis('author').to_csv(
        os.path.join(args.out, 'author_scores.csv'))
    tt = tested_table(authors, A, rng)
    ex = exploratory_table(A, rng)
    n_pass = int(tt.loc[tt['feature'] != 'dialogue', 'pass'].sum())
    dlg = tt[tt['feature'] == 'dialogue'].iloc[0]
    dlg_n = int(dlg['n_authors'])
    dlg_pass = bool(dlg['pass']) and dlg_n >= 3
    verdict = verdict_from(n_pass, dlg_pass)
    print(f'{len(authors)} authors: {n_pass} of {len(TESTED)} features pass; dialogue lower: {dlg_pass}. '
          f'Verdict: {verdict}.', flush=True)
    print('Placebo check...', flush=True)
    pl = placebo(groups, rng, args.placebo_draws)

    tt.rename(columns={'pass': 'passed'}).to_csv(os.path.join(args.out, 'register_results.csv'), index=False)
    ex.to_csv(os.path.join(args.out, 'exploratory_results.csv'), index=False)
    pl.to_csv(os.path.join(args.out, 'placebo.csv'), index=False)
    tt_plot = tt.rename(columns={'pass': 'pass_'})
    plot_effects(tt_plot, os.path.join(args.out, 'plots', 'register_effects.png'), len(authors))
    write_report(os.path.join(args.out, 'register_report.md'), args, tt_plot, ex, pl, authors, groups,
                 {'excluded': excluded, 'styles': styles}, verdict, n_pass, dlg_pass, dlg_n, time.time() - t0)
    print(f'Wrote register_report.md (read this first), register_results.csv, exploratory_results.csv, '
          f'author_scores.csv, placebo.csv, plots/register_effects.png to {args.out}/ ({time.time() - t0:.0f}s).')


if __name__ == '__main__':
    main()
