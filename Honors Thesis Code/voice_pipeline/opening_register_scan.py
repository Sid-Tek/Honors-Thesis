"""
Step 1 of the convention -> voice experiment: the OPENING REGISTER SCAN.

Question: which of the 24 features shift in the opening of a novel the same
way for everyone (the shared "opening register"), and where does each shift
end?

This is a RAW-feature question, not a distinctiveness one: no author
profiles or backgrounds are involved. Every book is compared with itself.

Method
------
1. Front matter is trimmed first (front_matter.py): title pages, contents,
   dedications, prefaces and chapter-1 epigraphs are cut, so window 0 is the
   first 300 words of the story. Earlier opening tests ran on the untrimmed
   files; the untrimmed version is re-run here for comparison.
2. Each book is cut into the pipeline's usual 300-word windows and the 24
   features are extracted per window (features.extract_features).
3. Per book and feature, the "rest of the book" baseline is a straight line
   fitted to every window from position BASE_START (6,000 words) onward.
   Each window's value becomes a z-score: (value - baseline) / SD of the
   baseline residuals. So z = +1 at position p means "this stretch is one
   typical window-to-window SD above what the rest of this book would
   predict here". The opening never influences its own baseline.
4. Authors are the unit. Each author's 3 books are averaged position by
   position (33 authors; Anon has only one book and is left out, as in
   every other analysis).
5. REGISTER TEST, per feature: the mean z over the first 1,500 words
   (windows 0-4, the span the Sep 23 word-length check pointed to), tested
   against 0 across authors with a sign-flip permutation test (two-sided,
   20,000 flips). Benjamini-Hochberg across the 24 features.
   600-word and 3,000-word versions are reported alongside as sensitivity.
6. WHERE IT ENDS, per feature that passes: a "fade-out" line is fitted to
   the corpus-average profile over the first 12,000 words: the offset
   starts at some height at word 0, falls in a straight line to zero at
   word E, and stays at zero after. E is the end of the register for that
   feature. A 95% interval for E comes from resampling authors (2,000
   bootstrap draws). A step model (constant offset that switches off at E)
   is reported too, as a check that the answer doesn't hinge on the shape.
7. PLACEBO: the same register test run at random mid-book positions (50
   draws). It should flag about 5% of features per draw at p < .05 and
   almost none after FDR. This checks the test isn't simply loose.

Run:  python opening_register_scan.py
Outputs (output_register/):
  front_matter_cuts.csv     where each book's narrative was taken to start
  register_tests.csv        one row per feature: effect, p, q, sensitivity, untrimmed
  register_boundaries.csv   end point E (+95% CI) per register feature
  position_profiles.csv     corpus-average z at every position, with 95% CI
  author_opening_means.csv  each author's mean z over the first 1,500 words
  placebo_calibration.csv
  plots/register_profiles.png, plots/register_ends.png
  opening_register_report.md
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from windowing import load_text, parse_sentences, local_windows
from features import extract_features, FEATURE_NAMES
from front_matter import load_trimmed, MANUAL_START
from run_full_corpus import discover_authors, CORPUS_DIR, WINDOW_SIZE

OUT = 'output_register'
CACHE = os.path.join(OUT, 'window_features.npz')
N_POS = 40            # profile length: 40 windows = 12,000 words
BASE_START = 20       # baseline fitted from window 20 (6,000 words) onward
OPEN_SPANS = {'600': 2, '1500': 5, '3000': 10}
PRIMARY = '1500'
N_FLIP = 20000
N_BOOT = 2000
N_PLACEBO = 50
ALPHA = 0.05
SEED = 20261003

LABELS = {
    'sent_len_mean': 'sentence length', 'sent_len_std': 'sentence-length spread',
    'ttr': 'type-token ratio (MATTR)', 'hapax_rate': 'hapax rate (MA)',
    'avg_word_len': 'word length (letters)', 'comma_rate': 'commas',
    'semicolon_rate': 'semicolons', 'dash_rate': 'dashes',
    'question_rate': 'questions', 'exclaim_rate': 'exclamations',
    'subordination_rate': 'subordinators', 'coordination_rate': 'coordinators',
    'syll_per_word': 'syllables per word', 'pct_monosyllabic': 'one-syllable words',
    'fw_the': '"the"', 'fw_and': '"and"', 'fw_of': '"of"', 'fw_to': '"to"',
    'fw_in': '"in"', 'fw_that': '"that"', 'fw_it': '"it"', 'fw_was': '"was"',
    'fw_but': '"but"', 'fw_as': '"as"',
}

INK, MUTED, GRID = '#333333', '#8a8a8a', '#e6e6e6'
C_UP, C_DOWN, C_NONE = '#2a78d6', '#eb6834', '#b5b5b5'


# ---------------------------------------------------------------- data ----
def _book_matrix(args):
    fn, trimmed = args
    path = os.path.join(CORPUS_DIR, fn)
    if trimmed:
        text, start, method, raw = load_trimmed(path, fn)
    else:
        text, start, method = load_text(path), 0, 'untrimmed'
    wins = local_windows(parse_sentences(text), WINDOW_SIZE)
    m = np.array([[extract_features(w)[f] for f in FEATURE_NAMES] for w in wins])
    return fn, trimmed, m, start, method


def load_window_features(rebuild=False):
    if not rebuild and os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        print(f'Loading cached window features from {CACHE}', flush=True)
        return z['mats'].item(), pd.read_csv(os.path.join(OUT, 'front_matter_cuts.csv'))
    by_author = discover_authors()
    books = [fn for a, bs in by_author.items() if len(bs) >= 2 for _, fn in bs]
    jobs = [(fn, t) for fn in books for t in (True, False)]
    mats, cuts = {}, []
    t0 = time.time()
    print(f'Extracting window features: {len(books)} books x (trimmed, untrimmed) = {len(jobs)} jobs '
          f'(one-time; cached afterwards)...', flush=True)
    with ProcessPoolExecutor() as ex:
        for i, (fn, trimmed, m, start, method) in enumerate(ex.map(_book_matrix, jobs), 1):
            print(f'  [{i}/{len(jobs)}] {fn} ({"trimmed" if trimmed else "untrimmed"}, '
                  f'{len(m)} windows) {time.time()-t0:.0f}s', flush=True)
            mats[(fn, trimmed)] = m
            if trimmed:
                with open(os.path.join(CORPUS_DIR, fn), encoding='utf-8', errors='ignore') as f:
                    raw = f.read().lstrip('﻿')
                cut_words = len(raw[:start].split())
                after = ' '.join(raw[start:start + 400].split())[:120]
                cuts.append({'book': fn, 'method': method, 'cut_char': start,
                             'words_trimmed': cut_words, 'n_windows': len(m),
                             'manual_reason': MANUAL_START.get(fn, ('', ''))[1],
                             'narrative_starts': after})
    cuts = pd.DataFrame(cuts).sort_values('book')
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(CACHE, mats=np.array(mats, dtype=object))
    cuts.to_csv(os.path.join(OUT, 'front_matter_cuts.csv'), index=False)
    return mats, cuts


def book_z(m, start=0, exclude=None):
    """z-scores of every window vs. a linear 'rest of book' baseline.
    Baseline uses windows >= start+BASE_START (and outside `exclude`, a
    (lo, hi) window range, for the placebo). Returns (n_windows x n_feat)."""
    n = len(m)
    x = np.arange(n)
    use = x >= start + BASE_START
    if exclude is not None:
        use = (x < exclude[0]) | (x >= exclude[1])
    z = np.full(m.shape, np.nan)
    for j in range(m.shape[1]):
        y = m[:, j]
        coef = np.polyfit(x[use], y[use], 1)
        r = y - np.polyval(coef, x)
        sd = r[use].std(ddof=2)
        if sd > 1e-12:
            z[:, j] = r / sd
    return z


def author_profiles(mats, trimmed=True):
    """33 x N_POS x 24 array of author-mean z, plus the author list."""
    by_author = discover_authors()
    authors = [a for a, bs in by_author.items() if len(bs) >= 2]
    A = np.full((len(authors), N_POS, len(FEATURE_NAMES)), np.nan)
    for i, a in enumerate(authors):
        zs = [book_z(mats[(fn, trimmed)])[:N_POS] for _, fn in by_author[a]]
        A[i] = np.nanmean(np.stack(zs), axis=0)
    return authors, A


# --------------------------------------------------------------- stats ----
def signflip_p(v, rng, n=N_FLIP):
    """Two-sided sign-flip permutation p for mean(v) != 0 (authors as units)."""
    v = v[~np.isnan(v)]
    obs = abs(v.mean())
    signs = rng.choice([-1.0, 1.0], size=(n, len(v)))
    null = np.abs((signs * v).mean(axis=1))
    return (1 + np.sum(null >= obs - 1e-12)) / (n + 1)


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1)
    return out


GRID_E = np.arange(1.0, N_POS - 1 + 1e-9, 0.25)   # end point, in windows


def _fit(profile, shape):
    """Least-squares fit of baseline b + a * g(p; E). Returns (E, a, b, sse)."""
    p = np.arange(len(profile))
    best = (np.nan, np.nan, np.nan, np.inf)
    for E in GRID_E:
        if shape == 'fade':
            # window p covers [p, p+1); average the fade over it
            g = np.clip(1 - (p + 0.5) / E, 0, None)
        else:
            g = (p < E).astype(float) if E == int(E) else np.clip(E - p, 0, 1)
        X = np.column_stack([g, np.ones_like(g)])
        coef, *_ = np.linalg.lstsq(X, profile, rcond=None)
        sse = float(np.sum((profile - X @ coef) ** 2))
        if sse < best[3]:
            best = (E, coef[0], coef[1], sse)
    return best


def end_point(Af, rng, shape='fade'):
    """Af: authors x positions for one feature. Point estimate + bootstrap CI
    of the end point E (in words)."""
    prof = np.nanmean(Af, axis=0)
    E, a, b, _ = _fit(prof, shape)
    boots = []
    n = len(Af)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        boots.append(_fit(np.nanmean(Af[idx], axis=0), shape)[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return E * WINDOW_SIZE, lo * WINDOW_SIZE, hi * WINDOW_SIZE, a, np.array(boots) * WINDOW_SIZE


def register_tests(A, rng):
    rows = []
    for j, f in enumerate(FEATURE_NAMES):
        r = {'feature': f, 'label': LABELS[f]}
        for name, k in OPEN_SPANS.items():
            v = np.nanmean(A[:, :k, j], axis=1)
            r[f'mean_z_{name}'] = np.nanmean(v)
            r[f'p_{name}'] = signflip_p(v, rng)
            if name == PRIMARY:
                r['n_authors'] = int(np.sum(~np.isnan(v)))
                r['n_authors_same_sign'] = int(np.sum(np.sign(v) == np.sign(np.nanmean(v))))
        rows.append(r)
    df = pd.DataFrame(rows)
    for name in OPEN_SPANS:
        df[f'q_{name}'] = bh(df[f'p_{name}'])
    return df


def placebo(mats, rng):
    """Register test at random mid-book pseudo-openings."""
    by_author = discover_authors()
    authors = [a for a, bs in by_author.items() if len(bs) >= 2]
    k = OPEN_SPANS[PRIMARY]
    rows = []
    for d in range(N_PLACEBO):
        V = np.full((len(authors), len(FEATURE_NAMES)), np.nan)
        for i, a in enumerate(authors):
            vals = []
            for _, fn in by_author[a]:
                m = mats[(fn, True)]
                n = len(m)
                s = int(rng.integers(int(0.3 * n), int(0.7 * n) - k))
                z = book_z(m, exclude=(s, s + BASE_START))
                vals.append(z[s:s + k].mean(axis=0))
            V[i] = np.nanmean(vals, axis=0)
        p = np.array([signflip_p(V[:, j], rng, 5000) for j in range(V.shape[1])])
        if (d + 1) % 10 == 0:
            print(f'  placebo draw {d + 1}/{N_PLACEBO}', flush=True)
        rows.append({'draw': d, 'n_p_below_05': int(np.sum(p < ALPHA)),
                     'n_q_below_05': int(np.sum(bh(p) < ALPHA))})
    return pd.DataFrame(rows)


def profile_table(A, tag, rng):
    rows = []
    for j, f in enumerate(FEATURE_NAMES):
        Af = A[:, :, j]
        mean = np.nanmean(Af, axis=0)
        n = len(Af)
        boots = np.array([np.nanmean(Af[rng.integers(0, n, n)], axis=0) for _ in range(1000)])
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        for p in range(N_POS):
            rows.append({'version': tag, 'feature': f, 'position': p,
                         'start_word': p * WINDOW_SIZE, 'mean_z': mean[p],
                         'ci_lo': lo[p], 'ci_hi': hi[p]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- plots ----
def _style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=7)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_profiles(prof, tests, ends, path):
    order = tests.sort_values('mean_z_1500', key=np.abs, ascending=False)['feature'].tolist()
    fig, axes = plt.subplots(6, 4, figsize=(11, 13), sharex=True, sharey=True)
    for ax, f in zip(axes.flat, order):
        t = tests.set_index('feature').loc[f]
        sig = t['q_1500'] < ALPHA
        col = (C_UP if t['mean_z_1500'] > 0 else C_DOWN) if sig else C_NONE
        d = prof[(prof['feature'] == f) & (prof['version'] == 'trimmed')]
        u = prof[(prof['feature'] == f) & (prof['version'] == 'untrimmed')]
        x = (d['position'] + 0.5) * WINDOW_SIZE / 1000
        ax.fill_between(x, d['ci_lo'], d['ci_hi'], color=col, alpha=0.18, linewidth=0)
        ax.plot(x, u['mean_z'], color=MUTED, lw=1, ls=':', label='untrimmed')
        ax.plot(x, d['mean_z'], color=col, lw=2, label='trimmed')
        ax.axhline(0, color=MUTED, lw=0.8)
        if sig and f in ends.index:
            ax.axvline(ends.loc[f, 'end_words'] / 1000, color=INK, lw=0.8, ls='--')
        star = ' *' if sig else ''
        ax.set_title(f"{LABELS[f]}{star}\nfirst 1,500 words: z = {t['mean_z_1500']:+.2f}, q = {t['q_1500']:.3f}",
                     fontsize=8, color=INK, loc='left')
        _style(ax)
    for ax in axes[-1]:
        ax.set_xlabel('thousand words into the narrative', fontsize=7, color=INK)
    for ax in axes[:, 0]:
        ax.set_ylabel('z vs. rest of book', fontsize=7, color=INK)
    axes.flat[0].set_ylim(-1.0, 1.0)
    axes.flat[0].legend(frameon=False, fontsize=7, loc='upper right')
    fig.suptitle('Opening register: corpus-average position profile per feature (33 authors, 95% band)\n'
                 'colour = shifts in the first 1,500 words after FDR (blue up, orange down); '
                 'dashed line = estimated end of the shift',
                 x=0.01, ha='left', fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ends(ends, path):
    e = ends.sort_values('end_words')
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(e) + 1.2))
    y = np.arange(len(e))
    cols = [C_UP if v > 0 else C_DOWN for v in e['mean_z_1500']]
    ax.hlines(y, e['ci_lo'] / 1000, e['ci_hi'] / 1000, color=cols, lw=2)
    ax.scatter(e['end_words'] / 1000, y, color=cols, s=40, zorder=3, edgecolor='white', linewidth=1)
    ax.scatter(e['step_end_words'] / 1000, y, marker='|', color=INK, s=60, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{LABELS[f]} ({'up' if v > 0 else 'down'})" for f, v in zip(e.index, e['mean_z_1500'])],
                       fontsize=8, color=INK)
    ax.set_xlabel('words into the narrative where the shift ends\n'
                  'dot = fade-out fit, bar = 95% interval over authors, tick = step-model end',
                  fontsize=8, color=INK)
    ax.set_xlim(0, N_POS * WINDOW_SIZE / 1000)
    ax.set_title('Where each part of the opening register ends', loc='left', fontsize=10, color=INK)
    _style(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.yaxis.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -------------------------------------------------------------- report ----
def write_report(tests, ends, cuts, plc, path):
    sig = tests[tests['q_1500'] < ALPHA].sort_values('mean_z_1500', key=np.abs, ascending=False)
    L = ['# Opening register scan (step 1)', '',
         f'Run {time.strftime("%Y-%m-%d")}. 33 authors x 3 books, front matter trimmed. '
         'z = how far a 300-word stretch sits from the rest of its own book (in typical '
         'window-to-window SDs); authors are the unit.', '',
         '## Which features shift in the opening', '',
         '| feature | first 1,500 words, mean z | authors same direction | p | q (FDR, 24) | first 600 words | first 3,000 words | untrimmed, first 1,500 |',
         '|---|---:|---:|---:|---:|---:|---:|---:|']
    for _, r in tests.sort_values('p_1500').iterrows():
        mark = '**' if r['q_1500'] < ALPHA else ''
        L.append(f"| {mark}{r['label']}{mark} | {r['mean_z_1500']:+.2f} | {r['n_authors_same_sign']}/{r['n_authors']} | "
                 f"{r['p_1500']:.4f} | {r['q_1500']:.3f} | {r['mean_z_600']:+.2f} (q={r['q_600']:.3f}) | "
                 f"{r['mean_z_3000']:+.2f} (q={r['q_3000']:.3f}) | {r['u_mean_z_1500']:+.2f} (q={r['u_q_1500']:.3f}) |")
    L += ['', f'**{len(sig)} of 24 features** shift in the first 1,500 words after FDR correction.', '',
          '## Where each shift ends', '',
          '| feature | direction | ends at (fade-out fit) | 95% interval | step-model end |',
          '|---|---|---:|---:|---:|']
    for f, r in ends.sort_values('end_words').iterrows():
        L.append(f"| {LABELS[f]} | {'up' if r['mean_z_1500'] > 0 else 'down'} | {r['end_words']:,.0f} | "
                 f"{r['ci_lo']:,.0f}-{r['ci_hi']:,.0f} | {r['step_end_words']:,.0f} |")
    L += ['', '## Calibration', '',
          f'Placebo (same test at random mid-book positions, {len(plc)} draws x 24 features): '
          f"{plc['n_p_below_05'].mean() / 24:.1%} of tests at p < .05 (expect ~5%); "
          f"draws with any FDR-significant feature: {(plc['n_q_below_05'] > 0).sum()} of {len(plc)}.", '',
          '## Front matter', '',
          f"Median {cuts['words_trimmed'].median():,.0f} words trimmed per book; "
          f"{(cuts['words_trimmed'] >= 300).sum()} of {len(cuts)} books had at least one full window (300 words) "
          f"of front matter; largest: " +
          ', '.join(f"{r.book} ({r.words_trimmed:,})" for r in cuts.nlargest(5, 'words_trimmed').itertuples()) + '.',
          '', 'See front_matter_cuts.csv for every cut.']
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')


# ---------------------------------------------------------------- main ----
def main():
    t0 = time.time()
    os.makedirs(os.path.join(OUT, 'plots'), exist_ok=True)
    rng = np.random.default_rng(SEED)
    mats, cuts = load_window_features()
    print(f'Features ready ({time.time()-t0:.0f}s). Running register tests and end-point bootstraps...', flush=True)

    authors, A = author_profiles(mats, trimmed=True)
    _, U = author_profiles(mats, trimmed=False)

    print('  author profiles built; testing trimmed and untrimmed openings...', flush=True)
    tests = register_tests(A, rng)
    ut = register_tests(U, rng)
    for c in ('mean_z_1500', 'p_1500', 'q_1500'):
        tests['u_' + c] = ut[c].values

    k = OPEN_SPANS[PRIMARY]
    pd.DataFrame(np.nanmean(A[:, :k, :], axis=1), index=authors, columns=FEATURE_NAMES) \
        .to_csv(os.path.join(OUT, 'author_opening_means.csv'), index_label='author')

    sig = tests[tests['q_1500'] < ALPHA]['feature'].tolist()
    ends = []
    print(f'  {len(sig)} of 24 features shift in the first 1,500 words (q < {ALPHA}): {", ".join(sig) or "none"}', flush=True)
    for n_f, f in enumerate(sig, 1):
        print(f'  end point {n_f}/{len(sig)}: {f} ({N_BOOT} bootstrap draws x 2 shapes)', flush=True)
        j = FEATURE_NAMES.index(f)
        E, lo, hi, a, _ = end_point(A[:, :, j], rng, 'fade')
        Es, *_ = end_point(A[:, :, j], rng, 'step')
        ends.append({'feature': f, 'end_words': E, 'ci_lo': lo, 'ci_hi': hi,
                     'start_height_z': a, 'step_end_words': Es,
                     'mean_z_1500': tests.set_index('feature').loc[f, 'mean_z_1500']})
    ends = pd.DataFrame(ends).set_index('feature') if ends else pd.DataFrame(
        columns=['end_words', 'ci_lo', 'ci_hi', 'start_height_z', 'step_end_words', 'mean_z_1500'])
    print(f'Tests + end points done ({time.time()-t0:.0f}s). Building profiles and running placebo ({N_PLACEBO} draws)...', flush=True)

    prof = pd.concat([profile_table(A, 'trimmed', rng), profile_table(U, 'untrimmed', rng)])
    plc = placebo(mats, rng)
    print(f'Placebo done ({time.time()-t0:.0f}s). Writing outputs...', flush=True)

    tests.to_csv(os.path.join(OUT, 'register_tests.csv'), index=False)
    ends.to_csv(os.path.join(OUT, 'register_boundaries.csv'))
    prof.to_csv(os.path.join(OUT, 'position_profiles.csv'), index=False)
    plc.to_csv(os.path.join(OUT, 'placebo_calibration.csv'), index=False)
    plot_profiles(prof, tests, ends, os.path.join(OUT, 'plots', 'register_profiles.png'))
    if len(ends):
        plot_ends(ends, os.path.join(OUT, 'plots', 'register_ends.png'))
    write_report(tests, ends, cuts, plc, os.path.join(OUT, 'opening_register_report.md'))
    print(f'Outputs written to {OUT}/ (report: opening_register_report.md)', flush=True)
    pd.set_option('display.width', 220)
    print(tests[['feature', 'mean_z_600', 'q_600', 'mean_z_1500', 'n_authors_same_sign', 'q_1500',
                 'mean_z_3000', 'q_3000', 'u_mean_z_1500', 'u_q_1500']].sort_values('q_1500').to_string(index=False))
    print(ends.to_string())
    print(plc.describe().to_string())
    print(f'done ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
