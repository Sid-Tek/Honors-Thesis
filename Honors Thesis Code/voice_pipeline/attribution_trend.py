"""
Attribution trend test: does author-attribution accuracy keep rising after the opening register?

Reads output_attrib/trials.csv (written by attribution_by_position.py); runs nothing slow.

Why: attribution_by_position.py compared three coarse bins (2-6k vs 12-30k, ...). Binning throws away
information. This script instead fits ONE straight line to accuracy against position, using every
passage position from START_POS onward, and asks whether the slope is above zero. Authors are
resampled (bootstrap) because an author's three books are not independent.

Slope units: percentage points of accuracy per 10,000 words of position.
  slope > 0 with an interval above 0 -> accuracy keeps rising: author signal strengthens with position.
  interval around 0                   -> no evidence of a rise; read the interval's upper end:
                                         rises larger than that are ruled out by this data.
  slope < 0 with an interval below 0  -> accuracy falls with position.

Training conditions read (from the trials file):
  all        whole-book training (the practical case). CAVEAT: whole books are mostly middle and late
             text, so late passages may match the training text better for reasons unrelated to voice.
  all_noreg  as 'all' without the opening-register function words.
  noopen     as 'all' without each training book's first 3,000 words.
  matched    training text taken from the SAME position in the other books, equal amount at every
             position. This is the cleanest condition for "is author signal stronger at position p":
             training amount and position are held fixed, only the position of the test passage varies.
  unmatched  equal amounts of text from random later positions (a control for the matched condition).

Sensitivity: the slope is refitted from several start positions (2k, 4k, 6k, 8k, 12k), because the
register ends around 3,000 words and the answer should not depend on exactly where the line starts.

Also reported: the share of authors whose own slope is positive (sign test), and the smallest slope
this data could reliably detect (about 2.8 x the standard error, for 80% power at 5% two-sided).

Run (from the voice_pipeline folder):
    python attribution_trend.py
    python attribution_trend.py --trials output_attrib/trials.csv --out output_attrib_trend --start 4000

Outputs (output_attrib_trend/): trend_report.md, trend_slopes.csv, plots/attribution_trend.png
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SENS_STARTS = [2000, 4000, 6000, 8000, 12000]
COND_LABELS = {'all': 'whole-book training', 'all_noreg': 'whole books, register words removed',
               'noopen': 'whole books, first 3,000 words dropped', 'matched': 'same-position training (equal amount)',
               'unmatched': 'random later training (equal amount)'}
INK, MUTED, GRID = '#333333', '#8a8a8a', '#e6e6e6'
COLORS = {'all': '#2a78d6', 'matched': '#eb6834', 'noopen': '#2a9d8f', 'all_noreg': '#1b4f8f', 'unmatched': '#b5b5b5'}


def author_matrix(trials, cond, method, measure):
    """authors x positions matrix of mean accuracy (or reciprocal rank), averaged over folds."""
    t = trials[(trials['cond'] == cond) & (trials['method'] == method)]
    m = t.pivot_table(index='author', columns='pos', values=measure, aggfunc='mean')
    return m.sort_index(axis=1)


def slope_of(curve_rows, x):
    """OLS slope for each row of curve_rows (R x P) against x (P,)."""
    xc = x - x.mean()
    return (curve_rows - curve_rows.mean(axis=1, keepdims=True)) @ xc / (xc @ xc)


def trend(M, start, n_boot, rng):
    cols = M.columns.values.astype(float)
    use = cols >= start
    x = cols[use]
    Y = M.values[:, use]                                   # A x P
    A = Y.shape[0]
    est = slope_of(Y.mean(axis=0, keepdims=True), x)[0] * 10000
    idx = rng.integers(0, A, size=(n_boot, A))
    boot = np.array([slope_of(Y[i].mean(axis=0, keepdims=True), x)[0] for i in idx]) * 10000
    own = slope_of(Y, x) * 10000                           # each author's own slope
    n_pos = int((own > 0).sum())
    n_neg = int((own < 0).sum())
    from math import comb
    k = max(n_pos, n_neg)
    n = n_pos + n_neg
    p_sign = min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n) if n else float('nan')
    se = float(boot.std(ddof=1))
    return dict(start=start, n_positions=int(use.sum()), slope=est, lo=float(np.percentile(boot, 2.5)),
                hi=float(np.percentile(boot, 97.5)), se=se, min_detectable=2.8 * se, authors=A,
                authors_up=n_pos, authors_down=n_neg, sign_p=p_sign,
                span=float(x.max() - x.min()), mean_level=float(Y.mean()))


def verdict(r):
    if r['lo'] > 0:
        return 'rising'
    if r['hi'] < 0:
        return 'falling'
    return 'no clear trend'


def fmt(v, pct=True):
    return f'{v*100:+.1f}' if pct else f'{v:+.2f}'


def plot(trials, args, path):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    for ax, conds, title in ((axes[0], ['all', 'noopen'], 'Whole-book training'),
                             (axes[1], ['matched', 'unmatched'], 'Equal amounts of training text')):
        for c in conds:
            M = author_matrix(trials, c, 'delta', 'correct')
            xs = M.columns.values.astype(float)
            mean = M.values.mean(axis=0)
            ax.plot(xs / 1000, mean, color=COLORS[c], lw=2, label=COND_LABELS[c])
            use = xs >= args.start
            if use.sum() >= 3:
                b, a = np.polyfit(xs[use], mean[use], 1)
                ax.plot(xs[use] / 1000, a + b * xs[use], color=COLORS[c], lw=1.2, ls='--', alpha=.8)
        ax.axvline(args.start / 1000, color=MUTED, lw=1, ls=':')
        ax.set_title(title, loc='left', fontsize=10.5, color=INK)
        ax.set_xlabel('start of the 2,000-word test passage (thousand words)', color=INK, fontsize=9.5)
        ax.grid(axis='y', color=GRID, lw=.8)
        ax.legend(frameon=False, fontsize=8.5, loc='lower right')
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel('share of passages attributed to the right author', color=INK, fontsize=9.5)
    fig.text(.01, .01, f'Burrows\' Delta; mean over authors; dashed = straight-line fit from {args.start/1000:.0f}k onward (dotted vertical line).',
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('Run (from')[0], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--trials', default=os.path.join('output_attrib', 'trials.csv'))
    ap.add_argument('--out', default='output_attrib_trend')
    ap.add_argument('--start', type=int, default=4000, help='first passage position used in the main fit')
    ap.add_argument('--boot', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=20261010)
    args = ap.parse_args()
    if not os.path.exists(args.trials):
        raise SystemExit(f'{args.trials} not found. Run attribution_by_position.py first.')
    os.makedirs(os.path.join(args.out, 'plots'), exist_ok=True)
    trials = pd.read_csv(args.trials)
    rng = np.random.default_rng(args.seed)
    conds = [c for c in ('all', 'all_noreg', 'noopen', 'matched', 'unmatched') if c in set(trials['cond'])]
    n_auth = trials['author'].nunique()

    rows = []
    for method in ('delta', 'cosine'):
        for measure, mlabel in (('correct', 'accuracy'), ('rr', 'mean reciprocal rank')):
            for c in conds:
                M = author_matrix(trials, c, method, measure)
                for s in SENS_STARTS:
                    if (M.columns.values >= s).sum() < 4:
                        continue
                    r = trend(M, s, args.boot, rng)
                    r.update(method=method, measure=mlabel, cond=c)
                    rows.append(r)
    res = pd.DataFrame(rows)
    res['verdict'] = res.apply(verdict, axis=1)
    res.to_csv(os.path.join(args.out, 'trend_slopes.csv'), index=False)
    plot(trials, args, os.path.join(args.out, 'plots', 'attribution_trend.png'))

    # ---- report ----
    def pick(method, measure, cond, start):
        q = res[(res.method == method) & (res.measure == measure) & (res.cond == cond) & (res.start == start)]
        return q.iloc[0] if len(q) else None

    L = ['# Attribution trend test: does accuracy keep rising after the opening?', '',
         f'Run {pd.Timestamp.now():%Y-%m-%d}. {n_auth} authors; passages from {int(trials["pos"].min()):,} to {int(trials["pos"].max()):,} words; '
         f'authors resampled {args.boot:,} times. Slopes are percentage points of accuracy per 10,000 words of position, '
         f'from a straight line fitted to every passage position at or after the start shown.', '']
    for method, mname in (('delta', "Burrows' Delta (primary)"), ('cosine', 'Cosine Delta (check)')):
        L += [f'## {mname}: main fit, from {args.start:,} words', '',
              '| training | accuracy slope (per 10k words) | 95% interval | verdict | authors rising / falling | smallest slope detectable |',
              '|---|---:|---|---|---:|---:|']
        for c in conds:
            r = pick(method, 'accuracy', c, args.start)
            if r is None:
                continue
            L.append(f'| {COND_LABELS[c]} | {fmt(r.slope)} | {fmt(r.lo)} to {fmt(r.hi)} | **{r.verdict}** | '
                     f'{r.authors_up} / {r.authors_down} (sign test p = {r.sign_p:.2f}) | {fmt(r.min_detectable).lstrip("+")} |')
        L += ['']
    L += ['## Sensitivity to where the line starts (Delta, accuracy slope per 10k words, 95% interval)', '',
          '| training | ' + ' | '.join(f'from {s//1000}k' for s in SENS_STARTS) + ' |',
          '|---|' + '---|' * len(SENS_STARTS)]
    for c in conds:
        cells = []
        for s in SENS_STARTS:
            r = pick('delta', 'accuracy', c, s)
            cells.append('-' if r is None else f'{fmt(r.slope)} ({fmt(r.lo)} to {fmt(r.hi)})')
        L.append(f'| {COND_LABELS[c]} | ' + ' | '.join(cells) + ' |')
    L += ['', '## Same, with mean reciprocal rank (a finer measure than right/wrong; slope per 10k words, x100)', '',
          '| training | ' + ' | '.join(f'from {s//1000}k' for s in SENS_STARTS) + ' |',
          '|---|' + '---|' * len(SENS_STARTS)]
    for c in conds:
        cells = []
        for s in SENS_STARTS:
            r = pick('delta', 'mean reciprocal rank', c, s)
            cells.append('-' if r is None else f'{fmt(r.slope)} ({fmt(r.lo)} to {fmt(r.hi)})')
        L.append(f'| {COND_LABELS[c]} | ' + ' | '.join(cells) + ' |')

    # plain reading
    ra, rm = pick('delta', 'accuracy', 'all', args.start), pick('delta', 'accuracy', 'matched', args.start)
    L += ['', '## Plain reading', '']
    for name, r in (('Whole-book training', ra), ('Same-position training (the cleaner test)', rm)):
        if r is None:
            continue
        span = r.span / 10000
        if r.verdict == 'rising':
            txt = (f'accuracy rises by about {fmt(r.slope)} points per 10,000 words (interval {fmt(r.lo)} to {fmt(r.hi)}), '
                   f'roughly {fmt(r.slope*span)} points across the range tested.')
        elif r.verdict == 'falling':
            txt = f'accuracy falls by about {fmt(-r.slope).lstrip("+")} points per 10,000 words (interval {fmt(r.lo)} to {fmt(r.hi)}).'
        else:
            txt = (f'no clear trend: the slope is {fmt(r.slope)} points per 10,000 words (interval {fmt(r.lo)} to {fmt(r.hi)}). '
                   f'A rise steeper than about {fmt(r.hi).lstrip("+")} points per 10,000 words (about {fmt(r.hi*span).lstrip("+")} points over the whole range) '
                   f'is ruled out; a smaller rise is not.')
        L.append(f'* **{name}:** {txt}')
    L += ['',
          'How to use this: the claim "voice keeps building after the opening register" predicts a clearly positive slope, '
          'especially in the same-position condition. A positive whole-book slope with a flat same-position slope would mean the rise '
          'comes from the training text being mostly late text, not from voice strengthening. A flat slope in both is evidence against '
          'build-up (within the limits stated by the interval); a wide interval that includes both zero and a large rise means the data cannot decide.',
          '', 'Limits: one corpus of ~30 authors, three books each; straight-line fit (a rise that levels off would be understated); '
          'accuracy at a single position is noisy, which is why the whole range is used.']
    with open(os.path.join(args.out, 'trend_report.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')

    print(f'Trend test ({n_auth} authors), Delta, accuracy slope per 10,000 words, line fitted from {args.start:,} words:')
    for c in conds:
        r = pick('delta', 'accuracy', c, args.start)
        if r is not None:
            print(f'  {COND_LABELS[c]:42} {fmt(r.slope):>6} pts  ({fmt(r.lo)} to {fmt(r.hi)})  -> {r.verdict}')
    print(f'\nReport: {os.path.join(args.out, "trend_report.md")}')


if __name__ == '__main__':
    main()
