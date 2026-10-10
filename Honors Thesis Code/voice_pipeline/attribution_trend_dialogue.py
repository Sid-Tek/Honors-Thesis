"""
Attribution trend, controlling for dialogue.

Question: attribution_trend.py found a small, slow rise in attribution accuracy with position
(about +3 to +4 points per 10,000 words, borderline). Later passages also contain more dialogue
than the opening, and dialogue changes which common words appear (dialogue_share_check.py). Does the
rise survive once each test passage's dialogue share is taken into account?

Method
------
1. Dialogue share of every test passage: the share of its 2,000 words that sit inside quotation marks
   (same parser as dialogue_share_check.py). Books with no detectable dialogue (Galsworthy_River,
   Morris_Water in the earlier run) are left out of this analysis; their passages have no usable share.
2. For passages at or after START (default 4,000 words), fit by least squares, with a separate
   baseline for each author (fixed effects, so author skill differences cannot drive the slope):
       A  outcome ~ position                                   (the unadjusted slope, same passages)
       B  outcome ~ position + dialogue share + share squared  (the slope with dialogue held fixed)
   The outcome is 1 / 0 for "attributed to the right author" (or the reciprocal rank).
3. Authors are resampled 5,000 times for every interval, as in attribution_trend.py.

Reading
-------
  B stays close to A and its interval stays above 0
        the rise is not explained by dialogue: author-specific signal grows with position.
  B much smaller than A (say under half) with an interval around 0
        most of the apparent rise was the changing mix of dialogue and narration.
  Either way, dialogue is only ONE aspect of scene mix. A surviving slope is not proof of 'voice';
  it is evidence the rise is not simply more quoted speech.
Slopes are percentage points of accuracy per 10,000 words of position.
'Dialogue effect' = the change in accuracy (points) for a passage with 10 percentage points more
dialogue, at the average dialogue share.

Run (from the voice_pipeline folder):
    python attribution_trend_dialogue.py
    python attribution_trend_dialogue.py --corpus-dir real_corpus --trials output_attrib/trials.csv --start 4000

Outputs (output_attrib_trend_dialogue/): trend_dialogue_report.md, trend_dialogue_slopes.csv,
  passage_dialogue.csv (dialogue share of every test passage), plots/trend_dialogue.png
(dialogue_cache.pkl keeps the per-book dialogue counts; delete it if the corpus changes.)
"""
import argparse
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from windowing import parse_sentences, WORD_RE
from front_matter import load_trimmed
from dialogue_share_check import detect_style, dialogue_words

COND_LABELS = {'all': 'whole-book training', 'all_noreg': 'whole books, register words removed',
               'noopen': 'whole books, first 3,000 words dropped', 'matched': 'same-position training (equal amount)',
               'unmatched': 'random later training (equal amount)'}
INK, MUTED, GRID = '#333333', '#8a8a8a', '#e6e6e6'
BLUE, ORANGE = '#2a78d6', '#eb6834'


# --------------------------------------------------------- dialogue per book ----
def _measure(args):
    path, fn = args
    text, _, _, _ = load_trimmed(path, fn)
    sents = parse_sentences(text)
    style, _, _ = detect_style(text)
    inw = dialogue_words(sents, style)
    nw = np.array([s.n_words for s in sents], dtype=float)
    n_tok = len(WORD_RE.findall(text.lower()))
    return fn, dict(style=style, cumw=np.cumsum(nw), cumd=np.cumsum(np.array(inw, dtype=float)),
                    n_sent_words=float(nw.sum()), n_tok=n_tok)


def load_dialogue(corpus_dir, books, cache):
    sig = tuple(sorted(books))
    if os.path.exists(cache):
        with open(cache, 'rb') as f:
            d = pickle.load(f)
        if d.get('sig') == sig:
            print(f'Loading cached dialogue counts from {cache}', flush=True)
            return d['data']
    print(f'Measuring dialogue in {len(books)} books (one-time; cached afterwards)...', flush=True)
    t0 = time.time()
    data = {}
    with ProcessPoolExecutor() as ex:
        for i, (fn, d) in enumerate(ex.map(_measure, [(os.path.join(corpus_dir, b), b) for b in books]), 1):
            data[fn] = d
            if i % 20 == 0 or i == len(books):
                print(f'  [{i}/{len(books)}] {time.time()-t0:.0f}s', flush=True)
    with open(cache, 'wb') as f:
        pickle.dump({'sig': sig, 'data': data}, f)
    return data


def passage_share(d, pos, width):
    if d['style'] == 'none':
        return np.nan
    xp = np.concatenate([[0.0], d['cumw']])
    fp = np.concatenate([[0.0], d['cumd']])
    lo, hi = float(pos), float(min(pos + width, xp[-1]))
    if hi - lo < width * 0.5:
        return np.nan
    return (np.interp(hi, xp, fp) - np.interp(lo, xp, fp)) / (hi - lo)


# ------------------------------------------------------------- regression ----
def _demean(x):
    return x - x.mean(axis=0, keepdims=True)


def prep(df, y_col, scale):
    """Per-author Gram matrices for models A, B and the dialogue-vs-position regression."""
    out = []
    for a, g in df.groupby('author'):
        if len(g) < 6:
            continue
        P = g['pos'].values / 10000.0
        S = g['share'].values
        Y = g[y_col].values * scale
        Pd, Sd, S2d, Yd = _demean(P[:, None]), _demean(S[:, None]), _demean((S ** 2)[:, None]), _demean(Y[:, None])
        XA = Pd
        XB = np.hstack([Pd, Sd, S2d])
        out.append(dict(GA=XA.T @ XA, gA=(XA.T @ Yd)[:, 0], GB=XB.T @ XB, gB=(XB.T @ Yd)[:, 0],
                        GS=Pd.T @ Pd, gS=(Pd.T @ _demean(S[:, None]))[:, 0], mean_share=S.mean(), n=len(g)))
    return out


def estimate(stack, counts):
    w = counts.astype(float)
    GA = sum(wi * s['GA'] for wi, s in zip(w, stack)); gA = sum(wi * s['gA'] for wi, s in zip(w, stack))
    GB = sum(wi * s['GB'] for wi, s in zip(w, stack)); gB = sum(wi * s['gB'] for wi, s in zip(w, stack))
    GS = sum(wi * s['GS'] for wi, s in zip(w, stack)); gS = sum(wi * s['gS'] for wi, s in zip(w, stack))
    ms = sum(wi * s['mean_share'] * s['n'] for wi, s in zip(w, stack)) / sum(wi * s['n'] for wi, s in zip(w, stack))
    bA = np.linalg.solve(GA, gA)[0]
    bB = np.linalg.solve(GB, gB)
    bS = np.linalg.solve(GS, gS)[0] * 100          # dialogue share slope, points of share per 10k words
    eff = (bB[1] + 2 * bB[2] * ms) * 0.10          # accuracy change for +10 points of dialogue share
    return dict(slope_A=bA, slope_B=bB[0], change=bB[0] - bA, dialogue_effect=eff, dialogue_slope=bS)


def run_model(df, y_col, scale, n_boot, rng):
    stack = prep(df, y_col, scale)
    A = len(stack)
    point = estimate(stack, np.ones(A))
    boots = {k: [] for k in point}
    for _ in range(n_boot):
        c = np.bincount(rng.integers(0, A, size=A), minlength=A)
        e = estimate(stack, c)
        for k in point:
            boots[k].append(e[k])
    res = {'authors': A, 'passages': int(sum(s['n'] for s in stack))}
    for k in point:
        b = np.array(boots[k])
        res[k] = point[k]
        res[k + '_lo'], res[k + '_hi'] = float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
    res['retained'] = point['slope_B'] / point['slope_A'] if abs(point['slope_A']) > 1e-9 else np.nan
    return res


def f1(v):
    return f'{v:+.1f}'


def ci(r, k):
    return f'{f1(r[k])} ({f1(r[k + "_lo"])} to {f1(r[k + "_hi"])})'


# ------------------------------------------------------------------ plot ----
def plot(ps, res, args, path):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    m = ps.groupby('pos')['share'].mean()
    ax.plot(m.index / 1000, m.values * 100, color=ORANGE, lw=2)
    ax.axvline(args.start / 1000, color=MUTED, lw=1, ls=':')
    ax.set_title('Dialogue share of the test passages', loc='left', fontsize=10.5, color=INK)
    ax.set_xlabel('start of the 2,000-word passage (thousand words)', fontsize=9.5, color=INK)
    ax.set_ylabel('% of words inside quotation marks', fontsize=9.5, color=INK)
    ax.grid(axis='y', color=GRID, lw=.8)
    ax = axes[1]
    conds = [c for c in COND_LABELS if ('delta', 'accuracy', c) in res]
    y = np.arange(len(conds))[::-1]
    for off, key, col, lab in ((0.14, 'slope_A', BLUE, 'not adjusted'), (-0.14, 'slope_B', ORANGE, 'dialogue held fixed')):
        for yi, c in zip(y, conds):
            r = res[('delta', 'accuracy', c)]
            ax.plot([r[key + '_lo'], r[key + '_hi']], [yi + off] * 2, color=col, lw=2)
            ax.plot(r[key], yi + off, 'o', color=col, ms=6, label=lab if yi == y[0] else None)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([COND_LABELS[c] for c in conds], fontsize=8.5)
    ax.set_xlabel('accuracy slope, points per 10,000 words (95% interval)', fontsize=9.5, color=INK)
    ax.set_title(f'Slope from {args.start/1000:.0f}k words on', loc='left', fontsize=10.5, color=INK)
    ax.grid(axis='x', color=GRID, lw=.8)
    ax.legend(frameon=False, fontsize=8.5, loc='lower right')
    for a in axes:
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('Run (from')[0], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--corpus-dir', default='real_corpus')
    ap.add_argument('--trials', default=os.path.join('output_attrib', 'trials.csv'))
    ap.add_argument('--out', default='output_attrib_trend_dialogue')
    ap.add_argument('--start', type=int, default=4000)
    ap.add_argument('--passage', type=int, default=2000)
    ap.add_argument('--boot', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=20261010)
    args = ap.parse_args()
    if not os.path.exists(args.trials):
        raise SystemExit(f'{args.trials} not found. Run attribution_by_position.py first.')
    os.makedirs(os.path.join(args.out, 'plots'), exist_ok=True)
    trials = pd.read_csv(args.trials)
    books = sorted(trials['book'].unique())
    dd = load_dialogue(args.corpus_dir, books, os.path.join(args.out, 'dialogue_cache.pkl'))

    bad = [b for b in books if abs(dd[b]['n_sent_words'] - dd[b]['n_tok']) > 0.02 * dd[b]['n_tok']]
    if bad:
        print(f'WARNING: word counts of the sentence parser and the word counter differ by >2% in {len(bad)} books '
              f'(positions may be slightly off): {", ".join(bad[:5])}', flush=True)
    no_dlg = [b for b in books if dd[b]['style'] == 'none']
    pk = trials[['author', 'book', 'pos']].drop_duplicates()
    pk['share'] = [passage_share(dd[b], p, args.passage) for b, p in zip(pk['book'], pk['pos'])]
    pk.to_csv(os.path.join(args.out, 'passage_dialogue.csv'), index=False)
    t = trials.merge(pk, on=['author', 'book', 'pos'])
    t = t[t['pos'] >= args.start].dropna(subset=['share'])
    if t.empty:
        raise SystemExit('No test passages with detectable dialogue.')
    ps = pk.dropna(subset=['share'])
    rng = np.random.default_rng(args.seed)

    res = {}
    rows = []
    for method in ('delta', 'cosine'):
        for ycol, mname, scale in (('correct', 'accuracy', 100.0), ('rr', 'reciprocal rank', 100.0)):
            for c in COND_LABELS:
                d = t[(t['cond'] == c) & (t['method'] == method)]
                if d.empty:
                    continue
                r = run_model(d, ycol, scale, args.boot, rng)
                res[(method, mname, c)] = r
                rows.append(dict(method=method, measure=mname, cond=c, **r))
    pd.DataFrame(rows).to_csv(os.path.join(args.out, 'trend_dialogue_slopes.csv'), index=False)
    plot(ps, res, args, os.path.join(args.out, 'plots', 'trend_dialogue.png'))

    # ---- report ----
    L = ['# Attribution trend, controlling for dialogue', '',
         f'Run {time.strftime("%Y-%m-%d")}. {t["author"].nunique()} authors; passages from {args.start:,} words on; each author has a separate baseline; '
         f'authors resampled {args.boot:,} times. Slopes: percentage points per 10,000 words of position.']
    if no_dlg:
        L += [f'Books with no detectable dialogue, left out here: {", ".join(no_dlg)}.']
    r0 = res[('delta', 'accuracy', 'all')]
    mm = ps.groupby('pos')['share'].mean()
    L += ['', '## 1. Does dialogue rise after the opening?', '',
          'Mean dialogue share of the test passages: ' + '; '.join(f'{int(p/1000)}k: {v*100:.0f}%' for p, v in mm.items() if p in (0, 2000, 4000, 8000, 12000, 20000, 28000)) + '.',
          f'From {args.start:,} words on, dialogue share changes by {f1(r0["dialogue_slope"])} points per 10,000 words '
          f'(95% interval {f1(r0["dialogue_slope_lo"])} to {f1(r0["dialogue_slope_hi"])}).', '']
    for method, mname in (('delta', "Burrows' Delta (primary)"), ('cosine', 'Cosine Delta (check)')):
        for meas in ('accuracy', 'reciprocal rank'):
            L += [f'## 2. {mname}: {meas} slope, not adjusted vs dialogue held fixed', '',
                  '| training | not adjusted | dialogue held fixed | share of the slope kept | dialogue effect (+10 points of dialogue) |',
                  '|---|---|---|---:|---|']
            for c in COND_LABELS:
                r = res.get((method, meas, c))
                if r is None:
                    continue
                ret = f'{r["retained"]*100:.0f}%' if np.isfinite(r['retained']) else '-'
                L.append(f'| {COND_LABELS[c]} | {ci(r, "slope_A")} | {ci(r, "slope_B")} | {ret} | {ci(r, "dialogue_effect")} |')
            L.append('')
    # plain reading
    L += ['## Plain reading', '']
    for c in ('all', 'matched'):
        r = res.get(('delta', 'accuracy', c))
        if r is None:
            continue
        if r['slope_B_lo'] > 0:
            state = 'survives: still clearly positive with dialogue held fixed'
        elif r['slope_B_hi'] < 0:
            state = 'reverses: negative with dialogue held fixed'
        elif np.isfinite(r['retained']) and r['retained'] >= 0.7 and r['slope_A'] > 0:
            state = 'mostly survives (at least 70% of the slope remains) but its interval now includes 0'
        elif np.isfinite(r['retained']) and r['retained'] < 0.5 and r['slope_A'] > 0:
            state = 'is mostly explained by dialogue (under half of the slope remains)'
        else:
            state = 'is partly reduced; the interval includes 0'
        L.append(f'* **{COND_LABELS[c]}:** the slope goes from {f1(r["slope_A"])} to {f1(r["slope_B"])} points per 10,000 words '
                 f'(interval {f1(r["slope_B_lo"])} to {f1(r["slope_B_hi"])}); the rise {state}.')
    L += ['', 'Limits: dialogue share is one aspect of scene mix (action, description and reflection also shift with position), '
              'and it is adjusted for with a simple curve (linear plus squared). A slope that survives is evidence the rise is not just '
              'more quoted speech, not proof of voice. The two intervals (not adjusted, adjusted) come from the same resamples of authors, '
              'so compare the point estimates and the share kept more than the overlap of the intervals.']
    with open(os.path.join(args.out, 'trend_dialogue_report.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')

    print(f'Dialogue-controlled trend ({t["author"].nunique()} authors), Delta accuracy, points per 10,000 words from {args.start:,}:')
    for c in COND_LABELS:
        r = res.get(('delta', 'accuracy', c))
        if r is not None:
            print(f'  {COND_LABELS[c]:42} not adjusted {ci(r, "slope_A"):24}  dialogue fixed {ci(r, "slope_B"):24}')
    print(f'\nReport: {os.path.join(args.out, "trend_dialogue_report.md")}')


if __name__ == '__main__':
    main()
