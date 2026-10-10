"""
Verification plan A3: SHUFFLED-AUTHOR CONTROL.

The worry: novels might simply drift as they go (longer sentences, more
dialogue, ...). Every author's curve would then rise in some features, and
"voice begins to build at ~6,000 words" would be general narrative drift, not
anything about individual voice.

The control: take the 100 books and deal them into random "authors" with the
same group sizes as the real ones (33 groups of three, and the single Anon
book as its own group). A random group has no shared voice. Then run the
whole pipeline on the fake authors exactly as for real ones:

  reference profile = the group's earliest book; test curves = the other two;
  background = every other group's profile; accumulator test (chunk-shuffle
  null, BH-FDR across all tests); the 'accumulators' pooled per group with
  the corpus opening register subtracted; hinge onset on the corpus average.

Whatever structure survives that shuffle is what general drift produces. It
is compared with the real labels on three things:

  1. how many accumulators there are (real: step 2's count)
  2. WHERE the pooled curve starts to rise (the onset)
  3. HOW MUCH it rises (late minus early, in SD units)

Reading it:
  * Shuffled labels reproduce the onset (often within +/-1,500 words of the
    real one) -> the timing is not specific to voice; drift produces it.
    Voice might still add to the amount (look at 1 and 3).
  * Shuffled labels do not reproduce it, and have far fewer accumulators /
    a much smaller rise -> what we measured is author-specific.
  * Either way, report 1-3 together. A control with a similar onset but a
    far smaller rise says "timing is shared, size is voice".

Each replicate redraws the groups, rebuilds the curves, and reruns the
tests. Permutations default to 5,000 per test. The smallest possible p is
1/(permutations+1), and Benjamini-Hochberg across ~800 tests needs at least
~800 x p_min / 0.05 tests sitting at that floor before any can pass (4 at
5,000 permutations, 16 at 1,000, 32 at 500). So very low settings can
undercount accumulators in shuffles where few tests are significant, which
would flatter the real labels; use 5,000 or more for anything you report.
Results are appended to
output_verify/shuffle_results.csv after every replicate, so an interrupted
run resumes where it stopped.

Run (from the voice_pipeline folder, after steps 1-3):
    python onset_shuffled_control.py --reps 3 --n-perm 1000   # quick look: only to see that it runs
    python onset_shuffled_control.py                          # 30 replicates
    python onset_shuffled_control.py --reps 100 --with-cut    # also count survivors of the 3,000-word cut
    python onset_shuffled_control.py --summary-only           # just redo the report from saved replicates

Outputs (output_verify/): shuffle_results.csv, shuffle_report.md,
  plots/shuffle_control.png
"""
import argparse
import contextlib
import io
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from features import FEATURE_NAMES
from verify_common import (
    OUT, COMPARISON_CSV, Cfg, load_features, test_curves, acc_dict, acc_from_comp, n_pairs,
    opening_residuals, estimate, onset_words, check_baseline, read_real_onsets, style_axes,
    INK, MUTED, GRID, C_COR, C_BAND)
from accumulator_retest import build_curves
from run_full_corpus import discover_authors

RESULTS_CSV = os.path.join(OUT, 'shuffle_results.csv')
NEAR = 1500   # "reproduces the onset" = within this many words of the real corrected onset


def pseudo_authors(by_author, rng):
    """Deal all books into random groups with the real group sizes. Within a
    group the books are ordered by year, as for real authors (earliest =
    reference book)."""
    sizes = [len(b) for b in by_author.values()]
    books = [bk for bs in by_author.values() for bk in bs]       # (year, filename)
    perm = rng.permutation(len(books))
    out, i = {}, 0
    for k, s in enumerate(sizes):
        out[f'P{k:02d}'] = sorted(books[j] for j in perm[i:i + s])
        i += s
    return out


def replicate(i, by_author, feats, cfg, n_perm, with_cut, seed0):
    t0 = time.time()
    rng = np.random.default_rng(seed0 + i)
    pseudo = pseudo_authors(by_author, rng)
    with contextlib.redirect_stdout(io.StringIO()):          # build_curves prints one line per author
        curves, habit = build_curves(pseudo, feats)
    resid = opening_residuals(pseudo, feats, cfg, 'all')
    df = test_curves(curves, cfg, n_perm, verbose=False)
    acc = acc_dict(df)
    res = estimate(curves, habit, acc, cfg, mode='corpus_loo', resid_all=resid, rng=rng, n_boot=0)
    row = {'rep': i, 'seed': seed0 + i, 'n_accumulators': n_pairs(acc),
           'n_groups_with_acc': len(res['authors']),
           'onset_cor_words': onset_words(res['cor'], cfg)[0],
           'onset_raw_words': onset_words(res['raw'], cfg)[0],
           'rise_cor_sd': res['cor']['rise'], 'rise_raw_sd': res['raw']['rise'],
           'slope_cor': res['cor']['slope'], 'seconds': np.nan}
    if with_cut:
        dfc = test_curves(curves, cfg, n_perm, sel=lambda n: (cfg.cut3000, n), verbose=False)
        both = df['sig'].values & dfc['sig'].values
        row['n_acc_after_cut'] = int(dfc['sig'].sum())
        row['n_survivors'] = int(both.sum())
    row['seconds'] = time.time() - t0
    return row


def real_reference(by_author, feats, cfg, curves, habit, comp, rng):
    """The same quantities for the real author labels (also re-checks the baseline)."""
    resid = opening_residuals(by_author, feats, cfg, 'all')
    acc = acc_from_comp(comp, comp['sig_trimmed_full'])
    res = estimate(curves, habit, acc, cfg, mode='corpus_loo', resid_all=resid, rng=rng, n_boot=2000)
    check_baseline(res, cfg)
    on, lo, hi = onset_words(res['cor'], cfg)
    ref = {'n_accumulators': n_pairs(acc), 'n_groups_with_acc': len(res['authors']),
           'onset_cor_words': on, 'ci_lo': lo, 'ci_hi': hi,
           'onset_raw_words': onset_words(res['raw'], cfg)[0],
           'rise_cor_sd': res['cor']['rise'], 'rise_raw_sd': res['raw']['rise'],
           'slope_cor': res['cor']['slope']}
    if 'sig_cut_3000' in comp:
        ref['n_acc_after_cut'] = int(comp['sig_cut_3000'].sum())
        ref['n_survivors'] = int((comp['sig_trimmed_full'].astype(bool) & comp['sig_cut_3000'].astype(bool)).sum())
    return ref


# --------------------------------------------------------------- report ----
def summarise(sh, ref):
    rows = []
    metrics = [('n_accumulators', 'accumulators (author x feature pairs)', '.0f'),
               ('n_groups_with_acc', 'authors / groups with an accumulator', '.0f'),
               ('onset_cor_words', 'onset, register subtracted (words)', ',.0f'),
               ('onset_raw_words', 'onset, raw curve (words)', ',.0f'),
               ('rise_cor_sd', 'rise, register subtracted (SD units)', '.2f'),
               ('rise_raw_sd', 'rise, raw curve (SD units)', '.2f')]
    if 'n_survivors' in sh:
        metrics += [('n_survivors', 'survivors of the 3,000-word cut', '.0f')]
    for key, label, fm in metrics:
        v = sh[key].dropna().values
        if len(v) == 0 or key not in ref:
            continue
        rows.append({'metric': label, 'real': format(ref[key], fm),
                     'shuffled mean': format(v.mean(), fm),
                     'shuffled 2.5%-97.5%': f'{format(np.percentile(v, 2.5), fm)} to {format(np.percentile(v, 97.5), fm)}',
                     'share of shuffles >= real': f'{np.mean(v >= ref[key]):.0%}'})
    return pd.DataFrame(rows)


def write_report(sh, ref, path):
    n = len(sh)
    L = ['# Shuffled-author control (verification plan A3)', '',
         f'Run {time.strftime("%Y-%m-%d")}. {n} replicates: the 100 books dealt into random groups with the real '
         'group sizes, then the full pipeline (reference profile, curves, accumulator test with FDR, pooled '
         'curve with the opening register subtracted, hinge onset) run on the fake authors. Real labels: '
         f'onset {ref["onset_cor_words"]:,.0f} words (95% {ref["ci_lo"]:,.0f}-{ref["ci_hi"]:,.0f}), '
         f'{ref["n_accumulators"]} accumulators.', '']
    S = summarise(sh, ref)
    L += ['| metric | real labels | shuffled: mean | shuffled: 2.5%-97.5% | share of shuffles >= real |',
          '|---|---:|---:|---:|---:|']
    for _, r in S.iterrows():
        L.append(f"| {r['metric']} | {r['real']} | {r['shuffled mean']} | {r['shuffled 2.5%-97.5%']} | "
                 f"{r['share of shuffles >= real']} |")
    on = sh['onset_cor_words'].dropna().values
    near = float(np.mean(np.abs(on - ref['onset_cor_words']) <= NEAR)) if len(on) else np.nan
    later = float(np.mean(on >= ref['onset_cor_words'])) if len(on) else np.nan
    p_acc = float((1 + np.sum(sh['n_accumulators'].values >= ref['n_accumulators'])) / (n + 1))
    p_rise = float((1 + np.sum(sh['rise_cor_sd'].dropna().values >= ref['rise_cor_sd'])) /
                   (1 + len(sh['rise_cor_sd'].dropna())))
    L += ['', f'Shuffled onsets within {NEAR:,} words of the real onset: **{near:.0%}** of replicates '
          f'(median shuffled onset {np.median(on):,.0f}; {later:.0%} of shuffles start later than the real one).',
          f'Empirical p (shuffles at least as extreme as real, +1 correction): accumulators {p_acc:.3f}, '
          f'rise {p_rise:.3f}.', '']
    if near >= 0.25:
        timing = ('**Shuffled labels often reproduce the onset timing**, so a ~6,000-word onset is not '
                  'evidence of voice by itself: general drift in the pooled curve starts about as late.')
    elif near <= 0.05:
        timing = ('**Shuffled labels rarely reproduce the onset timing**, so the timing is not simply what '
                  'general drift produces.')
    else:
        timing = ('Shuffled labels **sometimes** reproduce the onset timing; the timing is only weakly '
                  'informative about voice.')
    if p_acc < 0.05 and p_rise < 0.05:
        amount = ('The real labels show **more accumulators and a larger rise** than shuffled labels: the '
                  'amount of accumulation is author-specific.')
    elif p_acc < 0.05 or p_rise < 0.05:
        amount = ('The real labels exceed the shuffles on **one of** accumulator count and rise: partly '
                  'author-specific, partly drift.')
    else:
        amount = ('The real labels do **not** clearly exceed the shuffles on accumulator count or rise: '
                  'the accumulation is not distinguishable from general drift.')
    L += [timing, '', amount, '',
          'Notes: the shuffles use the same register correction (leave-one-out corpus opening profile) and the '
          'same 5-window chunk null as the real run. Shuffled groups mix eras, which real authors do not; if the '
          'results are borderline, that is the first thing to vary. Permutations per test: '
          f'{int(sh.attrs.get("n_perm", 0)) or "see run settings"}.', '']
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L))


def plot(sh, ref, path):
    panels = [('n_accumulators', 'accumulators', ref['n_accumulators'], None),
              ('onset_cor_words', 'onset (thousand words)', ref['onset_cor_words'] / 1000, 1000.0),
              ('rise_cor_sd', 'rise (SD units)', ref['rise_cor_sd'], None)]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for ax, (col, lab, real, div) in zip(axes, panels):
        v = sh[col].dropna().values
        v = v / div if div else v
        ax.hist(v, bins=15, color=C_COR, alpha=0.75, edgecolor='white', linewidth=0.6)
        ax.axvline(real, color=C_BAND, lw=2, label='real labels')
        ax.set_xlabel(lab, fontsize=8, color=INK)
        ax.set_ylabel('shuffled replicates', fontsize=8, color=INK)
        style_axes(ax)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle('Real author labels vs. shuffled labels', x=0.01, ha='left', fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('Run (from')[0], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--reps', type=int, default=30, help='number of shuffled replicates (default 30)')
    ap.add_argument('--n-perm', type=int, default=5000, help='permutations per accumulator test (default 5000)')
    ap.add_argument('--with-cut', action='store_true', help='also count survivors of the 3,000-word cut')
    ap.add_argument('--seed', type=int, default=20261010)
    ap.add_argument('--summary-only', action='store_true')
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(os.path.join(OUT, 'plots'), exist_ok=True)
    if not os.path.exists(COMPARISON_CSV):
        raise SystemExit(f'{COMPARISON_CSV} not found -- run accumulator_retest.py first.')
    cfg = Cfg(300)
    by_author = discover_authors()
    feats = load_features(by_author, 300)
    comp = pd.read_csv(COMPARISON_CSV)

    print('Real labels (reference for the comparison)...', flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        curves, habit = build_curves(by_author, feats)
    ref = real_reference(by_author, feats, cfg, curves, habit, comp, np.random.default_rng(args.seed))
    print(f'  real: {ref["n_accumulators"]} accumulators, onset {ref["onset_cor_words"]:,.0f} words', flush=True)

    done = pd.read_csv(RESULTS_CSV) if os.path.exists(RESULTS_CSV) else pd.DataFrame()
    have = set(done['rep']) if len(done) else set()
    if not args.summary_only:
        for i in range(args.reps):
            if i in have:
                continue
            row = replicate(i, by_author, feats, cfg, args.n_perm, args.with_cut, args.seed)
            row['n_perm'] = args.n_perm
            done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
            done.to_csv(RESULTS_CSV, index=False)
            print(f'  replicate {i + 1}/{args.reps}: {row["n_accumulators"]} accumulators, onset '
                  f'{row["onset_cor_words"]:,.0f} words, rise {row["rise_cor_sd"]:.2f} SD '
                  f'({row["seconds"]:.0f}s; total {time.time()-t0:.0f}s)', flush=True)
    if len(done) == 0:
        raise SystemExit('No replicates saved yet.')
    done.attrs['n_perm'] = int(done['n_perm'].iloc[-1]) if 'n_perm' in done else 0
    write_report(done, ref, os.path.join(OUT, 'shuffle_report.md'))
    plot(done, ref, os.path.join(OUT, 'plots', 'shuffle_control.png'))
    print('\n' + summarise(done, ref).to_string(index=False))
    print(f'\nOutputs written to {OUT}/ (report: shuffle_report.md) -- total {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
