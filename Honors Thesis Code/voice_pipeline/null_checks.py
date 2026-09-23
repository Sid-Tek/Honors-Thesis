"""
Null-calibration checks referenced in output_sweep/burst_sweep_report.md.

1. Opening test (original single-window shuffle) on curves whose real
   position structure has been destroyed by shuffling 5-window chunks:
   how often is p < .05? (Should be ~5%.)
2. Placebo for the position-rank test: rank a random MID-book stretch
   instead of the opening. (Mean p should be ~0.5, p < .05 about 5%.)
3. Accumulator test re-run with the 5-window chunk-shuffle null, which
   respects the short-range autocorrelation between neighbouring windows.
   Compare the count of q < .05 accumulators with the original null.

Writes output_sweep/null_checks.txt and output_sweep/accumulator_null_comparison.csv
"""
import numpy as np
import pandas as pd

from build_curves import load_curves, OUT_DIR
from burst_sweep import perm_index, detrend, opening_perm_pvalues, CHUNK, TIE_TOL
from stats_tests import bh_fdr

N_PERM = 20000


def main():
    curves = load_curves()
    rng = np.random.default_rng(5)
    lines = []

    ks = (1, 2, 3, 5, 8, 13)
    flags = {k: [] for k in ks}
    placebo = []
    for key, y in curves.items():
        if np.isnan(y).any():
            continue
        n = len(y)
        pieces = [y[i:i + CHUNK] for i in range(0, n, CHUNK)]
        shuffled = np.concatenate([pieces[j] for j in rng.permutation(len(pieces))])
        res = opening_perm_pvalues(shuffled, ks, 2000)
        for k in ks:
            flags[k].append(res[k][1] < 0.05)
        d = detrend(y)
        c = np.concatenate([[0.0], np.cumsum(d)])
        means = (c[2:] - c[:-2]) / 2
        pos = rng.integers(int(0.2 * len(means)), int(0.8 * len(means)))
        placebo.append(np.mean(means >= means[pos] - TIE_TOL))
    lines.append('1. Opening test, single-window null, on position-free curves: % with p < .05 (target 5%)')
    lines += [f'   {k * 300:>5} words: {100 * np.mean(v):.1f}%' for k, v in flags.items()]
    placebo = np.array(placebo)
    lines.append(f'2. Rank-test placebo (mid-book stretch): mean p = {placebo.mean():.3f}, '
                 f'p < .05 in {100 * (placebo < 0.05).mean():.1f}% of curves (target 0.5 / 5%)')

    rows = []
    for key, y in curves.items():
        author, feat = key.split('|')
        n = len(y)
        if np.isnan(y).any() or n < 4:
            rows.append({'author': author, 'feature': feat, 'p_perm': np.nan, 'p_block': np.nan})
            continue
        x = np.arange(n) - (n - 1) / 2
        ssx = (x ** 2).sum()
        obs = (x * (y - y.mean())).sum() / ssx
        row = {'author': author, 'feature': feat, 'slope': obs}
        for kind in ('perm', 'block'):
            idx = perm_index(n, N_PERM, kind)
            ex = 0
            for s in range(0, N_PERM, 2000):
                b = y[idx[s:s + 2000]]
                ex += int(np.sum(((b - b.mean(1, keepdims=True)) * x).sum(1) / ssx >= obs - TIE_TOL))
            row[f'p_{kind}'] = (1 + ex) / (N_PERM + 1)
        rows.append(row)
    df = pd.DataFrame(rows)
    df['q_perm'] = bh_fdr(df['p_perm'].values)
    df['q_block'] = bh_fdr(df['p_block'].values)
    df.to_csv(f'{OUT_DIR}/accumulator_null_comparison.csv', index=False)
    lines.append(f'3. Accumulators at q < .05: single-window null {int((df.q_perm < .05).sum())}, '
                 f'5-window chunk null {int((df.q_block < .05).sum())}, '
                 f'significant under both {int(((df.q_perm < .05) & (df.q_block < .05)).sum())}')
    text = '\n'.join(lines)
    print(text)
    with open(f'{OUT_DIR}/null_checks.txt', 'w') as f:
        f.write(text + '\n')


if __name__ == '__main__':
    main()
