"""
Step 3b: IS THERE A GAP between the end of the opening register and the
start of voice accumulation?

Step 1 put the end of the opening register at roughly 2,200-3,800 words
(middle half of 13 features); step 3 put the start of accumulation at
~5,700 words (95%: 900-12,900). The ranges overlap, so comparing them by
eye can't say whether there's a real gap. This tests the gap directly.

Method
------
1. ONE register end point. The 13 opening-register features (step 1) are
   combined into one "opening register" curve per author: each feature's
   step-1 z-profile (opening vs. rest of the same book), flipped so every
   feature points the same way (e.g. "fewer one-syllable words" counts as
   "more register"), then averaged. The step-1 fade-out line is fitted to
   the corpus average: the end point E is where it reaches zero.
2. ONE voice onset. Exactly step 3's corpus-level estimate: the authors'
   pooled accumulation curves (opening register removed) averaged over the
   first 30,000 words, with a flat-then-rising line; onset T.
3. PAIRED BOOTSTRAP. 2,000 times: draw 33 authors with replacement, and
   compute BOTH E and T from that same draw (T uses whichever drawn authors
   have accumulators). Gap = T - E for each draw. Using the same authors
   for both cancels some shared noise, so the gap is estimated more
   tightly than either number alone.

Reading it:
  95% interval of the gap entirely above 0 -> real gap: voice starts
     building some distance after the register ends.
  interval includes 0 -> no evidence of a gap: voice starts as the
     register ends, or the data can't tell them apart.
  entirely below 0 -> voice starts before the register has ended.

Run after opening_register_scan.py, accumulator_retest.py and
accumulation_onset.py:   python onset_gap_test.py
Outputs (output_onset/): gap_test.csv, gap_test_report.md, plots/gap_test.png
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
from run_full_corpus import discover_authors, WINDOW_SIZE
import opening_register_scan as step1
import accumulation_onset as step3
from accumulator_retest import load_features, build_curves

OUT = step3.OUT
N_BOOT = 2000
SEED = 20261004
E_GRID = np.arange(1.0, step1.N_POS - 1 + 1e-9, 0.25)   # register end, in windows
T_GRID = np.arange(0, step3.GRAND_MAX_T + 1, dtype=float)  # voice onset, in windows

INK, MUTED, GRID = '#333333', '#8a8a8a', '#e6e6e6'
C_BAR, C_LINE = '#2a78d6', '#eb6834'

warnings.filterwarnings('ignore', category=RuntimeWarning)


# ------------------------------------------------- register end point ----
def fade_design(n):
    p = np.arange(n, dtype=float)
    G = np.clip(1 - (p[None, :] + 0.5) / E_GRID[:, None], 0, None)
    Gc = G - G.mean(axis=1, keepdims=True)
    return Gc, (Gc ** 2).sum(axis=1)


def fit_fade(Y, design):
    """Y: (B x positions). Fade-out end point per row (positive offsets only),
    same model as opening_register_scan.end_point(shape='fade'). In windows."""
    Gc, Sxx = design
    Yc = Y - np.nanmean(Y, axis=1, keepdims=True)
    Sxy = Yc @ Gc.T
    gain = np.where(Sxy > 0, Sxy ** 2 / Sxx, -np.inf)
    k = np.argmax(gain, axis=1)
    ok = np.isfinite(gain[np.arange(len(Y)), k])
    return np.where(ok, E_GRID[k], np.nan)


def register_curves():
    """authors list, (33 x N_POS) combined register curve per author, feature list."""
    reg = pd.read_csv(os.path.join('output_register', 'register_tests.csv'))
    reg = reg[reg['q_1500'] < step1.ALPHA]
    feats = reg['feature'].tolist()
    signs = np.sign(reg['mean_z_1500'].values)
    mats, _ = step1.load_window_features()
    authors, A = step1.author_profiles(mats, trimmed=True)
    idx = [FEATURE_NAMES.index(f) for f in feats]
    R = np.nanmean(A[:, :, idx] * signs[None, None, :], axis=2)
    return authors, R, feats


# -------------------------------------------------------- voice curves ----
def voice_curves(authors):
    """(33 x GRAND_LEN) pooled corrected accumulation curve per author, NaN rows
    for authors without accumulators -- built exactly as in step 3."""
    by_author = discover_authors()
    feats = load_features(by_author)
    curves, habit = build_curves(by_author, feats)
    conv_res = step3.convention_residuals(by_author, feats)
    comp = pd.read_csv(step3.COMPARISON_CSV)
    acc = comp[comp[f'sig_{step3.ACCUM_SET}'].astype(bool)].groupby('author')['feature'].apply(list).to_dict()
    V = np.full((len(authors), step3.GRAND_LEN), np.nan)
    for i, a in enumerate(authors):
        if a not in acc:
            continue
        conv_a = np.mean([conv_res[b] for b in conv_res if b != a], axis=0)
        _, cor, used = step3.pooled_curves(curves[a], acc[a], habit[a], conv_a)
        if cor is None:
            continue
        y = cor[:step3.GRAND_LEN]
        V[i, :len(y)] = y
    return V


# --------------------------------------------------------------- main ----
def main():
    t0 = time.time()
    os.makedirs(os.path.join(OUT, 'plots'), exist_ok=True)
    rng = np.random.default_rng(SEED)

    print('Building the combined opening-register curve per author (step 1 data)...', flush=True)
    authors, R, reg_feats = register_curves()
    print(f'  {len(reg_feats)} register features: {", ".join(reg_feats)}', flush=True)

    print('Building pooled voice curves per author (step 3 method)...', flush=True)
    V = voice_curves(authors)
    has_voice = ~np.all(np.isnan(V), axis=1)
    print(f'  {int(has_voice.sum())} of {len(authors)} authors have accumulators ({time.time()-t0:.0f}s)',
          flush=True)

    fdes = fade_design(R.shape[1])
    hdes = step3.hinge_design(step3.GRAND_LEN, T_GRID)
    E0 = fit_fade(np.nanmean(R, axis=0)[None, :], fdes)[0]
    T0 = step3.fit_hinge(np.nanmean(V[has_voice], axis=0)[None, :], T_GRID, hdes)[0][0]

    print(f'Paired bootstrap ({N_BOOT} draws of {len(authors)} authors)...', flush=True)
    idx = rng.integers(0, len(authors), (N_BOOT, len(authors)))
    Eb = fit_fade(np.nanmean(R[idx], axis=1), fdes)
    Vb = np.nanmean(V[idx], axis=1)                    # NaN rows (no accumulators) drop out
    ok = ~np.all(np.isnan(Vb), axis=1)
    Tb = np.full(N_BOOT, np.nan)
    Tb[ok] = step3.fit_hinge(Vb[ok], T_GRID, hdes)[0]
    gap = (Tb - Eb) * WINDOW_SIZE
    good = ~np.isnan(gap)

    w = WINDOW_SIZE
    e_lo, e_hi = np.nanpercentile(Eb, [2.5, 97.5]) * w
    t_lo, t_hi = np.nanpercentile(Tb, [2.5, 97.5]) * w
    g_lo, g_hi = np.nanpercentile(gap, [2.5, 97.5])
    share_pos = float(np.mean(gap[good] > 0))
    res = pd.DataFrame([
        {'quantity': 'register end (combined)', 'estimate_words': E0 * w, 'ci_lo': e_lo, 'ci_hi': e_hi},
        {'quantity': 'voice onset (corrected)', 'estimate_words': T0 * w, 'ci_lo': t_lo, 'ci_hi': t_hi},
        {'quantity': 'gap = onset - register end', 'estimate_words': (T0 - E0) * w, 'ci_lo': g_lo, 'ci_hi': g_hi},
    ])
    res['share_of_draws_gap_above_0'] = [np.nan, np.nan, share_pos]
    res['n_valid_draws'] = int(good.sum())
    res.to_csv(os.path.join(OUT, 'gap_test.csv'), index=False)

    if g_lo > 0:
        verdict = 'Real gap: voice starts building some distance AFTER the opening register ends.'
    elif g_hi < 0:
        verdict = 'Voice starts building BEFORE the opening register has ended.'
    else:
        verdict = ('No evidence of a gap: the interval includes 0, so voice may start right as the register '
                   'ends -- the data cannot separate the two.')

    # plot
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.hist(gap[good] / 1000, bins=40, color=C_BAR, edgecolor='white', linewidth=0.6)
    ax.axvline(0, color=INK, lw=1)
    ax.axvline((T0 - E0) * w / 1000, color=C_LINE, lw=2, label=f'estimate: {(T0 - E0) * w:,.0f} words')
    ax.axvspan(g_lo / 1000, g_hi / 1000, color=C_LINE, alpha=0.10, linewidth=0,
               label=f'95% interval: {g_lo:,.0f} to {g_hi:,.0f}')
    ax.set_xlabel('gap = voice onset minus end of opening register (thousand words)', fontsize=8, color=INK)
    ax.set_ylabel('bootstrap draws', fontsize=8, color=INK)
    ax.set_title('Is there a gap between the opening register and voice building?', loc='left',
                 fontsize=10, color=INK)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'plots', 'gap_test.png'), dpi=150)
    plt.close(fig)

    L = ['# Gap test: opening register end vs. voice onset (step 3b)', '',
         f'Run {time.strftime("%Y-%m-%d")}. Paired bootstrap: {N_BOOT} draws of {len(authors)} authors; '
         f'register end and voice onset computed from the same draw each time '
         f'({int(good.sum())} valid draws). Voice onset uses the {int(has_voice.sum())} authors with '
         'accumulators.', '',
         '| quantity | estimate | 95% interval |', '|---|---:|---:|']
    for _, r in res.iterrows():
        L.append(f"| {r['quantity']} | {r['estimate_words']:,.0f} words | {r['ci_lo']:,.0f} to {r['ci_hi']:,.0f} |")
    L += ['', f'Share of draws where voice starts after the register ends: {share_pos:.0%}.', '',
          f'**{verdict}**', '',
          f'Register features combined ({len(reg_feats)}): {", ".join(reg_feats)}.']
    with open(os.path.join(OUT, 'gap_test_report.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')

    print('\n' + res.to_string(index=False))
    print(f'\nShare of draws with gap > 0: {share_pos:.0%}')
    print(verdict)
    print(f'\nOutputs written to {OUT}/ (report: gap_test_report.md) -- total {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
