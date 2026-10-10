"""
Verification plan A1 + A2: IS THE ~5,700-WORD VOICE ONSET ROBUST?

Step 3 estimated, from pooled accumulator curves with the opening register
subtracted, that voice starts building ~5,700 words in (95%: 900-12,900).
Three things kept that from being a solid claim: (a) it depends on the
register correction, (b) the interval is wide, (c) it is circular -- the
features were picked because they rise over the WHOLE book, and the onset was
then measured on those same curves. This script attacks (a) and (c).

PART 1 (A1)  Remove the circularity
-----------------------------------
 split-half   Pick each author's accumulators using ONLY the second half of
              the book (accumulator test on windows n/2..n), then estimate the
              onset on the FIRST half only. The features are chosen on text the
              onset never sees. A paired row shows the same first-half
              estimate with the original whole-book selection, so you can see
              how much the circularity mattered.
              Caveat: a feature that rises early and then flattens will not be
              picked on the second half, so this version can lean late.
 split-book   Each author has two test books. Pick accumulators on one book's
              curve, estimate the onset on the OTHER book's curve (both
              directions, then pooled). Different text for selection and
              estimation, and no assumption about the shape of the rise.
              Cost: one book's curve is noisier than the two-book average, so
              fewer accumulators pass the test.

PART 2 (A2)  Change one thing at a time from the step-3 baseline
-----------------------------------------------------------------
 correction   none (raw) | the author's OWN reference-book opening shift |
              drop the first 3,000 words instead of subtracting (onset can't
              be earlier than 3,000 by construction)
 shape        step (flat then a jump) | quad (flat then a curved rise) | hinge
              (baseline). A step's position is where the level changes, which
              for a gradual rise is later than the hinge point; read each
              shape against its own interval.
 accumulators the 52 "survivors" (accumulators that stay significant after the
              first 3,000 words are cut) | everything significant after the cut
              | without exclaim/question/dash (likely scene structure) | only
              the features that are NOT opening-register features (so the
              register correction barely touches them)
 windows      500-word windows instead of 300 (all constants rescaled to the
              same number of words; accumulators re-selected on the 500-word
              curves)

The question for each row: does the onset stay roughly 3,000-8,000 words?
If it does across the board, the timing is not an artifact of one analysis
choice. The intervals will stay wide; that is the sample (23 authors), not a bug.

Run (from the voice_pipeline folder, after steps 1-3):
    python onset_robustness.py --quick        # 2,000 permutations, 500 bootstraps: a fast first pass
    python onset_robustness.py                # full: 20,000 permutations, 2,000 bootstraps
    python onset_robustness.py --part split   # or --part variants
    python onset_robustness.py --fracs 0.4 0.5 0.6   # more split-half cut points
Saved test tables are reused when the settings match (delete output_verify/
tests_*.csv to force a rerun).

Outputs (output_verify/): split_results.csv, variants_results.csv,
  robustness_report.md, plots/split_forest.png, plots/variants_forest.png
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from features import FEATURE_NAMES
from verify_common import (
    OUT, COMPARISON_CSV, DIALOGUE_FEATURES, Cfg, load_features, test_curves, acc_dict, acc_from_comp,
    n_pairs, build_curves_by_book, opening_residuals, estimate, fit_grand, onset_words, check_baseline,
    register_band, register_directions, forest_plot, md_table, read_real_onsets)
from accumulator_retest import build_curves
from run_full_corpus import discover_authors

ACCEPT = (3000, 8000)   # the plan's "stays in this range = good" criterion (words)


# ----------------------------------------------------------------- rows ----
def make_row(group, name, res, cfg, acc, cut=0, note=''):
    on, lo, hi = onset_words(res['cor'], cfg, cut)
    onr = onset_words(res['raw'], cfg, cut)[0]
    at_floor = bool(cut > 0 and np.isfinite(res['cor']['T']) and res['cor']['T'] == 0)
    if at_floor:
        note = (note + '; ' if note else '') + 'at the floor: onset is at or before the cut'
    return {'group': group, 'variant': name,
            'onset_words': on, 'ci_lo': lo, 'ci_hi': hi, 'onset_raw_words': onr,
            'n_authors': res['cor']['n_authors'], 'n_features': res['n_features'],
            'n_accumulators': n_pairs(acc) if acc is not None else np.nan,
            'rise_sd': res['cor']['rise'],
            'in_3k_8k': bool(np.isfinite(on) and ACCEPT[0] <= on <= ACCEPT[1] and not at_floor),
            'note': note}


def combine_directions(res_list, cfg, rng, n_boot):
    """Pool two estimate() results that cover the same authors on different
    books: each author's curve is the average of its available curves."""
    by = {}
    for res in res_list:
        for a, r_raw, r_cor in zip(res['authors'], res['G_raw'], res['G_cor']):
            by.setdefault(a, []).append((r_raw, r_cor))
    authors = sorted(by)
    if not authors:
        empty = fit_grand(np.empty((0, 1)), cfg, 'hinge', rng, 0)
        return {'raw': empty, 'cor': empty, 'authors': [], 'n_features': 0}
    Gr = np.array([np.nanmean([x[0] for x in by[a]], axis=0) for a in authors])
    Gc = np.array([np.nanmean([x[1] for x in by[a]], axis=0) for a in authors])
    return {'raw': fit_grand(Gr, cfg, 'hinge', rng, n_boot),
            'cor': fit_grand(Gc, cfg, 'hinge', rng, n_boot),
            'authors': authors, 'n_features': sum(r['n_features'] for r in res_list)}


# ---------------------------------------------------------------- part 1 ----
def run_split(ctx):
    cfg, rng = ctx['cfg'], ctx['rng']
    common = dict(resid_all=ctx['resid_all'], resid_ref=ctx['resid_ref'], rng=rng, n_boot=ctx['n_boot'])
    rows = []

    for frac in ctx['fracs']:
        tag = f'{int(round(frac * 100))}/{100 - int(round(frac * 100))}'
        print(f'\nSplit-half ({tag}): select on the last {100 - int(round(frac * 100))}% of each book, '
              f'estimate on the first {int(round(frac * 100))}%...', flush=True)
        sel = lambda n, f=frac: (int(n * f), n)
        est = lambda n, f=frac: (0, int(n * f))
        df = test_curves(ctx['curves'], cfg, ctx['n_perm'], sel=sel,
                         label=f'splithalf_{int(round(frac * 100))}')
        acc = acc_dict(df)
        halves = [int(len(c[FEATURE_NAMES[0]]) * frac) for c in ctx['curves'].values()]
        L = min(cfg.grand_len, int(np.median(halves)))
        res_clean = estimate(ctx['curves'], ctx['habit'], acc, cfg, est=est, L=L, **common)
        res_circ = estimate(ctx['curves'], ctx['habit'], ctx['acc_full'], cfg, est=est, L=L, **common)
        rows.append(make_row('split-half', f'select on last {100 - int(round(frac * 100))}%, '
                             f'estimate on first {int(round(frac * 100))}% (clean)', res_clean, cfg, acc,
                             note=f'curve length {L} windows'))
        rows.append(make_row('split-half', f'whole-book selection, estimate on first '
                             f'{int(round(frac * 100))}% (circular, for comparison)', res_circ, cfg,
                             ctx['acc_full'], note=f'curve length {L} windows'))

    print('\nSplit-book: select on one test book, estimate on the other...', flush=True)
    pb = build_curves_by_book(ctx['by_author'], ctx['feats'])
    authors = [a for a, v in pb.items() if len(v) >= 2]
    clean_res, circ_res, accs = [], [], []
    for d in (0, 1):
        cur_sel = {a: pb[a][d] for a in authors}
        cur_est = {a: pb[a][1 - d] for a in authors}
        print(f'  selection on test book {d + 1}:', flush=True)
        df = test_curves(cur_sel, cfg, ctx['n_perm'], label=f'splitbook_sel{d + 1}')
        acc = acc_dict(df)
        accs.append(acc)
        r_clean = estimate(cur_est, ctx['habit'], acc, cfg, **common)
        r_circ = estimate(cur_est, ctx['habit'], ctx['acc_full'], cfg, **common)
        clean_res.append(r_clean)
        circ_res.append(r_circ)
        rows.append(make_row('split-book', f'select on test book {d + 1}, estimate on test book {2 - d} (clean)',
                             r_clean, cfg, acc))
        rows.append(make_row('split-book', f'whole-book selection, estimate on test book {2 - d} '
                             f'(circular, for comparison)', r_circ, cfg, ctx['acc_full']))
    both = combine_directions(clean_res, cfg, rng, ctx['n_boot'])
    both_c = combine_directions(circ_res, cfg, rng, ctx['n_boot'])
    n_both = n_pairs(accs[0]) + n_pairs(accs[1])
    rows.append(make_row('split-book', 'both directions pooled (clean)', both, cfg, None,
                         note=f'{n_both} accumulator picks across the two directions'))
    rows.append(make_row('split-book', 'both directions pooled (circular, for comparison)', both_c, cfg,
                         ctx['acc_full']))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- part 2 ----
def run_variants(ctx):
    cfg, rng = ctx['cfg'], ctx['rng']
    rows = []

    def add(name, note='', acc=None, c=None, h=None, cfg_=None, **kw):
        a = ctx['acc_full'] if acc is None else acc
        cf = cfg_ or cfg
        res = estimate(c or ctx['curves'], h or ctx['habit'], a, cf, rng=rng, n_boot=ctx['n_boot'], **kw)
        rows.append(make_row('variant', name, res, cf, a, cut=kw.get('cut', 0), note=note))
        o = onset_words(res['cor'], cf, kw.get('cut', 0))
        print(f'  {name:62s} {o[0]:8,.0f}  ({o[1]:,.0f}-{o[2]:,.0f})  [{res["cor"]["n_authors"]} authors]',
              flush=True)
        return res

    rr = dict(resid_all=ctx['resid_all'], resid_ref=ctx['resid_ref'])
    print('\nVariants (one change at a time from the step-3 baseline):', flush=True)
    base = add('baseline (step 3: corpus register subtracted, hinge, 300-word windows)', mode='corpus_loo', **rr)
    check_baseline(base, cfg)

    # --- register correction ---
    add('correction: none (raw curve)', mode='none', **rr)
    add("correction: author's own reference-book register", mode='own_ref', **rr)
    add('correction: drop first 3,000 words, no subtraction', mode='none', cut=cfg.cut3000,
        note='onset cannot be earlier than 3,000 by construction', **rr)

    # --- onset shape ---
    add('shape: quadratic (smooth start)', mode='corpus_loo', shape='quad', **rr)
    add('shape: step (jump to a new level)', mode='corpus_loo', shape='step',
        note="step position = where the level changes; later than a hinge's start for a gradual rise", **rr)

    # --- accumulator set ---
    comp = ctx['comp']
    acc_cut = acc_from_comp(comp, comp['sig_cut_3000'])
    acc_surv = acc_from_comp(comp, comp['sig_trimmed_full'].astype(bool) & comp['sig_cut_3000'].astype(bool))
    add(f'accumulators: survivors of the 3,000-word cut ({n_pairs(acc_surv)})', acc=acc_surv,
        mode='corpus_loo', note='selected on text from word 3,000 on, so may lean late', **rr)
    add(f'accumulators: everything significant after the cut ({n_pairs(acc_cut)})', acc=acc_cut,
        mode='corpus_loo', note='selected on text from word 3,000 on, so may lean late', **rr)
    add('accumulators: without exclaim/question/dash rates', mode='corpus_loo',
        drop_feats=DIALOGUE_FEATURES, **rr)
    reg = register_directions()
    if reg:
        non_reg = set(FEATURE_NAMES) - set(reg)
        add(f'accumulators: non-register features only ({len(non_reg)} features)', mode='corpus_loo',
            keep_feats=non_reg, note='the register correction barely touches these', **rr)
        add('accumulators: non-register features only, no correction', mode='none',
            keep_feats=non_reg, **rr)
    else:
        print('  (non-register-features variant skipped: output_register/register_tests.csv not found)')

    # --- 500-word windows ---
    cfg5 = Cfg(500)
    print(f'\n  500-word windows (constants rescaled: register {cfg5.conv_pos} windows, '
          f'curve {cfg5.grand_len}, chunk {cfg5.chunk}):', flush=True)
    feats5 = load_features(ctx['by_author'], 500)
    curves5, habit5 = build_curves(ctx['by_author'], feats5)
    df5 = test_curves(curves5, cfg5, ctx['n_perm'], label='w500_full')
    acc5 = acc_dict(df5)
    resid5 = opening_residuals(ctx['by_author'], feats5, cfg5, 'all')
    resid5r = opening_residuals(ctx['by_author'], feats5, cfg5, 'ref')
    add(f'windows: 500 words, accumulators re-selected ({n_pairs(acc5)})', acc=acc5, c=curves5, h=habit5,
        cfg_=cfg5, mode='corpus_loo', resid_all=resid5, resid_ref=resid5r,
        note=f'{cfg5.chunk}-window chunk null, constants rescaled to the same word counts')
    add('windows: 500 words, raw curve', acc=acc5, c=curves5, h=habit5, cfg_=cfg5, mode='none',
        resid_all=resid5, resid_ref=resid5r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------- report ----
def fmt_ci(r):
    if not np.isfinite(r['ci_lo']):
        return '-'
    return f"{r['ci_lo']:,.0f}-{r['ci_hi']:,.0f}"


def section(df, title, intro):
    L = [f'## {title}', '', intro, '']
    d = df.copy()
    d['ci'] = [fmt_ci(r) for _, r in d.iterrows()]
    d['in'] = np.where(d['in_3k_8k'], 'yes', 'no')
    L += md_table(d, [('variant', 'analysis'), ('onset_words', 'onset (words)'), ('ci', '95% interval'),
                      ('n_authors', 'authors'), ('n_accumulators', 'accumulator pairs'),
                      ('in', 'in 3,000-8,000?'), ('note', 'note')],
                  {'onset_words': ',.0f', 'n_authors': '.0f', 'n_accumulators': '.0f'})
    return L


def write_report(split, var, ctx, path):
    real = read_real_onsets()
    L = ['# Onset robustness (verification plan A1-A2)', '',
         f'Run {time.strftime("%Y-%m-%d")}. Permutations per accumulator test: {ctx["n_perm"]:,}; bootstrap '
         f'draws: {ctx["n_boot"]:,}; intervals come from resampling authors. Step 3 baseline: '
         + (f'{real["corrected"][0]:,.0f} words ({real["corrected"][1]:,.0f}-{real["corrected"][2]:,.0f}).'
            if real else 'not found.'), '']
    if split is not None:
        L += section(split, 'A1. Removing the circularity',
                     'Accumulators are chosen on text the onset never sees. "Circular" rows use the original '
                     'whole-book selection on the same estimation text, so the gap between a clean row and '
                     'its circular partner is the effect of the circularity. Split-half can lean late (features '
                     'that rise early then flatten are not picked); split-book has fewer accumulators.')
        L.append('')
    if var is not None:
        L += section(var, 'A2. One change at a time',
                     'Each row changes one thing from the baseline. The plan\'s criterion: an onset that stays '
                     'around 3,000-8,000 words in every row is not an artifact of one analysis choice. '
                     'Intervals stay wide with 23 authors; look at whether the point estimates move together '
                     'rather than at any single interval.')
        k = int(var['in_3k_8k'].sum())
        out = var[~var['in_3k_8k']]
        L += ['', f'**{k} of {len(var)} variants put the onset inside 3,000-8,000 words.**']
        if len(out):
            L.append('Outside the range: ' + '; '.join(f"{r['variant']} ({r['onset_words']:,.0f})"
                                                       for _, r in out.iterrows()) + '.')
        L.append('')
    L += ['## How to read this', '',
          '- A clean row near the baseline and near its circular partner: the selection circularity did not '
          'create the onset.',
          '- A clean row much earlier than the baseline (toward 0): the "later onset" came partly from choosing '
          'features that rise late, and voice may build from the start.',
          '- A clean row much later: either the onset really is later, or the selection lean-late caveat applies '
          '(split-half, cut-based rows).',
          '- Rows with very wide intervals say little either way; do not over-read their point estimates.', '']
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L))


# ----------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('Run (from')[0], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--part', choices=['all', 'split', 'variants'], default='all')
    ap.add_argument('--quick', action='store_true', help='2,000 permutations and 500 bootstrap draws')
    ap.add_argument('--n-perm', type=int, default=None)
    ap.add_argument('--n-boot', type=int, default=None)
    ap.add_argument('--fracs', type=float, nargs='+', default=[0.5],
                    help='split-half cut points: fraction of the book used for estimation (default 0.5)')
    ap.add_argument('--seed', type=int, default=20261010)
    args = ap.parse_args()
    n_perm = args.n_perm or (2000 if args.quick else 20000)
    n_boot = args.n_boot or (500 if args.quick else 2000)

    t0 = time.time()
    os.makedirs(os.path.join(OUT, 'plots'), exist_ok=True)
    if not os.path.exists(COMPARISON_CSV):
        raise SystemExit(f'{COMPARISON_CSV} not found -- run accumulator_retest.py first.')

    cfg = Cfg(300)
    by_author = discover_authors()
    feats = load_features(by_author, 300)
    curves, habit = build_curves(by_author, feats)
    print('Computing the opening-register profiles...', flush=True)
    comp = pd.read_csv(COMPARISON_CSV)
    ctx = {'cfg': cfg, 'by_author': by_author, 'feats': feats, 'curves': curves, 'habit': habit,
           'resid_all': opening_residuals(by_author, feats, cfg, 'all'),
           'resid_ref': opening_residuals(by_author, feats, cfg, 'ref'),
           'comp': comp, 'acc_full': acc_from_comp(comp, comp['sig_trimmed_full']),
           'n_perm': n_perm, 'n_boot': n_boot, 'fracs': args.fracs,
           'rng': np.random.default_rng(args.seed)}
    print(f'{n_pairs(ctx["acc_full"])} baseline accumulators (step 2, whole book, trimmed). '
          f'n_perm={n_perm:,}, n_boot={n_boot:,}.', flush=True)

    split = var = None
    band = register_band()
    if args.part in ('all', 'split'):
        split = run_split(ctx)
        split.to_csv(os.path.join(OUT, 'split_results.csv'), index=False)
        forest_plot([{'label': r['variant'], 'onset': r['onset_words'], 'lo': r['ci_lo'], 'hi': r['ci_hi']}
                     for _, r in split.iterrows()], band,
                    os.path.join(OUT, 'plots', 'split_forest.png'),
                    'Voice onset with accumulators chosen on separate text')
        print('\n' + split[['variant', 'onset_words', 'ci_lo', 'ci_hi', 'n_authors', 'n_accumulators']]
              .to_string(index=False, float_format=lambda v: f'{v:,.0f}'), flush=True)
    if args.part in ('all', 'variants'):
        var = run_variants(ctx)
        var.to_csv(os.path.join(OUT, 'variants_results.csv'), index=False)
        forest_plot([{'label': r['variant'], 'onset': r['onset_words'], 'lo': r['ci_lo'], 'hi': r['ci_hi']}
                     for _, r in var.iterrows()], band,
                    os.path.join(OUT, 'plots', 'variants_forest.png'),
                    'Voice onset under different analysis choices')
    write_report(split, var, ctx, os.path.join(OUT, 'robustness_report.md'))
    print(f'\nOutputs written to {OUT}/ (report: robustness_report.md) -- total {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
