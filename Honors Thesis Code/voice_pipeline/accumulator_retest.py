"""
Step 2 of the convention -> voice experiment: RE-TEST THE ACCUMULATORS
WITHOUT THE OPENING.

The worry: step 1 (opening_register_scan.py) found that every novel opens
in a shared register -- longer sentences and words, more "the"/"of"/"in",
fewer "to"/"but" -- that fades within roughly the first 3,000 words. For
an author whose own habit runs AGAINST that register (e.g. a short-word
author), the opening makes them look less like themselves; as it fades,
their distinctiveness curve rises. That rise looks exactly like an
accumulator, but it is the convention wearing off, not voice building.

So: cut the opening off and test again.

Conditions (all on front-matter-trimmed text, same reference profiles,
same curves -- only the stretch of the curve being tested changes):
  trimmed_full   whole curve                    (effect of trimming alone)
  cut_3000       first 10 windows (3,000 words) dropped   <- the main test
  cut_6000       first 20 windows (6,000 words) dropped   (stricter cutoff)
  end_cut_3000   LAST 10 windows dropped        (control, see below)

Why the end-cut control: dropping 10 windows also means less text, and a
slope test on less text loses a little power. Some accumulators could drop
out for that reason alone. Dropping the same number of windows from the
END removes the same amount of text but none of the opening, so it shows
how many would be lost just from having less text. If cut_3000 loses about
as many as end_cut_3000, the opening wasn't creating accumulators.

Everything else matches run_full_corpus.py: earliest book = reference
profile, background = average of every other author's profile (Anon
included as background only), curves = test books averaged position by
position and truncated to the shorter book, accumulator_test with the
chunk-shuffle null (20,000 permutations), Benjamini-Hochberg FDR across
all 33 x 24 tests within each condition.

Mechanism check: for accumulators on the 13 opening-register features, the
convention should mostly create FAKE accumulators for authors whose habit
opposes the register's direction. So if the opening matters, the
accumulators lost in cut_3000 should be mostly "opposed" authors.

Run (after opening_register_scan.py, which supplies the register features):
    python accumulator_retest.py

Outputs (output_accum_retest/):
  accum_results_long.csv    every author x feature x condition: slope, p, q
  accum_comparison.csv      one row per author x feature: significant in each
                            condition, status vs. the original 75, habit vs.
                            register direction
  accum_retest_report.md    the summary
  plots/accum_by_feature.png
"""
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from windowing import parse_sentences, local_windows
from features import extract_features, FEATURE_NAMES
from scoring import corpus_stats, author_profile, distinctiveness_scores
from stats_tests import accumulator_test, bh_fdr
from front_matter import load_trimmed
from run_full_corpus import discover_authors, average_profiles, CORPUS_DIR, WINDOW_SIZE
from run_full_corpus import OUT_DIR as FULL_DIR

OUT = 'output_accum_retest'
CACHE = os.path.join(OUT, 'trimmed_window_features.pkl')
ORIGINAL_CSV = os.path.join(FULL_DIR, 'author_feature_results.csv')
REGISTER_CSV = os.path.join('output_register', 'register_tests.csv')
ALPHA = 0.05

# name -> (windows dropped from the start, windows dropped from the end)
CONDITIONS = {
    'trimmed_full': (0, 0),
    'cut_3000': (10, 0),
    'cut_6000': (20, 0),
    'end_cut_3000': (0, 10),
}
COND_LABEL = {
    'original': 'original run (Sep 23)',
    'trimmed_full': 'front matter trimmed, whole book',
    'cut_3000': 'first 3,000 words cut',
    'cut_6000': 'first 6,000 words cut',
    'end_cut_3000': 'control: last 3,000 words cut',
}

INK, MUTED, GRID = '#333333', '#8a8a8a', '#e6e6e6'
C_ORIG, C_3000, C_6000 = '#b5b5b5', '#2a78d6', '#1b4f8f'


# ------------------------------------------------------------- features ----
def _extract(fn):
    text, _, _, _ = load_trimmed(os.path.join(CORPUS_DIR, fn), fn)
    wins = local_windows(parse_sentences(text), WINDOW_SIZE)
    return fn, [extract_features(w) for w in wins]


def load_features(by_author):
    """Per-window feature dicts (with raw counts, for the dash floor) for
    every book, front matter trimmed. Cached after the first run."""
    if os.path.exists(CACHE):
        print(f'Loading cached window features from {CACHE}', flush=True)
        with open(CACHE, 'rb') as f:
            return pickle.load(f)
    files = [fn for books in by_author.values() for _, fn in books]
    print(f'Extracting window features for {len(files)} books (one-time; cached afterwards)...', flush=True)
    t0 = time.time()
    feats = {}
    with ProcessPoolExecutor() as ex:
        for i, (fn, fl) in enumerate(ex.map(_extract, files), 1):
            feats[fn] = fl
            print(f'  [{i}/{len(files)}] {fn} ({len(fl)} windows) {time.time()-t0:.0f}s', flush=True)
    with open(CACHE, 'wb') as f:
        pickle.dump(feats, f)
    return feats


# --------------------------------------------------------------- curves ----
def build_curves(by_author, feats):
    """Distinctiveness curves exactly as run_full_corpus.py builds them, but
    on trimmed text. Returns {author: {feature: curve}}, {author: {feature: habit sign}}."""
    print('Building reference profiles and distinctiveness curves...', flush=True)
    ref = {a: feats[books[0][1]] for a, books in by_author.items()}
    stats = corpus_stats([d for fl in ref.values() for d in fl])
    profiles = {a: author_profile(fl, stats) for a, fl in ref.items()}
    subjects = [a for a, books in by_author.items() if len(books) >= 2]
    curves, habit = {}, {}
    for i, a in enumerate(subjects):
        own = profiles[a]
        bg = average_profiles([p for b, p in profiles.items() if b != a])
        per_book = [[distinctiveness_scores(d, own, bg, stats) for d in feats[fn]]
                    for _, fn in by_author[a][1:]]
        n = min(len(pb) for pb in per_book)
        curves[a] = {f: np.array([np.mean([pb[j].get(f, np.nan) for pb in per_book]) for j in range(n)])
                     for f in FEATURE_NAMES}
        habit[a] = {f: (np.nan if np.isnan(own[f]) or np.isnan(bg[f]) else float(np.sign(own[f] - bg[f])))
                    for f in FEATURE_NAMES}
        print(f'  [{i+1}/{len(subjects)}] {a}: {n} windows', flush=True)
    return curves, habit


# ---------------------------------------------------------------- tests ----
def run_condition(name, curves):
    s, e = CONDITIONS[name]
    print(f'Condition {name} ({COND_LABEL[name]}): testing {len(curves)} authors x {len(FEATURE_NAMES)} features...',
          flush=True)
    t0 = time.time()
    rows = []
    for i, (a, fc) in enumerate(curves.items()):
        for f in FEATURE_NAMES:
            c = fc[f]
            c = c[s:len(c) - e]
            at = accumulator_test(c)
            rows.append({'condition': name, 'author': a, 'feature': f, 'n_windows': len(c),
                         'accum_slope': at['slope'], 'accum_p': at['p_value']})
        print(f'  [{i+1}/{len(curves)}] {a} ({time.time()-t0:.0f}s)', flush=True)
    df = pd.DataFrame(rows)
    df['accum_q'] = bh_fdr(df['accum_p'].values)
    df['accum_sig'] = df['accum_q'] < ALPHA
    print(f'  -> {int(df["accum_sig"].sum())} accumulators', flush=True)
    return df


def register_directions():
    """feature -> +1/-1 for the 13 opening-register features (step 1 output)."""
    if not os.path.exists(REGISTER_CSV):
        print(f'  ({REGISTER_CSV} not found -- run opening_register_scan.py first for the mechanism check)')
        return {}
    r = pd.read_csv(REGISTER_CSV)
    r = r[r['q_1500'] < ALPHA]
    return {f: float(np.sign(z)) for f, z in zip(r['feature'], r['mean_z_1500'])}


# --------------------------------------------------------------- report ----
def compare(long, habit, reg):
    w = long.pivot_table(index=['author', 'feature'], columns='condition', values='accum_sig').astype(bool)
    q = long.pivot_table(index=['author', 'feature'], columns='condition', values='accum_q')
    w.columns = [f'sig_{c}' for c in w.columns]
    q.columns = [f'q_{c}' for c in q.columns]
    comp = w.join(q).reset_index()
    orig = pd.read_csv(ORIGINAL_CSV)[['author', 'feature', 'accum_q']].rename(columns={'accum_q': 'q_original'})
    comp = comp.merge(orig, on=['author', 'feature'], how='left')
    comp['sig_original'] = comp['q_original'] < ALPHA

    def status(r, cond):
        if r['sig_original'] and r[f'sig_{cond}']:
            return 'survived'
        if r['sig_original']:
            return 'lost'
        if r[f'sig_{cond}']:
            return 'new'
        return ''
    for c in CONDITIONS:
        comp[f'status_{c}'] = comp.apply(status, axis=1, cond=c)
    comp['register_feature'] = comp['feature'].isin(reg)
    comp['habit_direction'] = [habit[a][f] for a, f in zip(comp['author'], comp['feature'])]
    comp['habit_vs_register'] = [
        ('' if f not in reg or np.isnan(h) or h == 0 else ('matches' if h == reg[f] else 'opposes'))
        for f, h in zip(comp['feature'], comp['habit_direction'])]
    return comp


def write_report(comp, reg, path):
    n_orig = int(comp['sig_original'].sum())
    L = ['# Accumulator re-test without the opening (step 2)', '',
         f'Run {time.strftime("%Y-%m-%d")}. 33 authors x 24 features; accumulator test with the chunk-shuffle '
         'null, FDR across all 792 tests in each condition. Same curves in every condition; only the stretch '
         'tested changes.', '',
         '## Headline', '',
         '| condition | accumulators | of the original 75: kept | lost | new (not in the 75) |',
         '|---|---:|---:|---:|---:|',
         f'| {COND_LABEL["original"]} | {n_orig} | - | - | - |']
    for c in CONDITIONS:
        st = comp[f'status_{c}']
        L.append(f'| {COND_LABEL[c]} | {int(comp[f"sig_{c}"].sum())} | {int((st == "survived").sum())} | '
                 f'{int((st == "lost").sum())} | {int((st == "new").sum())} |')
    L += ['', 'How to read it: compare the "first 3,000 words cut" row with the control row. The control removes '
          'the same amount of text from the END, so its "lost" count is what you lose just from having less text. '
          'If cutting the opening loses about the same number, the opening was not creating accumulators. If it '
          'loses clearly more, some accumulators were the convention fading.', '']

    L += ['## Opening-register features vs. the rest', '',
          f'Register features (step 1, {len(reg)}): ' + ', '.join(sorted(reg)) + '.', '',
          '| features | original | first 3,000 cut: kept | lost | new | control: lost |',
          '|---|---:|---:|---:|---:|---:|']
    for lab, m in (('register features', comp['register_feature']), ('other features', ~comp['register_feature'])):
        s = comp[m]
        L.append(f'| {lab} | {int(s["sig_original"].sum())} | {int((s["status_cut_3000"] == "survived").sum())} | '
                 f'{int((s["status_cut_3000"] == "lost").sum())} | {int((s["status_cut_3000"] == "new").sum())} | '
                 f'{int((s["status_end_cut_3000"] == "lost").sum())} |')
    L += ['', 'If the opening creates accumulators, the extra losses should be in the register features, not the '
          'others.', '']

    L += ['## Mechanism check', '',
          'Among the original accumulators on register features: does the author\'s habit run with or against '
          'the opening register? The convention should create fake accumulators mainly for "opposes" authors, '
          'so those should be lost more often when the opening is cut.', '',
          '| author habit vs. register | original accumulators | lost when first 3,000 cut | lost in control |',
          '|---|---:|---:|---:|']
    for hv in ('opposes', 'matches'):
        s = comp[comp['sig_original'] & (comp['habit_vs_register'] == hv)]
        L.append(f'| {hv} | {len(s)} | {int((s["status_cut_3000"] == "lost").sum())} | '
                 f'{int((s["status_end_cut_3000"] == "lost").sum())} |')
    L += ['', 'Counts are small, so read this as a direction, not a test.', '']

    L += ['## Accumulators after cutting the first 3,000 words', '',
          '| author | feature | q (original) | q (whole, trimmed) | q (3,000 cut) | q (6,000 cut) | status |',
          '|---|---|---:|---:|---:|---:|---|']
    show = comp[comp['sig_original'] | comp['sig_cut_3000']].sort_values(['status_cut_3000', 'author', 'feature'])
    for _, r in show.iterrows():
        L.append(f"| {r['author']} | {r['feature']} | {r['q_original']:.3f} | {r['q_trimmed_full']:.3f} | "
                 f"{r['q_cut_3000']:.3f} | {r['q_cut_6000']:.3f} | {r['status_cut_3000']} |")
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')


def plot_by_feature(comp, reg, path):
    g = comp.groupby('feature')[['sig_original', 'sig_cut_3000', 'sig_cut_6000']].sum()
    g = g.loc[[f for f in FEATURE_NAMES]].sort_values('sig_original')
    y = np.arange(len(g))
    h = 0.27
    fig, ax = plt.subplots(figsize=(7.5, 8))
    for k, (col, c, lab) in enumerate((('sig_original', C_ORIG, COND_LABEL['original']),
                                       ('sig_cut_3000', C_3000, COND_LABEL['cut_3000']),
                                       ('sig_cut_6000', C_6000, COND_LABEL['cut_6000']))):
        ax.barh(y + (1 - k) * h, g[col], height=h - 0.03, color=c, label=lab)
    ax.set_yticks(y)
    ax.set_yticklabels([f + ('  (register)' if f in reg else '') for f in g.index], fontsize=8, color=INK)
    ax.set_xlabel('number of authors with a significant accumulator (of 33)', fontsize=8, color=INK)
    ax.set_title('Accumulators per feature, before and after cutting the opening', loc='left',
                 fontsize=10, color=INK)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------- main ----
def main():
    t0 = time.time()
    os.makedirs(os.path.join(OUT, 'plots'), exist_ok=True)
    by_author = discover_authors()
    feats = load_features(by_author)
    curves, habit = build_curves(by_author, feats)
    print(f'Curves ready ({time.time()-t0:.0f}s).', flush=True)

    long = pd.concat([run_condition(c, curves) for c in CONDITIONS], ignore_index=True)
    long.to_csv(os.path.join(OUT, 'accum_results_long.csv'), index=False)

    print('Comparing with the original 75 and writing outputs...', flush=True)
    reg = register_directions()
    comp = compare(long, habit, reg)
    comp.to_csv(os.path.join(OUT, 'accum_comparison.csv'), index=False)
    write_report(comp, reg, os.path.join(OUT, 'accum_retest_report.md'))
    plot_by_feature(comp, reg, os.path.join(OUT, 'plots', 'accum_by_feature.png'))

    print(f'\nOriginal: {int(comp["sig_original"].sum())} accumulators')
    for c in CONDITIONS:
        st = comp[f'status_{c}']
        print(f'{COND_LABEL[c]:36s} {int(comp[f"sig_{c}"].sum()):3d} accumulators | of the 75: '
              f'kept {int((st == "survived").sum())}, lost {int((st == "lost").sum())} | new {int((st == "new").sum())}')
    print(f'\nOutputs written to {OUT}/ (report: accum_retest_report.md) -- total {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
