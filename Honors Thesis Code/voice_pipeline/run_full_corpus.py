"""
Full-corpus driver: every author (with >=2 books) compared against a
background built from all the OTHER authors -- no pairwise matching, no
33-choose-2 combinatorics. O(n) in the number of authors, not O(n^2).

Design:
  - Each author's own book count varies (Anon has 1, everyone else has 3).
    For each author: the earliest-year book is held out as REFERENCE (used
    to build that author's own profile); the rest are TEST books (pooled,
    same as the Dickens/Eliot pilot).
  - Each author's "own profile" is built once from their reference book.
  - An author's BACKGROUND profile = the average of every OTHER author's own
    profile (not a pool of raw text) -- so a prolific author with a longer
    reference book doesn't get to dominate "the rest" just by word count;
    every other author counts equally.
  - Authors with only 1 book (just Anon in this corpus) have an own profile
    (so they still count in everyone else's background) but are skipped as
    a SUBJECT -- there's no held-out test material to run their own
    emergence curves on without reusing the same text that built their
    profile.
  - FDR correction is study-wide (all authors x all features), not
    per-author, applied separately to the burst family and the accumulator
    family (each 33 x 24 tests).
  - Permutation nulls use the chunk shuffle (stats_tests.DEFAULT_NULL,
    Sep 23): 5-window pieces shuffled intact, because neighbouring windows
    are autocorrelated. Under this null no single author x feature burst can
    clear study-wide FDR (p floor ~5/n windows) -- see stats_tests.py.

Output: output_full/author_feature_results.csv (every author x feature row),
output_full/feature_summary.md + .csv (aggregated across authors -- this is
the actual answer to "which features burst / accumulate / both / neither,
in general"), and output_full/plots/ (the two master scatters).
"""
import os
import re
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from windowing import load_text, parse_sentences, local_windows
from features import extract_features, extract_features_cumulative, FEATURE_NAMES
from scoring import corpus_stats, author_profile, distinctiveness_scores
from stats_tests import burst_test, accumulator_test, bh_fdr, DEFAULT_NULL, CHUNK_SIZE

NULL_DESC = (f'chunk shuffle ({CHUNK_SIZE}-window pieces kept intact)' if DEFAULT_NULL == 'chunk'
             else 'single-window shuffle')
from synthesis import plot_feature_curve

CORPUS_DIR = 'real_corpus'
WINDOW_SIZE = 300
OUT_DIR = 'output_full'
FNAME_RE = re.compile(r'^([A-Za-z]+)_(.+)_(\d{4})\.txt$')


def discover_authors():
    by_author = defaultdict(list)
    for fn in sorted(os.listdir(CORPUS_DIR)):
        m = FNAME_RE.match(fn)
        if not m:
            continue
        author, title, year = m.groups()
        by_author[author].append((int(year), fn))
    for author in by_author:
        by_author[author].sort()  # earliest year first
    return by_author


def book_local_windows(filename):
    text = load_text(os.path.join(CORPUS_DIR, filename))
    sents = parse_sentences(text)
    return local_windows(sents, WINDOW_SIZE)


def average_profiles(profiles):
    # nanmean, not mean: for a floored feature (see scoring.MIN_RAW_COUNT),
    # most authors' own_profile[f] will legitimately be NaN (e.g. the ~27
    # authors who never use a dash at all). A plain mean would let those
    # NaNs poison the whole background average to NaN, silently erasing
    # every subject's result for that feature even when the subject AND a
    # handful of other authors have perfectly good data. nanmean instead
    # averages over whichever authors do have a trustworthy value.
    return {f: float(np.nanmean([p[f] for p in profiles])) for f in FEATURE_NAMES}


def main():
    t_start = time.time()
    by_author = discover_authors()
    print(f"Found {len(by_author)} authors, {sum(len(v) for v in by_author.values())} books total.")

    # ---- Pass 1: build every author's own profile from their reference book ----
    print("\nBuilding reference profiles for all authors...")
    reference_feats_all = []   # for the global z-score stats
    author_ref_feats = {}      # author -> list of feature dicts (their reference book's windows)
    author_test_files = {}     # author -> list of test filenames
    for author, books in by_author.items():
        ref_file = books[0][1]
        author_test_files[author] = [fn for (_, fn) in books[1:]]
        loc = book_local_windows(ref_file)
        feats = [extract_features(w) for w in loc]
        author_ref_feats[author] = feats
        reference_feats_all.extend(feats)
    print(f"  ...done ({time.time()-t_start:.1f}s elapsed)")

    stats = corpus_stats(reference_feats_all)
    author_profiles = {a: author_profile(f, stats) for a, f in author_ref_feats.items()}

    subjects = [a for a in by_author if len(author_test_files[a]) >= 1]
    excluded = [a for a in by_author if a not in subjects]
    print(f"\n{len(subjects)} authors have >=2 books and will be run as subjects.")
    if excluded:
        print(f"Excluded (only 1 book, background-only): {excluded}")

    # ---- Pass 2: for each subject, background = average of every OTHER author's profile ----
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'plots'), exist_ok=True)
    rows = []
    for i, author in enumerate(subjects):
        t_author = time.time()
        other_profiles = [p for a, p in author_profiles.items() if a != author]
        background = average_profiles(other_profiles)
        own = author_profiles[author]

        test_files = author_test_files[author]
        per_book_local = []
        n_windows = []
        for fn in test_files:
            loc = book_local_windows(fn)
            n_windows.append(len(loc))
            feats = [extract_features(w) for w in loc]
            per_book_local.append([distinctiveness_scores(f, own, background, stats) for f in feats])

        n_books = len(per_book_local)
        min_len = min(len(w) for w in per_book_local)

        for feat in FEATURE_NAMES:
            curve = np.array([
                np.mean([per_book_local[b][j].get(feat, np.nan) for b in range(n_books)])
                for j in range(min_len)
            ])
            bt = burst_test(curve)
            at = accumulator_test(curve)
            rows.append({
                'author': author, 'feature': feat, 'n_test_books': n_books,
                'n_windows': min_len,
                'burst_stat': bt['stat'], 'burst_p': bt['p_value'],
                'accum_slope': at['slope'], 'accum_p': at['p_value'],
            })

        print(f"  [{i+1}/{len(subjects)}] {author}: {n_books} test book(s), "
              f"{n_windows} windows -- {time.time()-t_author:.1f}s")

    df = pd.DataFrame(rows)

    # ---- Study-wide FDR correction (not per-author) ----
    df['burst_q'] = bh_fdr(df['burst_p'].values)
    df['accum_q'] = bh_fdr(df['accum_p'].values)
    df['burst_sig'] = df['burst_q'] < 0.05
    df['accum_sig'] = df['accum_q'] < 0.05

    def call(row):
        if row['burst_sig'] and not row['accum_sig']:
            return 'burst'
        if row['accum_sig'] and not row['burst_sig']:
            return 'accumulator'
        if row['burst_sig'] and row['accum_sig']:
            return 'both'
        return 'null'

    df['verdict'] = df.apply(call, axis=1)
    df.to_csv(os.path.join(OUT_DIR, 'author_feature_results.csv'), index=False)

    # ---- Aggregate across authors, per feature ----
    n_subjects = len(subjects)
    agg_rows = []
    for feat in FEATURE_NAMES:
        sub = df[df['feature'] == feat]
        counts = sub['verdict'].value_counts()
        n_burst = int(counts.get('burst', 0))
        n_accum = int(counts.get('accumulator', 0))
        n_both = int(counts.get('both', 0))
        n_null = int(counts.get('null', 0))
        # Under study-wide FDR correction, 'null' is almost always the single
        # largest bucket for every feature (most individual author tests
        # don't survive correction) -- comparing it against burst/accum/both
        # would just paint every feature gray and hide which features DO
        # show a real, if minority, signal. So: which of the three SIGNAL
        # categories is most common (majority among signals, not overall)?
        known_counts = {'burst': n_burst, 'accumulator': n_accum, 'both': n_both}
        dominant_signal = max(known_counts, key=known_counts.get) if max(known_counts.values()) > 0 else 'null'
        agg_rows.append({
            'feature': feat,
            'n_authors_burst': n_burst,
            'n_authors_accumulator': n_accum,
            'n_authors_both': n_both,
            'n_authors_null': n_null,
            'pct_burst': n_burst / n_subjects * 100,
            'pct_accumulator': n_accum / n_subjects * 100,
            'pct_any_signal': (n_burst + n_accum + n_both) / n_subjects * 100,
            'dominant_signal': dominant_signal,
            'mean_burst_effect': sub['burst_stat'].mean(),
            'mean_accum_slope': sub['accum_slope'].mean(),
        })
    agg_df = pd.DataFrame(agg_rows).sort_values('pct_any_signal', ascending=False)
    agg_df.to_csv(os.path.join(OUT_DIR, 'feature_summary.csv'), index=False)

    # ---- Plot 1: every (author, feature) point, colored by verdict ----
    verdict_colors = {'burst': '#a53b3b', 'accumulator': '#3b6ea5', 'both': '#8b3ba5', 'null': '#bbbbbb'}
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for verdict, color in verdict_colors.items():
        sub = df[df['verdict'] == verdict]
        ax.scatter(sub['burst_stat'], sub['accum_slope'], color=color, s=18, alpha=0.65,
                   label=f'{verdict} (n={len(sub)})')
    ax.axhline(0, color='gray', linewidth=0.7, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.7, linestyle='--')
    ax.set_xlabel('burst effect (opening minus rest)')
    ax.set_ylabel('accumulator slope (trend across book)')
    ax.set_title(f'Every author x feature point ({n_subjects} authors x {len(FEATURE_NAMES)} features)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'plots', 'master_scatter_all_points.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)

    # ---- Plot 2: one point per feature, at its across-author mean position.
    # Color = which signal (burst/accumulator/both) is most common for this
    # feature, among authors where SOMETHING was significant. Size = what
    # fraction of the 33 authors showed any significant pattern at all --
    # this is the axis that actually varies here, since under study-wide FDR
    # correction "null" dominates every feature's raw count (see README).
    fig, ax = plt.subplots(figsize=(8, 6.8))
    for _, r in agg_df.iterrows():
        color = verdict_colors.get(r['dominant_signal'], '#bbbbbb')
        size = 40 + 6 * (r['n_authors_burst'] + r['n_authors_accumulator'] + r['n_authors_both'])
        ax.scatter(r['mean_burst_effect'], r['mean_accum_slope'], color=color, s=size,
                   edgecolor='white', linewidth=0.6)
        label = f"{r['feature']} ({r['n_authors_burst']}b/{r['n_authors_accumulator']}a/{n_subjects})"
        ax.annotate(label, (r['mean_burst_effect'], r['mean_accum_slope']),
                    fontsize=7, xytext=(5, 4), textcoords='offset points')
    ax.axhline(0, color='gray', linewidth=0.7, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.7, linestyle='--')
    ax.set_xlabel('mean burst effect across authors')
    ax.set_ylabel('mean accumulator slope across authors')
    ax.set_title(f'Feature battery, averaged across {n_subjects} authors\n'
                 f'(label = feature (#authors burst / #authors accumulator / {n_subjects} total); '
                 f'point size = how many authors showed any significant pattern)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'plots', 'master_scatter_by_feature.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)

    # ---- Console + markdown summary ----
    pd.set_option('display.width', 160)
    print("\n=== Feature summary, sorted by how many authors showed ANY significant pattern ===")
    print(agg_df[['feature', 'n_authors_burst', 'n_authors_accumulator', 'n_authors_both',
                   'n_authors_null', 'dominant_signal']].to_string(index=False))

    with open(os.path.join(OUT_DIR, 'feature_summary.md'), 'w') as f:
        f.write(f"# Corpus-wide feature summary ({n_subjects} authors, {len(by_author)} total incl. background-only)\n\n")
        f.write("Each author compared against a background = average of every OTHER author's own "
                "profile (not a matched pair, not raw pooled text). Permutation null: "
                f"{NULL_DESC}. BH-FDR is study-wide, applied separately to the burst and "
                f"accumulator families ({n_subjects} authors x {len(FEATURE_NAMES)} features each), "
                "not per-author.\n\n")
        if excluded:
            f.write(f"Excluded as subjects (only 1 book available): {excluded}\n\n")
        f.write(agg_df.to_markdown(index=False))
        f.write('\n')

    print(f"\nTotal runtime: {time.time()-t_start:.1f}s")
    print(f"Wrote {OUT_DIR}/author_feature_results.csv (raw, {len(df)} rows), "
          f"{OUT_DIR}/feature_summary.csv/.md, {OUT_DIR}/plots/master_scatter_*.png")


if __name__ == '__main__':
    main()