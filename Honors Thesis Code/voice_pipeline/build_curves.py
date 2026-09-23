"""
Builds and caches every (author, feature) distinctiveness curve that
run_full_corpus.py tests, so follow-up experiments (burst_sweep.py, and
later ones) can re-test the SAME curves under different settings without
re-parsing 100 novels each time.

The curve construction is copied line-for-line from run_full_corpus.main()
(same reference/test split, same background = average of every other
author's profile, same pooling across test books, same truncation to the
shortest test book). burst_sweep.py checks this: at n_burst_windows=2 its
burst p-values must match output_full/author_feature_results.csv exactly.

Output: output_sweep/curves.npz   (one array per "author|feature" key)
        output_sweep/curves_index.csv
"""
import os
import time

import numpy as np
import pandas as pd

from features import extract_features, FEATURE_NAMES
from scoring import corpus_stats, author_profile, distinctiveness_scores
from run_full_corpus import discover_authors, book_local_windows, average_profiles

OUT_DIR = 'output_sweep'
CACHE = os.path.join(OUT_DIR, 'curves.npz')


def build_curves():
    t0 = time.time()
    by_author = discover_authors()
    author_ref_feats, author_test_files, reference_feats_all = {}, {}, []
    for author, books in by_author.items():
        author_test_files[author] = [fn for (_, fn) in books[1:]]
        feats = [extract_features(w) for w in book_local_windows(books[0][1])]
        author_ref_feats[author] = feats
        reference_feats_all.extend(feats)
    stats = corpus_stats(reference_feats_all)
    author_profiles = {a: author_profile(f, stats) for a, f in author_ref_feats.items()}
    subjects = [a for a in by_author if len(author_test_files[a]) >= 1]
    print(f"Reference profiles built ({time.time()-t0:.0f}s). {len(subjects)} subjects.")

    curves, index_rows = {}, []
    for i, author in enumerate(subjects):
        background = average_profiles([p for a, p in author_profiles.items() if a != author])
        own = author_profiles[author]
        per_book_local = []
        for fn in author_test_files[author]:
            feats = [extract_features(w) for w in book_local_windows(fn)]
            per_book_local.append([distinctiveness_scores(f, own, background, stats) for f in feats])
        n_books = len(per_book_local)
        min_len = min(len(w) for w in per_book_local)
        for feat in FEATURE_NAMES:
            curve = np.array([
                np.mean([per_book_local[b][j].get(feat, np.nan) for b in range(n_books)])
                for j in range(min_len)
            ])
            curves[f"{author}|{feat}"] = curve
            index_rows.append({'author': author, 'feature': feat, 'n_windows': min_len,
                               'n_test_books': n_books, 'has_nan': bool(np.isnan(curve).any())})
        print(f"  [{i+1}/{len(subjects)}] {author}: {min_len} windows ({time.time()-t0:.0f}s)")

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(CACHE, **curves)
    pd.DataFrame(index_rows).to_csv(os.path.join(OUT_DIR, 'curves_index.csv'), index=False)
    return curves


def load_curves(rebuild: bool = False):
    if rebuild or not os.path.exists(CACHE):
        return build_curves()
    with np.load(CACHE) as z:
        return {k: z[k] for k in z.files}


if __name__ == '__main__':
    build_curves()
