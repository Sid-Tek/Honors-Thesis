"""
Stage B: reference profiles + per-window, per-feature distinctiveness scores.

For each feature, define the axis connecting the two authors' typical
(z-scored) values. distinctiveness(window, feature) is how far along that
axis -- in the direction of the window's OWN author, relative to the
midpoint between the two authors -- the window falls:

    direction = sign(own_profile[f] - other_profile[f])
    midpoint  = (own_profile[f] + other_profile[f]) / 2
    score     = direction * (window_z[f] - midpoint)

Positive => the window leans toward its own author's characteristic side of
the spectrum; the further past its own author's typical position (i.e. a
LOUDER-than-usual dose of that author's own habit), the higher the score.
This is deliberately NOT |distance to other| - |distance to own|: that
formulation punishes a window for being an unusually extreme (even if
correctly-directioned) example of its own author's habit, which is exactly
the "opening burst" pattern this pipeline is trying to detect.

Rare-feature floor (fixes the dash_rate outlier bug from the Aug 18
full-corpus run -- corrected Sep 13 after real-corpus testing exposed a
second-order bug in the first version, see below): a rate built from a
small integer count divided by window size is dominated by pure counting
noise when that count is small. z-scoring that noisy rate against a
corpus-wide std that is itself small, because dashes are rare almost
everywhere, turns ordinary sampling noise into apparently large effect
sizes (the two outliers that didn't survive FDR correction, q=0.21/1.0).

The floor operates at the AUTHOR level, not the window level: it checks
whether an author's REFERENCE BOOK, in total across all its windows,
contains enough raw occurrences of the rare event to trust that author's
own_profile baseline for that feature at all. If not, own_profile[f] is
NaN for that author (and propagates through distinctiveness_scores as
before). Individual test windows are NOT floored -- they're z-scored
normally, same as any other feature, once an author has cleared the
baseline-trust bar.

(First version of this fix floored every individual window instead, which
seemed reasonable in isolation but broke catastrophically once tested on
the real 33-author corpus: run_full_corpus.py's curve-building step
averages across an author's pooled test books at every window position,
and burst_test/accumulator_test treat a single NaN anywhere in that curve
as "abort, insufficient data for the whole book." A per-window floor
essentially guarantees at least one low-count window somewhere across
hundreds of positions, so it nuked the curve for every author regardless
of overall dash usage -- confirmed directly: Ward uses dashes heavily in
both her reference book and both test books, and still came back fully
NaN under the per-window version. Gating once, at the reference-book
level, avoids injecting per-window NaNs into that averaged curve at all.)
"""
import math
from typing import Dict, List
from features import FEATURE_NAMES

# Minimum TOTAL raw occurrence count across an author's entire reference
# book before that author's own_profile for this feature is trusted. Only
# dash_rate has hit this in practice; add another entry here if a
# different sparse feature turns out to need it.
MIN_RAW_COUNT = {
    'dash_rate': 20,
}

# Maps a floored rate feature to the companion raw-count key features.py
# emits alongside it (e.g. 'dash_rate' -> 'dash_count').
RATE_COUNT_KEY = {
    'dash_rate': 'dash_count',
    'semicolon_rate': 'semicolon_count',
    'comma_rate': 'comma_count',
    'question_rate': 'question_count',
    'exclaim_rate': 'exclaim_count',
}


def _reference_passes_floor(f: str, reference_feature_dicts: List[Dict[str, float]]) -> bool:
    floor = MIN_RAW_COUNT.get(f, 0)
    if floor <= 0:
        return True
    count_key = RATE_COUNT_KEY.get(f)
    if not count_key:
        return True
    total = sum(d.get(count_key, 0) for d in reference_feature_dicts)
    return total >= floor


def corpus_stats(reference_feature_dicts: List[Dict[str, float]]) -> Dict[str, tuple]:
    """Mean/std per feature, pooled across all authors' reference material.
    Used to z-score everything onto a comparable scale."""
    stats = {}
    for f in FEATURE_NAMES:
        vals = [d[f] for d in reference_feature_dicts if f in d]
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        sd = var ** 0.5
        stats[f] = (m, sd if sd > 1e-9 else 1e-9)
    return stats


def zscore(feats: Dict[str, float], stats: Dict[str, tuple]) -> Dict[str, float]:
    return {f: (feats[f] - stats[f][0]) / stats[f][1] for f in FEATURE_NAMES if f in feats}


def author_profile(reference_feature_dicts: List[Dict[str, float]], stats: Dict[str, tuple]) -> Dict[str, float]:
    """Mean z-scored feature vector for one author's held-out reference text.
    For a floored feature (see MIN_RAW_COUNT), the entire profile value is
    NaN if the reference book's TOTAL raw occurrence count falls short --
    checked once here, not per-window."""
    z_list = [zscore(d, stats) for d in reference_feature_dicts]
    profile = {}
    for f in FEATURE_NAMES:
        if not _reference_passes_floor(f, reference_feature_dicts):
            profile[f] = float('nan')
            continue
        vals = [z[f] for z in z_list if f in z]
        profile[f] = sum(vals) / len(vals) if vals else float('nan')
    return profile


def distinctiveness_scores(window_feats: Dict[str, float], own_profile: Dict[str, float],
                            other_profile: Dict[str, float], stats: Dict[str, tuple]) -> Dict[str, float]:
    z = zscore(window_feats, stats)
    scores = {}
    for f in FEATURE_NAMES:
        if f not in z:
            continue
        own = own_profile.get(f, float('nan'))
        other = other_profile.get(f, float('nan'))
        if math.isnan(z[f]) or math.isnan(own) or math.isnan(other):
            scores[f] = float('nan')  # this author (or too much of the background) lacks a trustworthy baseline for this feature -- missing, not a real extreme value
            continue
        gap = own - other
        direction = 0.0 if abs(gap) < 1e-9 else (1.0 if gap > 0 else -1.0)
        midpoint = (own + other) / 2
        scores[f] = direction * (z[f] - midpoint)
    return scores