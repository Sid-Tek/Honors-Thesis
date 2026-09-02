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
"""
from typing import Dict, List
from features import FEATURE_NAMES


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
    """Mean z-scored feature vector for one author's held-out reference text."""
    z_list = [zscore(d, stats) for d in reference_feature_dicts]
    profile = {}
    for f in FEATURE_NAMES:
        vals = [z[f] for z in z_list if f in z]
        profile[f] = sum(vals) / len(vals)
    return profile


def distinctiveness_scores(window_feats: Dict[str, float], own_profile: Dict[str, float],
                            other_profile: Dict[str, float], stats: Dict[str, tuple]) -> Dict[str, float]:
    z = zscore(window_feats, stats)
    scores = {}
    for f in FEATURE_NAMES:
        if f not in z:
            continue
        gap = own_profile[f] - other_profile[f]
        direction = 0.0 if abs(gap) < 1e-9 else (1.0 if gap > 0 else -1.0)
        midpoint = (own_profile[f] + other_profile[f]) / 2
        scores[f] = direction * (z[f] - midpoint)
    return scores
