"""
Stage C: burst test + accumulator test, both via within-book permutation
nulls (shuffle window order, recompute the statistic, see how often the
shuffled version is as extreme as the real one), plus BH-FDR correction
across the whole feature battery.

Vectorized with numpy (all n_perm permutations generated and scored at once)
rather than looping scipy.stats.linregress calls -- real novels can have
hundreds to low thousands of local windows, and a per-permutation Python
loop calling scipy for each one does not scale to that.
"""
import numpy as np


def burst_test(local_scores, n_burst_windows: int = 2, n_perm: int = 2000, seed: int = 0):
    """Is the opening anomalously distinctive relative to the rest of the book?
    stat = mean(score[:n_burst_windows]) - mean(score[n_burst_windows:])
    One-sided: we only care whether the opening is HIGHER than the rest."""
    scores = np.asarray(local_scores, dtype=float)
    n = len(scores)
    if n < n_burst_windows + 2 or np.isnan(scores).any():
        return {'stat': np.nan, 'p_value': np.nan, 'n_windows': n}

    observed = scores[:n_burst_windows].mean() - scores[n_burst_windows:].mean()

    rng = np.random.default_rng(seed)
    # argsort of random keys = a fast vectorized way to get n_perm independent
    # random permutations of range(n), one per row.
    perm_idx = np.argsort(rng.random((n_perm, n)), axis=1)
    perm_scores = scores[perm_idx]  # shape (n_perm, n)
    perm_stat = perm_scores[:, :n_burst_windows].mean(axis=1) - perm_scores[:, n_burst_windows:].mean(axis=1)

    p = (1 + np.sum(perm_stat >= observed)) / (n_perm + 1)
    return {'stat': float(observed), 'p_value': float(p), 'n_windows': n}


def accumulator_test(local_scores, n_perm: int = 2000, seed: int = 0):
    """Is there a genuine rising trend in distinctiveness across window
    position (not just a pooling/power artifact)? One-sided: positive slope."""
    scores = np.asarray(local_scores, dtype=float)
    n = len(scores)
    if n < 4 or np.isnan(scores).any():
        return {'slope': np.nan, 'p_value': np.nan, 'n_windows': n}

    x = np.arange(n, dtype=float)
    x_centered = x - x.mean()
    ssx = (x_centered ** 2).sum()

    observed = float((x_centered * (scores - scores.mean())).sum() / ssx)

    rng = np.random.default_rng(seed)
    perm_idx = np.argsort(rng.random((n_perm, n)), axis=1)
    perm_scores = scores[perm_idx]  # shape (n_perm, n)
    perm_means = perm_scores.mean(axis=1, keepdims=True)
    perm_slopes = (x_centered[None, :] * (perm_scores - perm_means)).sum(axis=1) / ssx

    p = (1 + np.sum(perm_slopes >= observed)) / (n_perm + 1)
    return {'slope': observed, 'p_value': float(p), 'n_windows': n}


def bh_fdr(pvalues):
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values in the
    same order as the input (NaNs pass through as NaN)."""
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    valid_mask = ~np.isnan(pvalues)
    adjusted = np.full(n, np.nan)
    valid_idx = np.where(valid_mask)[0]
    m = len(valid_idx)
    if m == 0:
        return adjusted
    order = valid_idx[np.argsort(pvalues[valid_idx])]
    ranked = pvalues[order]
    bh = ranked * m / (np.arange(m) + 1)
    bh_min = np.minimum.accumulate(bh[::-1])[::-1]
    bh_min = np.clip(bh_min, 0, 1)
    adjusted[order] = bh_min
    return adjusted
