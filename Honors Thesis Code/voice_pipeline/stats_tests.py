"""
Stage C: burst test + accumulator test, both via within-book permutation
nulls (shuffle window order, recompute the statistic, see how often the
shuffled version is as extreme as the real one), plus BH-FDR correction
across the whole feature battery.

Vectorized with numpy (all n_perm permutations generated and scored at once)
rather than looping scipy.stats.linregress calls -- real novels can have
hundreds to low thousands of local windows, and a per-permutation Python
loop calling scipy for each one does not scale to that.

Changes (Sep 18):
  * N_PERM default 2,000 -> 20,000. With 2,000 permutations the smallest
    possible p-value is 1/2001 = 0.0005, and BH-FDR across ~780 tests needs
    roughly 8+ tests tied at that floor before ANY of them can reach
    q < 0.05. Verdicts near that edge were being decided by how many other
    tests happened to tie at the floor, not by the data. 20,000 permutations
    lowers the floor to 0.00005.
  * The permutation index matrix is now cached per (n_windows, n_perm, seed).
    Because the seed is fixed, every feature and both tests for a given
    author already used the IDENTICAL permutations (they were regenerated
    from scratch 48 times per author). Caching is therefore an exact no-op
    on results -- verified bit-for-bit against the previous version at
    n_perm=2000 -- and it is what makes 20,000 permutations affordable.
    The first 2,000 permutations at n_perm=20000 are the same 2,000 the old
    code used, so old and new runs are directly comparable.
  * Permutation generation and scoring are chunked to cap peak memory.
"""
import numpy as np

N_PERM = 20000
_CHUNK = 2000            # rows generated / scored at a time (bounds peak memory)

_perm_cache = {}         # holds only the most recent (n, n_perm, seed) entry


def _perm_index(n: int, n_perm: int, seed: int) -> np.ndarray:
    """(n_perm, n) int32 matrix; each row is an independent random permutation
    of range(n). Same construction as before (argsort of random keys); chunking
    consumes the RNG stream in the same order, so rows are identical to what a
    single big call would produce."""
    key = (n, n_perm, seed)
    if key not in _perm_cache:
        _perm_cache.clear()
        rng = np.random.default_rng(seed)
        idx = np.empty((n_perm, n), dtype=np.int32)
        for start in range(0, n_perm, _CHUNK):
            stop = min(start + _CHUNK, n_perm)
            idx[start:stop] = np.argsort(rng.random((stop - start, n)), axis=1)
        _perm_cache[key] = idx
    return _perm_cache[key]


def burst_test(local_scores, n_burst_windows: int = 2, n_perm: int = N_PERM, seed: int = 0):
    """Is the opening anomalously distinctive relative to the rest of the book?
    stat = mean(score[:n_burst_windows]) - mean(score[n_burst_windows:])
    One-sided: we only care whether the opening is HIGHER than the rest."""
    scores = np.asarray(local_scores, dtype=float)
    n = len(scores)
    if n < n_burst_windows + 2 or np.isnan(scores).any():
        return {'stat': np.nan, 'p_value': np.nan, 'n_windows': n}

    observed = scores[:n_burst_windows].mean() - scores[n_burst_windows:].mean()

    perm_idx = _perm_index(n, n_perm, seed)
    perm_stat = np.empty(n_perm)
    for start in range(0, n_perm, _CHUNK):
        stop = min(start + _CHUNK, n_perm)
        block = scores[perm_idx[start:stop]]
        perm_stat[start:stop] = block[:, :n_burst_windows].mean(axis=1) - block[:, n_burst_windows:].mean(axis=1)

    p = (1 + np.sum(perm_stat >= observed)) / (n_perm + 1)
    return {'stat': float(observed), 'p_value': float(p), 'n_windows': n}


def accumulator_test(local_scores, n_perm: int = N_PERM, seed: int = 0):
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

    perm_idx = _perm_index(n, n_perm, seed)
    perm_slopes = np.empty(n_perm)
    for start in range(0, n_perm, _CHUNK):
        stop = min(start + _CHUNK, n_perm)
        block = scores[perm_idx[start:stop]]
        means = block.mean(axis=1, keepdims=True)
        perm_slopes[start:stop] = (x_centered[None, :] * (block - means)).sum(axis=1) / ssx

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