"""
Stage C: burst test + accumulator test, both via within-book permutation
nulls (shuffle window order, recompute the statistic, see how often the
shuffled version is as extreme as the real one), plus BH-FDR correction
across the whole feature battery.

Vectorized with numpy (all n_perm permutations generated and scored at once)
rather than looping scipy.stats.linregress calls -- real novels can have
hundreds to low thousands of local windows, and a per-permutation Python
loop calling scipy for each one does not scale to that.

Changes (Sep 23): CHUNK-SHUFFLE NULL is now the default.
  * The old null shuffled single 300-word windows, which assumes every
    window is independent of its neighbours. They aren't: a scene or a
    dialogue-heavy chapter spans several windows (median lag-1
    autocorrelation of the corpus curves ~0.13, fading by lag ~5). Single-
    window shuffling breaks those runs up, so the shuffled "chance" books
    are unrealistically smooth and ordinary runs in the real book look
    significant. Measured on curves with no real position effect, it gave
    22.5% false positives for a sliding scan (should be 5%) and 4-8% for
    the opening test (burst_sweep.py / null_checks.py, output_sweep/).
  * The chunk null cuts the curve into consecutive CHUNK_SIZE-window pieces
    (default 5, ~1,500 words; the last piece may be shorter), shuffles the
    ORDER of the pieces, and keeps each piece intact. "Chance" books keep
    realistic short-range structure.
  * null='window' reproduces the old behaviour exactly (same permutations
    as the Sep 18 run), for comparison.
  * Consequence for the burst test: a book of n windows has only ~n/5
    chunks, so there are only ~n/5 distinct "openings" a shuffle can
    produce and the smallest possible burst p is ~5/n (e.g. ~0.017 for a
    300-window book). No single author x feature burst can clear FDR across
    ~790 tests under this null. That is a real limit of asking "is THIS
    book's opening unusual?" with ~n/5 effectively independent stretches,
    not a bug. Corpus-level burst questions should pool across authors
    (see burst_sweep.py, opening_corpus_level.csv). The accumulator test
    is not limited this way (a slope over the whole book can take many
    values under chunk shuffles).
  * TIES: for discrete-valued features (question_rate, exclaim_rate, ...)
    many permuted statistics exactly equal the observed one, and whether
    `perm >= observed` counted them depended on floating-point rounding
    (p moved by up to 0.6 between Windows and Linux on the same curve).
    Ties now count as "at least as extreme" via a 1e-9 tolerance -- the
    standard conservative convention.

Changes (Sep 18):
  * N_PERM default 2,000 -> 20,000 (p-value floor 0.0005 -> 0.00005).
  * Permutation index matrix cached per (n_windows, n_perm, seed, null).
  * Permutation generation and scoring are chunked to cap peak memory.
"""
import numpy as np

N_PERM = 20000
CHUNK_SIZE = 5           # windows per chunk for the chunk-shuffle null (~1,500 words)
DEFAULT_NULL = 'chunk'   # 'chunk' (Sep 23 default) or 'window' (Sep 18 behaviour)
TIE_TOL = 1e-9
_CHUNK = 2000            # permutation rows generated / scored at a time (bounds peak memory)

_perm_cache = {}         # holds only the most recent entry


def _perm_index(n: int, n_perm: int, seed: int, null: str = DEFAULT_NULL) -> np.ndarray:
    """(n_perm, n) int32 matrix; each row is one shuffled ordering of range(n).
    null='window': independent shuffle of single windows (argsort of random
                   keys) -- identical rows to the Sep 18 code.
    null='chunk':  shuffle the order of consecutive CHUNK_SIZE-window pieces,
                   keeping each piece intact."""
    key = (n, n_perm, seed, null)
    if key not in _perm_cache:
        _perm_cache.clear()
        idx = np.empty((n_perm, n), dtype=np.int32)
        if null == 'window':
            rng = np.random.default_rng(seed)
            for start in range(0, n_perm, _CHUNK):
                stop = min(start + _CHUNK, n_perm)
                idx[start:stop] = np.argsort(rng.random((stop - start, n)), axis=1)
        elif null == 'chunk':
            rng = np.random.default_rng(seed + 7919)
            pieces = [np.arange(i, min(i + CHUNK_SIZE, n)) for i in range(0, n, CHUNK_SIZE)]
            for r in range(n_perm):
                idx[r] = np.concatenate([pieces[j] for j in rng.permutation(len(pieces))])
        else:
            raise ValueError(f"null must be 'chunk' or 'window', got {null!r}")
        _perm_cache[key] = idx
    return _perm_cache[key]


def burst_test(local_scores, n_burst_windows: int = 2, n_perm: int = N_PERM, seed: int = 0,
               null: str = DEFAULT_NULL):
    """Is the opening anomalously distinctive relative to the rest of the book?
    stat = mean(score[:n_burst_windows]) - mean(score[n_burst_windows:])
    One-sided: we only care whether the opening is HIGHER than the rest.
    See the module docstring for the chunk-null p-value floor (~5/n)."""
    scores = np.asarray(local_scores, dtype=float)
    n = len(scores)
    if n < n_burst_windows + 2 or np.isnan(scores).any():
        return {'stat': np.nan, 'p_value': np.nan, 'n_windows': n}

    observed = scores[:n_burst_windows].mean() - scores[n_burst_windows:].mean()

    perm_idx = _perm_index(n, n_perm, seed, null)
    perm_stat = np.empty(n_perm)
    for start in range(0, n_perm, _CHUNK):
        stop = min(start + _CHUNK, n_perm)
        block = scores[perm_idx[start:stop]]
        perm_stat[start:stop] = block[:, :n_burst_windows].mean(axis=1) - block[:, n_burst_windows:].mean(axis=1)

    p = (1 + np.sum(perm_stat >= observed - TIE_TOL)) / (n_perm + 1)
    return {'stat': float(observed), 'p_value': float(p), 'n_windows': n}


def accumulator_test(local_scores, n_perm: int = N_PERM, seed: int = 0, null: str = DEFAULT_NULL):
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

    perm_idx = _perm_index(n, n_perm, seed, null)
    perm_slopes = np.empty(n_perm)
    for start in range(0, n_perm, _CHUNK):
        stop = min(start + _CHUNK, n_perm)
        block = scores[perm_idx[start:stop]]
        means = block.mean(axis=1, keepdims=True)
        perm_slopes[start:stop] = (x_centered[None, :] * (block - means)).sum(axis=1) / ssx

    p = (1 + np.sum(perm_slopes >= observed - TIE_TOL)) / (n_perm + 1)
    return {'slope': observed, 'p_value': float(p), 'n_windows': n}


def opening_rank_p(local_scores, n_burst_windows: int = 2):
    """Author-level burst check: of every n_burst_windows-long stretch in the
    book (every start position), what share is at least as distinctive as the
    opening? Computed on the linearly detrended curve, so a rising or falling
    trend doesn't push the opening up or down. p = 1/(number of stretches)
    means the opening is THE most distinctive stretch in the book.

    Needs no independence assumption (so it is autocorrelation-safe), but its
    floor is ~1/n, so it answers "is this opening unusual for THIS book?" --
    not a study-wide significance test. Under no opening effect, about 1% of
    curves land at p <= 0.01 by chance (placebo check: null_checks.py)."""
    scores = np.asarray(local_scores, dtype=float)
    n, k = len(scores), n_burst_windows
    if n < 4 * k or np.isnan(scores).any():
        return np.nan
    x = np.arange(n, dtype=float)
    resid = scores - np.polyval(np.polyfit(x, scores, 1), x)
    c = np.concatenate([[0.0], np.cumsum(resid)])
    means = (c[k:] - c[:-k]) / k
    return float(np.mean(means >= means[0] - TIE_TOL))


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
