# Two-timescale voice pipeline

Tests the hypothesis: voice operates on (at least) two timescales — a strong,
concentrated opening-burst for some stylistic markers, layered on a separate,
slower-accumulating signal that doesn't peak early at all.

## Status

The pipeline (Stages A–D) is fully coded, validated end to end on a
**synthetic demo corpus** with known injected ground truth (`run_pipeline.py`
→ `output/summary.md`), and now also run on a **real corpus**: 100 British
novels (33 authors × 3 books + 1 Anon, 1839–1930) supplied by the user
(`run_real_corpus.py` → `output_real/dickens/` and `output_real/eliot/`).

Two performance fixes were needed to make Stage A–C scale to real novels
(150k–360k words each, vs. ~6,000 for the synthetic demo):
- Cumulative-window features were being recomputed from scratch at every
  window (O(n²) overall) — replaced with an incremental running aggregator
  (`features.RunningAggregator` / `extract_features_cumulative`) that's
  O(n) instead. A ~360k-word novel now processes in under 2 seconds.
- The burst/accumulator permutation tests looped `scipy.stats.linregress`
  per permutation — replaced with a fully vectorized numpy version. 2,000
  permutations × 24 features on a 1,100-window book now takes ~2 seconds
  instead of being impractically slow.

### Pilot result: Dickens vs. Eliot (both directions)

Reference books: `Dickens_Oliver_1839`, `Eliot_Felix_1866`. Test books (2
per author, pooled): `Dickens_Bleak_1853` + `Dickens_Expectations_1861`;
`Eliot_Adam_1859` + `Eliot_Daniel_1876`. Chosen as a reasonably era/genre-
matched pair (both mid-Victorian "social novel" writers) with famously
distinct voices — not a definitive pair, just the first pilot. 33 authors
are available in `real_corpus/` for other comparisons.

**Eliot** shows the cleanest match to the two-timescale hypothesis: `fw_of`
and `ttr` come back as significant **burst** features (q≈0.024) — a sharp
spike in the first two ~300-word windows that decays to a stable baseline —
while `hapax_rate`, `semicolon_rate`, and `exclaim_rate` come back as
significant **accumulators** (q≈0.004) — flat-to-noisy early, then a
genuine rising trend across the whole book. The two-cluster check
(silhouette 0.41) shows these sitting in separate corners of the
burst-effect-vs-accumulator-slope scatter, not a continuum.

**Dickens** shows a real two-family split too (silhouette 0.52), but along
a *rising vs. falling* axis rather than *burst vs. accumulate* — `ttr`,
`syll_per_word`, `pct_monosyllabic`, `avg_word_len`, and `fw_was` all climb
steadily across the book (q<0.04), but nothing cleared the burst-
significance bar in this run. Worth noting rather than smoothing over: the
hypothesis's burst half didn't show up for this author on this feature
battery — that's a real (if partial) result, not a pipeline failure.

**Known caveat, discovered on real data:** `ttr` (type-token ratio) and
`hapax_rate` are mechanically sample-size-dependent — their raw value
shrinks as a window gets bigger, independent of any authorial signal. This
shows up as a steep, misleading-looking drop in their *cumulative*-window
plot panel (now labeled `[sample-size artifact]` in the generated plots).
It does **not** affect the actual burst/accumulator significance tests,
which only ever use the fixed-300-word *local*-window curve — but don't
read the cumulative panel for those two features as a real trend. A
standardized/moving-average TTR would remove the need for this caveat in a
future iteration.

## The four stages

- **Stage A** (`windowing.py`, `features.py`) — loads a book, splits it into
  local windows (fixed word count, sequential from the start) and cumulative
  windows (first N words, first 2N, ...), and extracts a battery of ~24
  features per window: sentence length, type-token ratio, hapax rate,
  average word length, comma/semicolon/dash rates, question/exclamation
  rates, subordination/coordination rates, syllables-per-word,
  percent-monosyllabic, and the rate of 10 common function words.
- **Stage B** (`scoring.py`) — builds a z-scored reference profile per
  author from held-out text, then scores every window on how far it leans
  toward its own author vs. the comparison author, per feature.
- **Stage C** (`stats_tests.py`) — per feature: a permutation-based burst
  test (is the opening anomalous vs. the rest of the book?) and accumulator
  test (is there a genuine rising trend across position?), with
  Benjamini–Hochberg FDR correction across the whole battery.
- **Stage D** (`synthesis.py`) — turns each feature's burst/accumulator
  results into a 2D shape descriptor, and checks whether the whole battery
  actually separates into two clusters (via k=2 clustering + silhouette
  score) rather than spreading out on a continuum. Also produces the plots.

## Known limitations (by design, given this sandbox has no live pip/nltk access)

- No real POS tagger — subordination/coordination rates use a small
  hand-picked word list, not a parser. Fine as a first pass; swap in spaCy
  or NLTK's tagger for production-quality syntax features.
- Syllable counts use a vowel-group heuristic, not the CMU Pronouncing
  Dictionary — reasonable for English but not exact, and stress-pattern
  features (which the original methodology discussion mentioned) aren't
  implemented yet.
- Sentence splitting is a simple regex on `.!?` — it will mis-split some
  abbreviations and honorifics. Good enough for a first pass on clean prose;
  worth revisiting if a corpus has heavy dialogue-tag punctuation.
- The demo corpus is synthetic placeholder text (see `synthetic_corpus.py`)
  with deliberately injected effects, generated purely to prove the
  statistics work — it says nothing about your actual authors' voices.

## Full-corpus run (all 33 authors, no pairwise matching)

`run_real_corpus.py` compares two specific authors — good for a close read
of one pair, but doesn't scale (33 authors = 528 possible pairs). `run_full_corpus.py`
instead compares **each author against a background built from every OTHER
author's own profile** (not raw pooled text — every other author counts
equally regardless of how long their reference book is). That's 33 runs,
not 528, and it takes about a minute end to end.

Design notes:
- Each author's earliest-year book is held out as their reference (builds
  their own profile); the rest are pooled test books, same as the pilot.
- Anon has only 1 book in the corpus, so it's excluded as a subject (no
  held-out test material) but still counts toward everyone else's
  background — 33 of the 34 authors run as subjects.
- **FDR correction is applied once across the WHOLE study** (33 authors ×
  24 features × 2 tests = 1,584 tests), not per-author like the pilot. This
  is deliberately much stricter — with this many simultaneous tests, most
  individual author-feature effects that looked "significant" in isolation
  don't survive. That's the correct behavior, not a bug: it's what stops a
  finding like "Eliot's `ttr` bursts" from being over-generalized to "voice
  bursts" without corpus-wide evidence.

Run it: `python3 run_full_corpus.py`. Output in `output_full/`:
- `author_feature_results.csv` — the raw 792-row table, one row per
  (author, feature): effect size, p-value, FDR-corrected q-value, verdict.
- `feature_summary.csv` / `.md` — aggregated per feature: out of 33
  authors, how many showed that feature as a significant burst, an
  accumulator, both, or neither.
- `plots/master_scatter_all_points.png` — every one of the 792
  (author, feature) points on the burst-effect-vs-accumulator-slope plane,
  colored by verdict. This is "where each factor falls" for every author at
  once.
- `plots/master_scatter_by_feature.png` — one point per feature, at its
  average position across all 33 authors; point size = how many authors
  showed any significant pattern at all for that feature (color alone
  wasn't enough to show this, since "null" is the largest bucket for nearly
  every feature under study-wide correction — see below).

### What the first full-corpus run found

Across all 33 authors: **101 (author, feature) points came back as
significant accumulators, only 19 as bursts, and none as both.** Every
single feature's *dominant* pattern (among authors where something was
significant) was accumulation, not bursting — `fw_but` was the only feature
with zero significant authors either way. In plain terms: on this feature
battery and this corpus, the "slow accumulator" half of the two-timescale
hypothesis is the one with broad support; the "opening burst" half shows up
for a real but small minority of authors on a handful of features (`ttr`,
`avg_word_len`, `fw_of`, `pct_monosyllabic` each had 3/33 authors burst).
Encouragingly, Eliot's `ttr` burst — the pilot's headline finding — shows up
again here independently, which is a real cross-check, not the same result
counted twice (this run's background is "average of all 32 other authors,"
completely different from the pilot's "Dickens specifically").

**Second caveat found by inspection (in addition to the ttr/hapax_rate
sample-size note above):** `dash_rate` produced two wild outliers (one
author's burst effect near +10, another's near -7) that were NOT
significant after correction (q=0.21 and q=1.0). Dashes are rare enough
that when an author's reference book happens to use very few of them, the
z-score denominator (that feature's standard deviation) gets small and any
window with a slightly unusual dash count produces an inflated score. Worth
treating `dash_rate` results cautiously, or adding a minimum-count floor
before z-scoring, in a future pass.

## Running this on a different author pair from the real corpus

`real_corpus/` has 100 books from 33 authors (3 each) + 1 Anon — filenames
are `Author_Title_Year.txt`. To try a different pair, edit the four file
lists near the bottom of `run_real_corpus.py`'s `main()`:

```python
dickens_ref_windows = [book_local_windows('Dickens_Oliver_1839.txt')]
eliot_ref_windows   = [book_local_windows('Eliot_Felix_1866.txt')]
dickens_test_files  = ['Dickens_Bleak_1853.txt', 'Dickens_Expectations_1861.txt']
eliot_test_files    = ['Eliot_Adam_1859.txt', 'Eliot_Daniel_1876.txt']
```

Swap in any other author's filenames (one held out as reference, the rest
as pooled test books) and re-run: `python3 run_real_corpus.py`. Remember the
corpus-design controls from the original methodology notes — try to match
era/genre where you can, and use multiple books per author (the synthetic
demo run shows concretely why a single book is often too thin to survive
FDR correction on an opening-burst effect).

## Running this on books outside the supplied corpus

Same idea, just point `book_local_windows()` at a different folder/file
path (or copy new `.txt` files into `real_corpus/`). Plain text only — if
you have `.docx`/`.pdf`/ebook files, convert to plain text first.

## Files

- `windowing.py`, `features.py` — Stage A (incl. `RunningAggregator` for
  fast cumulative features on long texts)
- `scoring.py` — Stage B
- `stats_tests.py` — Stage C (vectorized permutation tests)
- `synthesis.py` — Stage D (stats + plotting)
- `synthetic_corpus.py` — synthetic demo corpus generator (placeholder only)
- `run_pipeline.py` — driver for the synthetic ground-truth validation run
- `run_real_corpus.py` — driver for the two-author pilot (Dickens vs. Eliot)
- `run_full_corpus.py` — driver for the full 33-author, no-pairwise-matching run
- `real_corpus/` — the 100-book corpus (not included in the delivered zip —
  copy your own `corpus/` folder here, or point `CORPUS_DIR` at it)
- `output/` — synthetic validation results: `feature_results.csv`,
  `summary.md`, `plots/*.png`
- `output_real/dickens/`, `output_real/eliot/` — two-author pilot results,
  same file layout
- `output_full/` — full 33-author results: `author_feature_results.csv`,
  `feature_summary.csv`/`.md`, `plots/master_scatter_*.png`
