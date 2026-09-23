# Burst-window sweep: Sep 23, 2026

**Question:** Did the Sep 18 run miss bursts because it only checked the first ~600 words of each book?

**Short answer:** Mostly no. Widening the opening does turn up more author × feature bursts under the original test. But the pattern is still a minority one, with only a modest excess over chance. It is not a corpus-wide effect for any feature. The scan found no significant mid-book bursts anywhere, though it can only detect large ones. Separately, the checks exposed a weakness in the original permutation null that also affects the accumulator count.

Scripts: `build_curves.py` builds and caches the exact curves `run_full_corpus.py` tests. `burst_sweep.py` runs sections A and B. `null_checks.py` runs the calibration numbers in section C. At the original 600-word setting, the sweep reproduces the Sep 18 burst verdicts exactly: the same 7 author × feature pairs.

---

## A. Wider openings (first 300 to 3,900 words)

| Opening | Bursts, original test (q < .05) | Authors | Openings in the top 5% of their own book (chance ≈ 39–46) |
|---:|---:|---:|---:|
| 300 words | 0 | 0 | 60 |
| **600 words (Sep 18)** | **7** | **4** | 70 |
| 900 words | 8 | 4 | 63 |
| 1,500 words | 27 | 13 | 61 |
| 2,400 words | 44 | 18 | 58 |
| 3,900 words | 47 | 20 | 57 |

The "chance" range comes from two estimates. 39 is 5% of 780. 46 comes from a placebo run: the same rank test applied at a random mid-book position instead of the opening flagged 5.9% of curves.

**What this shows**

- **The original test finds more bursts as the opening widens: 7 at 600 words, 27 at 1,500, 47 at 3,900.** Most of these are real in a narrow sense. In 25 of the 27 at 1,500 words, the opening ranks in the top 5% of all same-length stretches of that book, after removing any linear trend.
- **The excess over chance is small.** At every opening length, about 57–70 of the 780 curves have an opening in the top 5% of their book, against roughly 39–46 expected by chance. That is roughly 15–30 curves above chance, not a wave of hidden bursts.
- **The new bursts are mostly one trait counted several times.** At 1,500 words, 14 of the 27 are `avg_word_len`, `syll_per_word` or `pct_monosyllabic`, which all measure word length. So "Trollope bursts on 3 features" is closer to "Trollope's opening uses unusually long words." A plausible reason is that openings are description-heavy, with little dialogue. The dialogue-share check would test this.
- **Testing six opening lengths is itself a multiple comparison.** A result that only appears at one length is weak evidence.

**Corpus level, per feature.** This part asks whether openings are more author-distinctive than the rest of the book on average across all 33 authors. The answer is no for every feature at every opening length. The closest case is `fw_it` at 2,400 words (q = 0.066).

Two features go the opposite way significantly. For `sent_len_mean` (300–1,500 words) and `sent_len_std` (300 words), openings are *less* author-typical than the rest of the book. Treat this cautiously. The first "sentence" of many books absorbs the title, epigraph and chapter heading. In *Tess*, for example, the first sentence is 100 words long. Front-matter trimming, as in the old n-gram pipeline, would be needed before interpreting this.

**The 7 original bursts hold up per author.** Each one is the most distinctive or near-most-distinctive 600 words of its book (top 0.2–0.5%). But none of them could pass study-wide correction under a null that allows for autocorrelation (see C). They are therefore defensible as individual author findings, not as corpus-wide effects.

## B. Sliding scan: the most distinctive block anywhere in the book

Blocks of about 600, 1,500 and 3,000 words were slid across the whole book, after linear detrending so that accumulators cannot pose as bursts at the end.

- **0 significant peaks at any block length** under the chunk-shuffle null. The peak locations are spread roughly evenly across the book, with no pile-up in the opening or at any particular point.
- **The scan can only catch large bursts.** When a 1,500-word burst was injected into real curves, the scan found it in the right place 41% of the time at 2 standard deviations, 6% at 1 SD and 3% at 0.5 SD.
- **So large mid-book bursts can be ruled out; modest ones cannot.** A pre-specified location, like the opening, is far easier to test than "anywhere".

## C. Methodological issue: the single-window shuffle null is too lenient

Neighbouring 300-word windows are correlated: the median lag-1 autocorrelation is 0.13, and it fades by about lag 5, which is roughly the length of a scene. Shuffling single windows ignores this.

Calibration on curves with no real position effect:

| Test | Should flag | Single-window shuffle flags | 5-window chunk shuffle flags |
|---|---:|---:|---:|
| Scan (1,500-word blocks) | 5% | **22.5%** | 8.3% |
| Opening test (300 to 3,900 words) | 5% | 4–8% | – |

- **The scan's "significant" peaks under the old null (75–81) are artifacts.** That is why the scan uses the chunk-shuffle null.
- **The opening test is only mildly affected** at 4–8% instead of 5%.
- **The accumulator test is affected as well.** Re-run with the chunk-shuffle null, accumulators fall from **107 to 75**. Accumulation is still by far the dominant pattern, but 75 is the more defensible number.

The honest-null consequence is also why no single author × feature opening can pass study-wide correction. A book of n windows only has about n stretches to compare its opening against, so the smallest possible p-value is about 1/n, for example 1/300. FDR across about 780 tests needs p around 0.0001. Corpus-level questions have to be answered by pooling across authors, as the per-feature test in section A does.

**Ties (minor).** For discrete-valued features such as `question_rate`, whether a tied permutation counts as "at least as extreme" depended on floating-point rounding. It shifted individual p-values by up to 0.6 between your Windows run and this Linux run. The sweep counts ties conservatively. The Sep 18 burst verdicts are identical either way, but `stats_tests.py` should get the same 1e-9 tolerance.

## Suggested thesis wording

> Across 33 authors, gradual accumulation is the dominant timing pattern for voice features (75 significant author × feature accumulators under an autocorrelation-robust null). Opening bursts exist but are author-specific rather than general. For a minority of authors, the opening 600–3,900 words is among the most distinctive stretches of the book, most often on word-length features. Averaged across authors, however, no feature is more author-distinctive in the opening than elsewhere, and no significant bursts appear later in the text.

## Next steps

1. Switch `accumulator_test` and `burst_test` to the chunk-shuffle null, and add the tie tolerance, so the headline numbers in `run_full_corpus.py` are the robust ones.
2. Trim front matter before windowing. This affects the sentence-length opening result and any first-window measure.
3. Dialogue-share check. Are word-length openings and accumulators just tracking how much of a window is dialogue?
4. Shuffled-author control for the 75 accumulators.

## Files (output_sweep/)

- `opening_sweep_results.csv`: every author × feature × opening length, with the original test p/q, the rank p/q and the opening percentile
- `opening_sweep_summary.csv`: the table above, plus the list of significant pairs at each length
- `opening_corpus_level.csv`: the per-feature test pooled across authors
- `scan_results.csv`, `scan_summary.csv`: peak location and p/q for every curve and block length
- `scan_calibration.csv`: false-positive and detection rates
- `null_checks.txt`, `accumulator_null_comparison.csv`: section C calibration, and every accumulator under both nulls
- `plots/opening_sweep.png`, `plots/opening_corpus_level.png`, `plots/scan_peak_positions.png`
