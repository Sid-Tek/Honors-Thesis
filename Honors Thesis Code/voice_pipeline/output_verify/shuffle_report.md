# Shuffled-author control (verification plan A3)

Run 2026-10-10. 30 replicates: the 100 books dealt into random groups with the real group sizes, then the full pipeline (reference profile, curves, accumulator test with FDR, pooled curve with the opening register subtracted, hinge onset) run on the fake authors. Real labels: onset 5,700 words (95% 0-12,900), 69 accumulators.

| metric | real labels | shuffled: mean | shuffled: 2.5%-97.5% | share of shuffles >= real |
|---|---:|---:|---:|---:|
| accumulators (author x feature pairs) | 69 | 45 | 24 to 67 | 3% |
| authors / groups with an accumulator | 23 | 19 | 13 to 27 | 20% |
| onset, register subtracted (words) | 5,700 | 5,870 | 0 to 13,882 | 57% |
| onset, raw curve (words) | 0 | 460 | 0 to 4,102 | 100% |
| rise, register subtracted (SD units) | 0.18 | 0.25 | -0.01 to 0.44 | 80% |
| rise, raw curve (SD units) | 0.31 | 0.37 | 0.08 to 0.57 | 80% |

Shuffled onsets within 1,500 words of the real onset: **37%** of replicates (median shuffled onset 6,000; 57% of shuffles start later than the real one).
Empirical p (shuffles at least as extreme as real, +1 correction): accumulators 0.065, rise 0.806.

**Shuffled labels often reproduce the onset timing**, so a ~6,000-word onset is not evidence of voice by itself: general drift in the pooled curve starts about as late.

The real labels do **not** clearly exceed the shuffles on accumulator count or rise: the accumulation is not distinguishable from general drift.

Notes: the shuffles use the same register correction (leave-one-out corpus opening profile) and the same 5-window chunk null as the real run. Shuffled groups mix eras, which real authors do not; if the results are borderline, that is the first thing to vary. Permutations per test: 5000.
