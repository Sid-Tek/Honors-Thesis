# Onset robustness (verification plan A1-A2)

Run 2026-10-10. Permutations per accumulator test: 20,000; bootstrap draws: 2,000; intervals come from resampling authors. Step 3 baseline: 5,700 words (900-12,900).

## A1. Removing the circularity

Accumulators are chosen on text the onset never sees. "Circular" rows use the original whole-book selection on the same estimation text, so the gap between a clean row and its circular partner is the effect of the circularity. Split-half can lean late (features that rise early then flatten are not picked); split-book has fewer accumulators.

| analysis | onset (words) | 95% interval | authors | accumulator pairs | in 3,000-8,000? | note |
|---|---|---|---|---|---|---|
| select on last 50%, estimate on first 50% (clean) | 0 | 0-18,000 | 4 | 5 | no | curve length 100 windows |
| whole-book selection, estimate on first 50% (circular, for comparison) | 5,700 | 300-14,100 | 23 | 69 | yes | curve length 100 windows |
| select on test book 1, estimate on test book 2 (clean) | 0 | 0-18,000 | 25 | 77 | no |  |
| whole-book selection, estimate on test book 2 (circular, for comparison) | 6,900 | 0-14,100 | 23 | 69 | yes |  |
| select on test book 2, estimate on test book 1 (clean) | 300 | 0-16,800 | 26 | 72 | no |  |
| whole-book selection, estimate on test book 1 (circular, for comparison) | 4,800 | 0-18,000 | 23 | 69 | yes |  |
| both directions pooled (clean) | 0 | 0-16,800 | 29 | - | no | 149 accumulator picks across the two directions |
| both directions pooled (circular, for comparison) | 5,100 | 0-15,000 | 23 | 69 | yes |  |

## A2. One change at a time

Each row changes one thing from the baseline. The plan's criterion: an onset that stays around 3,000-8,000 words in every row is not an artifact of one analysis choice. Intervals stay wide with 23 authors; look at whether the point estimates move together rather than at any single interval.

| analysis | onset (words) | 95% interval | authors | accumulator pairs | in 3,000-8,000? | note |
|---|---|---|---|---|---|---|
| baseline (step 3: corpus register subtracted, hinge, 300-word windows) | 5,700 | 900-12,900 | 23 | 69 | yes |  |
| correction: none (raw curve) | 0 | 0-6,900 | 23 | 69 | no |  |
| correction: author's own reference-book register | 4,800 | 0-15,000 | 23 | 69 | yes |  |
| correction: drop first 3,000 words, no subtraction | 3,300 | 3,000-7,823 | 23 | 69 | yes | onset cannot be earlier than 3,000 by construction |
| shape: quadratic (smooth start) | 0 | 0-4,800 | 23 | 69 | no |  |
| shape: step (jump to a new level) | 17,700 | 7,800-18,000 | 23 | 69 | no | step position = where the level changes; later than a hinge's start for a gradual rise |
| accumulators: survivors of the 3,000-word cut (52) | 6,600 | 900-9,600 | 20 | 52 | yes | selected on text from word 3,000 on, so may lean late |
| accumulators: everything significant after the cut (56) | 9,600 | 2,700-15,000 | 21 | 56 | no | selected on text from word 3,000 on, so may lean late |
| accumulators: without exclaim/question/dash rates | 5,700 | 1,200-12,900 | 23 | 69 | yes |  |
| accumulators: non-register features only (11 features) | 10,200 | 645-18,000 | 18 | 69 | no | the register correction barely touches these |
| accumulators: non-register features only, no correction | 1,200 | 0-16,500 | 18 | 69 | no |  |
| windows: 500 words, accumulators re-selected (67) | 9,000 | 1,000-15,500 | 24 | 67 | no | 3-window chunk null, constants rescaled to the same word counts |
| windows: 500 words, raw curve | 1,000 | 0-15,000 | 24 | 67 | no |  |

**5 of 13 variants put the onset inside 3,000-8,000 words.**
Outside the range: correction: none (raw curve) (0); shape: quadratic (smooth start) (0); shape: step (jump to a new level) (17,700); accumulators: everything significant after the cut (56) (9,600); accumulators: non-register features only (11 features) (10,200); accumulators: non-register features only, no correction (1,200); windows: 500 words, accumulators re-selected (67) (9,000); windows: 500 words, raw curve (1,000).

## How to read this

- A clean row near the baseline and near its circular partner: the selection circularity did not create the onset.
- A clean row much earlier than the baseline (toward 0): the "later onset" came partly from choosing features that rise late, and voice may build from the start.
- A clean row much later: either the onset really is later, or the selection lean-late caveat applies (split-half, cut-based rows).
- Rows with very wide intervals say little either way; do not over-read their point estimates.
