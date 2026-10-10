# Attribution trend test: does accuracy keep rising after the opening?

Run 2026-10-10. 31 authors; passages from 0 to 28,000 words; authors resampled 5,000 times. Slopes are percentage points of accuracy per 10,000 words of position, from a straight line fitted to every passage position at or after the start shown.

## Burrows' Delta (primary): main fit, from 4,000 words

| training | accuracy slope (per 10k words) | 95% interval | verdict | authors rising / falling | smallest slope detectable |
|---|---:|---|---|---:|---:|
| whole-book training | +3.7 | +0.3 to +6.9 | **rising** | 20 / 10 (sign test p = 0.10) | 4.7 |
| whole books, register words removed | +4.2 | +0.9 to +7.6 | **rising** | 19 / 10 (sign test p = 0.14) | 4.7 |
| whole books, first 3,000 words dropped | +3.3 | -0.0 to +6.6 | **no clear trend** | 20 / 10 (sign test p = 0.10) | 4.7 |
| same-position training (equal amount) | +2.9 | -1.4 to +7.2 | **no clear trend** | 15 / 14 (sign test p = 1.00) | 6.2 |
| random later training (equal amount) | +2.9 | -0.5 to +6.4 | **no clear trend** | 17 / 13 (sign test p = 0.58) | 4.9 |

## Cosine Delta (check): main fit, from 4,000 words

| training | accuracy slope (per 10k words) | 95% interval | verdict | authors rising / falling | smallest slope detectable |
|---|---:|---|---|---:|---:|
| whole-book training | +3.2 | -0.5 to +7.0 | **no clear trend** | 21 / 9 (sign test p = 0.04) | 5.3 |
| whole books, register words removed | +3.3 | -1.2 to +7.5 | **no clear trend** | 20 / 10 (sign test p = 0.10) | 6.1 |
| whole books, first 3,000 words dropped | +3.5 | -0.5 to +7.3 | **no clear trend** | 21 / 9 (sign test p = 0.04) | 5.5 |
| same-position training (equal amount) | +2.3 | -3.1 to +7.7 | **no clear trend** | 16 / 13 (sign test p = 0.71) | 7.6 |
| random later training (equal amount) | +2.6 | -1.3 to +6.4 | **no clear trend** | 17 / 13 (sign test p = 0.58) | 5.5 |

## Sensitivity to where the line starts (Delta, accuracy slope per 10k words, 95% interval)

| training | from 2k | from 4k | from 6k | from 8k | from 12k |
|---|---|---|---|---|---|
| whole-book training | +2.9 (-0.3 to +6.0) | +3.7 (+0.3 to +6.9) | +3.8 (+0.3 to +7.4) | +4.1 (+0.7 to +7.8) | +6.9 (+0.8 to +13.3) |
| whole books, register words removed | +2.8 (-0.4 to +6.0) | +4.2 (+0.9 to +7.6) | +3.8 (+0.3 to +7.4) | +4.3 (+0.6 to +8.2) | +7.7 (+1.8 to +14.0) |
| whole books, first 3,000 words dropped | +2.2 (-0.9 to +5.4) | +3.3 (-0.0 to +6.6) | +3.1 (-0.3 to +6.6) | +3.6 (-0.2 to +7.5) | +7.2 (+1.1 to +13.6) |
| same-position training (equal amount) | +2.2 (-2.1 to +6.4) | +2.9 (-1.4 to +7.2) | +2.8 (-2.0 to +7.7) | +2.9 (-2.6 to +8.7) | +2.5 (-4.9 to +10.1) |
| random later training (equal amount) | +1.6 (-1.7 to +5.0) | +2.9 (-0.5 to +6.4) | +4.5 (+0.6 to +8.6) | +4.0 (-0.5 to +8.9) | +4.4 (-0.5 to +9.5) |

## Same, with mean reciprocal rank (a finer measure than right/wrong; slope per 10k words, x100)

| training | from 2k | from 4k | from 6k | from 8k | from 12k |
|---|---|---|---|---|---|
| whole-book training | +2.1 (-0.4 to +4.6) | +2.6 (+0.0 to +5.2) | +2.8 (+0.2 to +5.6) | +2.7 (-0.0 to +5.6) | +5.1 (+0.6 to +9.9) |
| whole books, register words removed | +2.0 (-0.4 to +4.4) | +2.9 (+0.2 to +5.6) | +2.9 (+0.1 to +5.7) | +2.9 (-0.0 to +6.1) | +5.8 (+1.3 to +10.6) |
| whole books, first 3,000 words dropped | +1.8 (-0.5 to +4.3) | +2.5 (-0.1 to +5.1) | +2.5 (-0.1 to +5.2) | +2.5 (-0.3 to +5.5) | +5.2 (+0.6 to +10.1) |
| same-position training (equal amount) | +1.8 (-1.8 to +5.4) | +2.4 (-1.5 to +6.1) | +2.4 (-1.6 to +6.2) | +2.7 (-2.1 to +7.1) | +2.2 (-4.1 to +8.3) |
| random later training (equal amount) | +2.0 (-0.5 to +4.6) | +2.4 (-0.2 to +4.9) | +3.6 (+0.7 to +6.6) | +3.2 (-0.1 to +6.6) | +3.0 (-0.6 to +6.6) |

## Plain reading

* **Whole-book training:** accuracy rises by about +3.7 points per 10,000 words (interval +0.3 to +6.9), roughly +8.8 points across the range tested.
* **Same-position training (the cleaner test):** no clear trend: the slope is +2.9 points per 10,000 words (interval -1.4 to +7.2). A rise steeper than about 7.2 points per 10,000 words (about 17.4 points over the whole range) is ruled out; a smaller rise is not.

How to use this: the claim "voice keeps building after the opening register" predicts a clearly positive slope, especially in the same-position condition. A positive whole-book slope with a flat same-position slope would mean the rise comes from the training text being mostly late text, not from voice strengthening. A flat slope in both is evidence against build-up (within the limits stated by the interval); a wide interval that includes both zero and a large rise means the data cannot decide.

Limits: one corpus of ~30 authors, three books each; straight-line fit (a rise that levels off would be understated); accuracy at a single position is noisy, which is why the whole range is used.
