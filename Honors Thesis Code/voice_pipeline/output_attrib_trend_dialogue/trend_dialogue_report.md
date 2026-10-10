# Attribution trend, controlling for dialogue

Run 2026-10-10. 31 authors; passages from 4,000 words on; each author has a separate baseline; authors resampled 5,000 times. Slopes: percentage points per 10,000 words of position.
Books with no detectable dialogue, left out here: Galsworthy_River_1933.txt, Morris_Water_1897.txt.

## 1. Does dialogue rise after the opening?

Mean dialogue share of the test passages: 0k: 20%; 2k: 28%; 4k: 31%; 8k: 33%; 12k: 34%; 20k: 36%; 28k: 32%.
From 4,000 words on, dialogue share changes by -0.3 points per 10,000 words (95% interval -2.2 to +1.6).

## 2. Burrows' Delta (primary): accuracy slope, not adjusted vs dialogue held fixed

| training | not adjusted | dialogue held fixed | share of the slope kept | dialogue effect (+10 points of dialogue) |
|---|---|---|---:|---|
| whole-book training | +3.9 (+0.5 to +7.1) | +3.7 (+0.4 to +6.9) | 96% | +0.6 (-1.6 to +2.9) |
| whole books, register words removed | +4.3 (+0.9 to +7.7) | +4.2 (+0.8 to +7.6) | 97% | +0.3 (-2.0 to +2.5) |
| whole books, first 3,000 words dropped | +3.5 (+0.1 to +6.9) | +3.4 (-0.0 to +6.7) | 96% | +0.7 (-1.5 to +2.9) |
| same-position training (equal amount) | +3.1 (-1.2 to +7.6) | +3.1 (-1.3 to +7.5) | 97% | -0.2 (-1.6 to +1.3) |
| random later training (equal amount) | +3.1 (-0.2 to +6.5) | +3.0 (-0.2 to +6.2) | 95% | -0.5 (-3.0 to +2.0) |

## 2. Burrows' Delta (primary): reciprocal rank slope, not adjusted vs dialogue held fixed

| training | not adjusted | dialogue held fixed | share of the slope kept | dialogue effect (+10 points of dialogue) |
|---|---|---|---:|---|
| whole-book training | +2.8 (+0.1 to +5.4) | +2.7 (+0.0 to +5.2) | 96% | +0.6 (-1.1 to +2.4) |
| whole books, register words removed | +3.0 (+0.4 to +5.8) | +2.9 (+0.3 to +5.6) | 97% | +0.4 (-1.3 to +2.1) |
| whole books, first 3,000 words dropped | +2.7 (-0.0 to +5.3) | +2.5 (-0.1 to +5.1) | 96% | +0.6 (-1.0 to +2.3) |
| same-position training (equal amount) | +2.6 (-1.1 to +6.2) | +2.5 (-1.2 to +6.1) | 96% | +0.1 (-1.1 to +1.4) |
| random later training (equal amount) | +2.5 (+0.0 to +5.0) | +2.4 (-0.0 to +4.8) | 94% | -0.1 (-2.1 to +1.9) |

## 2. Cosine Delta (check): accuracy slope, not adjusted vs dialogue held fixed

| training | not adjusted | dialogue held fixed | share of the slope kept | dialogue effect (+10 points of dialogue) |
|---|---|---|---:|---|
| whole-book training | +3.1 (-0.7 to +6.7) | +2.9 (-1.0 to +6.5) | 93% | +0.8 (-1.2 to +2.9) |
| whole books, register words removed | +3.2 (-1.1 to +7.6) | +3.1 (-1.3 to +7.5) | 95% | -0.0 (-2.3 to +2.3) |
| whole books, first 3,000 words dropped | +3.4 (-0.7 to +7.3) | +3.2 (-0.9 to +7.2) | 94% | +0.7 (-1.4 to +2.9) |
| same-position training (equal amount) | +2.6 (-2.7 to +8.0) | +2.4 (-2.7 to +7.7) | 95% | -0.7 (-2.6 to +1.3) |
| random later training (equal amount) | +2.8 (-1.0 to +6.6) | +2.7 (-1.1 to +6.4) | 95% | -0.2 (-2.4 to +2.0) |

## 2. Cosine Delta (check): reciprocal rank slope, not adjusted vs dialogue held fixed

| training | not adjusted | dialogue held fixed | share of the slope kept | dialogue effect (+10 points of dialogue) |
|---|---|---|---:|---|
| whole-book training | +2.5 (-0.4 to +5.3) | +2.4 (-0.6 to +5.1) | 94% | +0.0 (-1.6 to +1.7) |
| whole books, register words removed | +2.6 (-0.7 to +5.6) | +2.5 (-0.9 to +5.5) | 95% | -0.4 (-2.1 to +1.3) |
| whole books, first 3,000 words dropped | +2.7 (-0.3 to +5.5) | +2.5 (-0.4 to +5.3) | 95% | -0.1 (-1.7 to +1.7) |
| same-position training (equal amount) | +2.9 (-1.5 to +7.7) | +2.8 (-1.6 to +7.5) | 95% | -0.4 (-2.0 to +1.2) |
| random later training (equal amount) | +2.3 (-0.4 to +5.1) | +2.2 (-0.5 to +4.9) | 95% | -0.2 (-2.1 to +1.6) |

## Plain reading

* **whole-book training:** the slope goes from +3.9 to +3.7 points per 10,000 words (interval +0.4 to +6.9); the rise survives: still clearly positive with dialogue held fixed.
* **same-position training (equal amount):** the slope goes from +3.1 to +3.1 points per 10,000 words (interval -1.3 to +7.5); the rise mostly survives (at least 70% of the slope remains) but its interval now includes 0.

Limits: dialogue share is one aspect of scene mix (action, description and reflection also shift with position), and it is adjusted for with a simple curve (linear plus squared). A slope that survives is evidence the rise is not just more quoted speech, not proof of voice. The two intervals (not adjusted, adjusted) come from the same resamples of authors, so compare the point estimates and the share kept more than the overlap of the intervals.
