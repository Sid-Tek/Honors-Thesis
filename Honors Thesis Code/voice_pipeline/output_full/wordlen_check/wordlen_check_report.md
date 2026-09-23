# Long-word vs short-word check

Authors split by their usual word length (from their reference book, vs. the other authors): **19 long-word**, **14 short-word**.

"Longer percentile" = where the opening ranks among every same-length stretch of its own book (1.0 = the opening has the longest words / most "of"; 0.5 = typical). Authors are the unit: each author's two books are averaged.

## Tests

| feature | mean, all authors | p (openings longer, all) | long-word authors | short-word authors | p (groups differ) |
|---|---:|---:|---:|---:|---:|
| average word length | 0.69 | 0.000 | 0.68 | 0.69 | 0.985 |
| syllables per word | 0.68 | 0.000 | 0.66 | 0.70 | 0.524 |
| share of one-syllable words | 0.72 | 0.000 | 0.71 | 0.73 | 0.729 |
| "of" | 0.74 | 0.000 | 0.74 | 0.73 | 0.956 |

Prediction A (descriptive openings): both groups above 0.5, no group difference.
Prediction B (voice amplified): long-word above 0.5, short-word below 0.5, groups differ.

## Bursts and anti-bursts per group (books)

Burst = opening leans toward the author's own habit (top 5% of its book). Anti-burst = opening leans away from it (bottom 5%).

| feature | group | books | bursts | anti-bursts | ~by chance (each) |
|---|---|---:|---:|---:|---:|
| average word length | long-word | 38 | 9 | 1 | 1.9 |
| average word length | short-word | 28 | 1 | 5 | 1.4 |
| syllables per word | long-word | 38 | 6 | 2 | 1.9 |
| syllables per word | short-word | 28 | 2 | 3 | 1.4 |
| share of one-syllable words | long-word | 38 | 10 | 1 | 1.9 |
| share of one-syllable words | short-word | 28 | 1 | 4 | 1.4 |
| "of" | long-word | 38 | 9 | 3 | 1.9 |
| "of" | short-word | 28 | 2 | 4 | 1.4 |

## Title/heading check

The first window includes each book's title and chapter heading. Same raw test, but the ~600-word "opening" starts later:

| feature | starts at word 0 | starts at word 300 | starts at word 600 | starts at word 1,500 |
|---|---:|---:|---:|---:|
| average word length | 0.69 (p=0.000) | 0.63 (p=0.000) | 0.61 (p=0.006) | 0.51 (p=0.451) |
| syllables per word | 0.68 (p=0.000) | 0.62 (p=0.000) | 0.60 (p=0.010) | 0.50 (p=0.535) |
| share of one-syllable words | 0.72 (p=0.000) | 0.62 (p=0.002) | 0.60 (p=0.011) | 0.51 (p=0.451) |
| "of" | 0.73 (p=0.000) | 0.64 (p=0.000) | 0.62 (p=0.001) | 0.60 (p=0.007) |
