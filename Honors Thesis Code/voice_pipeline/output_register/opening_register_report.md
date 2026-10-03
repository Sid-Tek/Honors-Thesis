# Opening register scan (step 1)

Run 2026-10-03. 33 authors x 3 books, front matter trimmed. z = how far a 300-word stretch sits from the rest of its own book (in typical window-to-window SDs); authors are the unit.

## Which features shift in the opening

| feature | first 1,500 words, mean z | authors same direction | p | q (FDR, 24) | first 600 words | first 3,000 words | untrimmed, first 1,500 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **sentence length** | +0.60 | 29/33 | 0.0000 | 0.000 | +0.92 (q=0.000) | +0.42 (q=0.001) | +0.58 (q=0.000) |
| **word length (letters)** | +0.40 | 28/33 | 0.0000 | 0.000 | +0.58 (q=0.000) | +0.27 (q=0.002) | +0.47 (q=0.000) |
| **"the"** | +0.48 | 29/33 | 0.0000 | 0.000 | +0.64 (q=0.000) | +0.34 (q=0.001) | +0.49 (q=0.000) |
| **"of"** | +0.52 | 28/33 | 0.0000 | 0.000 | +0.80 (q=0.000) | +0.35 (q=0.001) | +0.57 (q=0.000) |
| **sentence-length spread** | +0.31 | 25/33 | 0.0001 | 0.000 | +0.45 (q=0.000) | +0.25 (q=0.001) | +0.42 (q=0.000) |
| **one-syllable words** | -0.38 | 27/33 | 0.0001 | 0.001 | -0.58 (q=0.000) | -0.25 (q=0.002) | -0.47 (q=0.000) |
| **syllables per word** | +0.35 | 28/33 | 0.0005 | 0.002 | +0.50 (q=0.000) | +0.23 (q=0.004) | +0.43 (q=0.000) |
| **"to"** | -0.23 | 25/33 | 0.0006 | 0.002 | -0.33 (q=0.000) | -0.17 (q=0.002) | -0.18 (q=0.040) |
| **"in"** | +0.23 | 26/33 | 0.0007 | 0.002 | +0.42 (q=0.001) | +0.12 (q=0.028) | +0.26 (q=0.000) |
| **"and"** | +0.23 | 26/33 | 0.0008 | 0.002 | +0.26 (q=0.002) | +0.18 (q=0.006) | +0.20 (q=0.009) |
| **"but"** | -0.16 | 23/33 | 0.0053 | 0.012 | -0.23 (q=0.003) | -0.12 (q=0.013) | -0.22 (q=0.001) |
| **type-token ratio (MATTR)** | -0.21 | 22/33 | 0.0112 | 0.022 | -0.29 (q=0.006) | -0.16 (q=0.027) | -0.20 (q=0.040) |
| **"was"** | +0.19 | 21/33 | 0.0184 | 0.034 | +0.22 (q=0.062) | +0.17 (q=0.013) | +0.11 (q=0.264) |
| "it" | -0.09 | 23/33 | 0.0341 | 0.059 | -0.13 (q=0.156) | -0.11 (q=0.016) | -0.12 (q=0.014) |
| commas | +0.16 | 23/33 | 0.0376 | 0.060 | +0.23 (q=0.062) | +0.17 (q=0.011) | +0.09 (q=0.308) |
| semicolons | +0.16 | 21/33 | 0.0452 | 0.068 | +0.23 (q=0.055) | +0.13 (q=0.058) | +0.10 (q=0.264) |
| subordinators | -0.12 | 19/33 | 0.0893 | 0.120 | -0.20 (q=0.051) | -0.08 (q=0.093) | -0.20 (q=0.009) |
| exclamations | -0.11 | 23/33 | 0.0903 | 0.120 | -0.28 (q=0.000) | -0.03 (q=0.668) | -0.12 (q=0.097) |
| coordinators | +0.09 | 17/33 | 0.1333 | 0.168 | +0.07 (q=0.328) | +0.08 (q=0.141) | +0.07 (q=0.390) |
| questions | -0.08 | 21/33 | 0.2178 | 0.261 | -0.15 (q=0.195) | -0.08 (q=0.210) | -0.12 (q=0.070) |
| hapax rate (MA) | -0.07 | 18/33 | 0.3423 | 0.391 | -0.09 (q=0.364) | -0.07 (q=0.315) | -0.03 (q=0.760) |
| dashes | +0.07 | 13/29 | 0.5006 | 0.546 | +0.07 (q=0.890) | +0.05 (q=0.541) | +0.08 (q=0.477) |
| "as" | -0.02 | 19/33 | 0.6258 | 0.653 | +0.05 (q=0.584) | +0.01 (q=0.814) | +0.01 (q=0.935) |
| "that" | -0.02 | 19/33 | 0.7427 | 0.743 | -0.00 (q=0.972) | -0.07 (q=0.243) | -0.01 (q=0.935) |

**13 of 24 features** shift in the first 1,500 words after FDR correction.

## Where each shift ends

| feature | direction | ends at (fade-out fit) | 95% interval | step-model end |
|---|---|---:|---:|---:|
| "in" | up | 600 | 300-1,950 | 375 |
| sentence length | up | 1,425 | 900-7,275 | 975 |
| "of" | up | 1,575 | 600-3,300 | 450 |
| "the" | up | 2,175 | 825-6,831 | 1,275 |
| one-syllable words | down | 2,475 | 675-6,375 | 1,275 |
| word length (letters) | up | 2,550 | 750-6,750 | 1,350 |
| syllables per word | up | 2,700 | 675-6,300 | 1,650 |
| "to" | down | 3,075 | 600-7,050 | 2,100 |
| "but" | down | 3,150 | 300-5,250 | 1,200 |
| type-token ratio (MATTR) | down | 3,825 | 300-6,450 | 2,700 |
| sentence-length spread | up | 4,650 | 973-11,250 | 3,300 |
| "and" | up | 5,850 | 1,650-11,700 | 4,275 |
| "was" | up | 7,050 | 525-11,700 | 2,100 |

## Calibration

Placebo (same test at random mid-book positions, 50 draws x 24 features): 5.0% of tests at p < .05 (expect ~5%); draws with any FDR-significant feature: 3 of 50.

## Front matter

Median 19 words trimmed per book; 20 of 99 books had at least one full window (300 words) of front matter; largest: James_Ambassadors_1903.txt (7,408), Eliot_Felix_1866.txt (3,584), Conrad_Nostromo_1904.txt (2,633), Haggard_Sheallan_1921.txt (2,200), Haggard_She_1887.txt (2,131).

See front_matter_cuts.csv for every cut.
