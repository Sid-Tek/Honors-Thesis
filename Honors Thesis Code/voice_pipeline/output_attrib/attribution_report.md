# Attribution by position (verification test 1)

Run 2026-10-10. Corpus: `real_corpus`. 31 candidate authors (chance 3.2%); 3 folds (each fold tests every author's next book, trained on the other 2); 2,000-word passages from the first 30,000 words; 500 most frequent words; 13,950 classifications. Intervals: authors resampled, 2,000 draws.
Left out because a book is shorter than the range tested: Lee, Schreiner.

## Accuracy by position: Burrows' Delta (primary)

| training | opening (0-2k) | early (2-6k) | middle (6-12k) | later (12-30k) | all positions |
|---|---:|---:|---:|---:|---:|
| all | 0.47 (0.34-0.60) | 0.57 (0.47-0.68) | 0.57 (0.49-0.66) | 0.59 (0.51-0.68) | 0.58 (0.50-0.66) |
| all_noreg | 0.47 (0.34-0.60) | 0.57 (0.47-0.67) | 0.58 (0.49-0.67) | 0.59 (0.51-0.68) | 0.58 (0.50-0.66) |
| noopen | 0.46 (0.33-0.59) | 0.58 (0.47-0.68) | 0.58 (0.50-0.67) | 0.59 (0.51-0.67) | 0.58 (0.50-0.66) |
| matched | 0.38 (0.27-0.49) | 0.36 (0.26-0.46) | 0.35 (0.27-0.44) | 0.39 (0.31-0.47) | 0.38 (0.30-0.46) |
| unmatched | 0.32 (0.22-0.43) | 0.46 (0.36-0.56) | 0.37 (0.28-0.46) | 0.43 (0.35-0.51) | 0.41 (0.34-0.49) |

## Accuracy by position: Cosine Delta (check)

| training | opening (0-2k) | early (2-6k) | middle (6-12k) | later (12-30k) | all positions |
|---|---:|---:|---:|---:|---:|
| all | 0.46 (0.34-0.60) | 0.60 (0.52-0.70) | 0.62 (0.54-0.71) | 0.63 (0.56-0.71) | 0.62 (0.54-0.69) |
| all_noreg | 0.46 (0.35-0.59) | 0.61 (0.52-0.70) | 0.62 (0.53-0.71) | 0.63 (0.55-0.70) | 0.61 (0.54-0.69) |
| noopen | 0.45 (0.34-0.58) | 0.60 (0.52-0.70) | 0.62 (0.55-0.70) | 0.63 (0.56-0.71) | 0.62 (0.55-0.69) |
| matched | 0.48 (0.37-0.61) | 0.39 (0.29-0.49) | 0.41 (0.32-0.51) | 0.43 (0.35-0.50) | 0.42 (0.35-0.50) |
| unmatched | 0.35 (0.25-0.47) | 0.45 (0.36-0.54) | 0.41 (0.33-0.50) | 0.46 (0.39-0.54) | 0.44 (0.37-0.52) |

## Predictions

Difference = first minus second; "higher" means the 95% interval is above 0, "lower" below 0.

### Burrows' Delta (primary)

| contrast | accuracy difference | verdict | mean reciprocal rank difference | verdict |
|---|---:|---|---:|---|
| P1  opening is harder than the middle (all training) | +0.100 (+0.004 to +0.204) | higher | +0.095 (+0.011 to +0.178) | higher |
| P2  accuracy keeps rising after the register: later vs early (all training) | +0.025 (-0.052 to +0.100) | inconclusive | +0.017 (-0.039 to +0.069) | inconclusive |
|     middle vs early (all training) | +0.004 (-0.072 to +0.077) | inconclusive | +0.002 (-0.055 to +0.056) | inconclusive |
|     later vs middle (all training) | +0.022 (-0.041 to +0.086) | inconclusive | +0.014 (-0.033 to +0.062) | inconclusive |
| P3  dropping openings from training helps: all positions | -0.001 (-0.007 to +0.006) | inconclusive | -0.001 (-0.005 to +0.002) | inconclusive |
|     ... for the opening passage | -0.011 (-0.032 to +0.000) | inconclusive | -0.009 (-0.023 to +0.001) | inconclusive |
|     ... for later passages | -0.005 (-0.012 to +0.004) | inconclusive | -0.003 (-0.007 to +0.002) | inconclusive |
| P4  position-matched training beats equal-size unmatched: opening | +0.054 (-0.054 to +0.161) | inconclusive | +0.054 (-0.024 to +0.134) | inconclusive |
|     ... early | -0.097 (-0.177 to -0.022) | lower | -0.061 (-0.125 to +0.001) | inconclusive |
|     ... middle | -0.018 (-0.090 to +0.054) | inconclusive | -0.025 (-0.077 to +0.024) | inconclusive |
|     ... later | -0.038 (-0.082 to +0.001) | inconclusive | -0.043 (-0.073 to -0.015) | lower |
| P5  rise after the register without the register words: later vs early | +0.024 (-0.051 to +0.100) | inconclusive | +0.013 (-0.044 to +0.069) | inconclusive |
|     removing the register words changes opening accuracy by | +0.000 (+0.000 to +0.000) | inconclusive | -0.003 (-0.009 to +0.002) | inconclusive |
|     removing the register words changes later accuracy by | -0.001 (-0.012 to +0.010) | inconclusive | -0.001 (-0.008 to +0.005) | inconclusive |

### Cosine Delta (check)

| contrast | accuracy difference | verdict | mean reciprocal rank difference | verdict |
|---|---:|---|---:|---|
| P1  opening is harder than the middle (all training) | +0.161 (+0.057 to +0.265) | higher | +0.116 (+0.044 to +0.191) | higher |
| P2  accuracy keeps rising after the register: later vs early (all training) | +0.031 (-0.045 to +0.105) | inconclusive | +0.026 (-0.032 to +0.082) | inconclusive |
|     middle vs early (all training) | +0.022 (-0.057 to +0.102) | inconclusive | +0.011 (-0.050 to +0.070) | inconclusive |
|     later vs middle (all training) | +0.010 (-0.050 to +0.073) | inconclusive | +0.015 (-0.030 to +0.062) | inconclusive |
| P3  dropping openings from training helps: all positions | -0.001 (-0.007 to +0.006) | inconclusive | -0.000 (-0.004 to +0.004) | inconclusive |
|     ... for the opening passage | -0.011 (-0.032 to +0.000) | inconclusive | -0.005 (-0.018 to +0.004) | inconclusive |
|     ... for later passages | +0.000 (-0.008 to +0.011) | inconclusive | +0.000 (-0.005 to +0.006) | inconclusive |
| P4  position-matched training beats equal-size unmatched: opening | +0.129 (+0.032 to +0.226) | higher | +0.093 (+0.019 to +0.177) | higher |
|     ... early | -0.059 (-0.129 to +0.016) | inconclusive | -0.044 (-0.099 to +0.012) | inconclusive |
|     ... middle | +0.004 (-0.054 to +0.057) | inconclusive | -0.022 (-0.066 to +0.021) | inconclusive |
|     ... later | -0.038 (-0.081 to +0.006) | inconclusive | -0.038 (-0.072 to -0.007) | lower |
| P5  rise after the register without the register words: later vs early | +0.021 (-0.053 to +0.095) | inconclusive | +0.022 (-0.035 to +0.078) | inconclusive |
|     removing the register words changes opening accuracy by | +0.000 (-0.032 to +0.032) | inconclusive | +0.003 (-0.014 to +0.019) | inconclusive |
|     removing the register words changes later accuracy by | -0.005 (-0.014 to +0.005) | inconclusive | -0.001 (-0.006 to +0.005) | inconclusive |

## Plain reading (primary method, accuracy)

* P1, the opening is harder than the middle: **higher**.
* P2, accuracy keeps rising after the register has faded (later vs. early): **inconclusive**. This is the one that bears on "voice builds after the opening"; P1 alone is expected from the register.
* P3, leaving the openings out of training helps: **inconclusive** (all positions).
* P4, same-position training beats equal-size random-position training in the opening: **inconclusive**. "higher" means the opening carries author-specific signal that shows up when compared like with like.
* P5, the rise after the register survives without the register words: **inconclusive**.

Read these together; none alone settles the hypothesis. A single "inconclusive" is not a failure to replicate, it is a lack of power at this sample size.
