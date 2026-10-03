# Accumulator re-test without the opening (step 2)

Run 2026-10-03. 33 authors x 24 features; accumulator test with the chunk-shuffle null, FDR across all 792 tests in each condition. Same curves in every condition; only the stretch tested changes.

## Headline

| condition | accumulators | of the original 75: kept | lost | new (not in the 75) |
|---|---:|---:|---:|---:|
| original run (Sep 23) | 75 | - | - | - |
| front matter trimmed, whole book | 69 | 69 | 6 | 0 |
| first 3,000 words cut | 56 | 53 | 22 | 3 |
| first 6,000 words cut | 38 | 35 | 40 | 3 |
| control: last 3,000 words cut | 65 | 57 | 18 | 8 |

How to read it: compare the "first 3,000 words cut" row with the control row. The control removes the same amount of text from the END, so its "lost" count is what you lose just from having less text. If cutting the opening loses about the same number, the opening was not creating accumulators. If it loses clearly more, some accumulators were the convention fading.

## Opening-register features vs. the rest

Register features (step 1, 13): avg_word_len, fw_and, fw_but, fw_in, fw_of, fw_the, fw_to, fw_was, pct_monosyllabic, sent_len_mean, sent_len_std, syll_per_word, ttr.

| features | original | first 3,000 cut: kept | lost | new | control: lost |
|---|---:|---:|---:|---:|---:|
| register features | 39 | 26 | 13 | 2 | 9 |
| other features | 36 | 27 | 9 | 1 | 9 |

If the opening creates accumulators, the extra losses should be in the register features, not the others.

## Mechanism check

Among the original accumulators on register features: does the author's habit run with or against the opening register? The convention should create fake accumulators mainly for "opposes" authors, so those should be lost more often when the opening is cut.

| author habit vs. register | original accumulators | lost when first 3,000 cut | lost in control |
|---|---:|---:|---:|
| opposes | 28 | 8 | 5 |
| matches | 11 | 5 | 4 |

Counts are small, so read this as a direction, not a test.

## Accumulators after cutting the first 3,000 words

| author | feature | q (original) | q (whole, trimmed) | q (3,000 cut) | q (6,000 cut) | status |
|---|---|---:|---:|---:|---:|---|
| Blackmore | ttr | 0.023 | 0.031 | 0.143 | 0.229 | lost |
| Cbronte | comma_rate | 0.048 | 0.043 | 0.094 | 0.105 | lost |
| Corelli | exclaim_rate | 0.044 | 0.065 | 0.147 | 0.074 | lost |
| Corelli | fw_as | 0.044 | 0.046 | 0.111 | 0.263 | lost |
| Doyle | fw_the | 0.036 | 0.038 | 0.169 | 0.295 | lost |
| Ford | fw_was | 0.023 | 0.016 | 0.055 | 0.219 | lost |
| Galsworthy | fw_that | 0.043 | 0.046 | 0.057 | 0.056 | lost |
| Haggard | fw_to | 0.013 | 0.040 | 0.098 | 0.256 | lost |
| Haggard | pct_monosyllabic | 0.044 | 0.070 | 0.069 | 0.074 | lost |
| James | fw_it | 0.022 | 0.038 | 0.099 | 0.073 | lost |
| James | fw_of | 0.022 | 0.274 | 0.440 | 0.560 | lost |
| James | fw_the | 0.031 | 0.365 | 0.441 | 0.469 | lost |
| James | fw_was | 0.019 | 0.129 | 0.190 | 0.198 | lost |
| James | ttr | 0.039 | 0.029 | 0.058 | 0.126 | lost |
| Lawrence | avg_word_len | 0.035 | 0.038 | 0.093 | 0.228 | lost |
| Meredith | hapax_rate | 0.039 | 0.042 | 0.071 | 0.122 | lost |
| Meredith | sent_len_mean | 0.032 | 0.035 | 0.112 | 0.559 | lost |
| Meredith | sent_len_std | 0.044 | 0.044 | 0.169 | 0.658 | lost |
| Stevenson | fw_it | 0.023 | 0.043 | 0.051 | 0.074 | lost |
| Trollope | fw_that | 0.043 | 0.047 | 0.098 | 0.180 | lost |
| Wcollins | subordination_rate | 0.031 | 0.036 | 0.079 | 0.081 | lost |
| Woolf | sent_len_mean | 0.014 | 0.015 | 0.054 | 0.224 | lost |
| Corelli | fw_of | 0.241 | 0.226 | 0.045 | 0.061 | new |
| Lytton | pct_monosyllabic | 0.053 | 0.050 | 0.027 | 0.041 | new |
| Ward | semicolon_rate | 0.063 | 0.067 | 0.016 | 0.004 | new |
| Braddon | fw_as | 0.023 | 0.035 | 0.027 | 0.024 | survived |
| Braddon | semicolon_rate | 0.002 | 0.002 | 0.002 | 0.003 | survived |
| Cbronte | dash_rate | 0.002 | 0.004 | 0.002 | 0.003 | survived |
| Cbronte | fw_in | 0.033 | 0.035 | 0.039 | 0.036 | survived |
| Cbronte | fw_it | 0.006 | 0.009 | 0.020 | 0.010 | survived |
| Cbronte | question_rate | 0.002 | 0.002 | 0.002 | 0.003 | survived |
| Conrad | dash_rate | 0.002 | 0.002 | 0.002 | 0.003 | survived |
| Conrad | sent_len_mean | 0.003 | 0.005 | 0.027 | 0.029 | survived |
| Conrad | sent_len_std | 0.002 | 0.002 | 0.010 | 0.017 | survived |
| Corelli | dash_rate | 0.002 | 0.002 | 0.006 | 0.004 | survived |
| Doyle | comma_rate | 0.002 | 0.002 | 0.008 | 0.020 | survived |
| Doyle | pct_monosyllabic | 0.039 | 0.043 | 0.045 | 0.126 | survived |
| Doyle | syll_per_word | 0.010 | 0.011 | 0.010 | 0.058 | survived |
| Eliot | comma_rate | 0.031 | 0.025 | 0.022 | 0.004 | survived |
| Eliot | fw_and | 0.031 | 0.031 | 0.027 | 0.026 | survived |
| Eliot | semicolon_rate | 0.014 | 0.025 | 0.037 | 0.056 | survived |
| Ford | fw_the | 0.025 | 0.011 | 0.029 | 0.098 | survived |
| Ford | fw_to | 0.035 | 0.015 | 0.039 | 0.032 | survived |
| Forster | fw_of | 0.023 | 0.025 | 0.027 | 0.074 | survived |
| Gaskell | fw_it | 0.002 | 0.002 | 0.002 | 0.003 | survived |
| Gissing | sent_len_mean | 0.039 | 0.043 | 0.027 | 0.198 | survived |
| Haggard | coordination_rate | 0.002 | 0.002 | 0.002 | 0.003 | survived |
| Haggard | exclaim_rate | 0.014 | 0.018 | 0.024 | 0.036 | survived |
| Haggard | syll_per_word | 0.004 | 0.015 | 0.019 | 0.029 | survived |
| Hardy | dash_rate | 0.002 | 0.002 | 0.002 | 0.003 | survived |
| Hardy | sent_len_mean | 0.002 | 0.002 | 0.002 | 0.004 | survived |
| Hardy | sent_len_std | 0.015 | 0.025 | 0.037 | 0.051 | survived |
| James | fw_that | 0.002 | 0.016 | 0.041 | 0.061 | survived |
| James | hapax_rate | 0.004 | 0.010 | 0.027 | 0.067 | survived |
| Kipling | subordination_rate | 0.030 | 0.027 | 0.016 | 0.029 | survived |
| Lawrence | hapax_rate | 0.002 | 0.002 | 0.002 | 0.003 | survived |
| Lawrence | pct_monosyllabic | 0.003 | 0.004 | 0.009 | 0.026 | survived |
| Lawrence | syll_per_word | 0.015 | 0.016 | 0.039 | 0.093 | survived |
| Lawrence | ttr | 0.002 | 0.002 | 0.004 | 0.003 | survived |
| Lytton | fw_and | 0.002 | 0.002 | 0.002 | 0.003 | survived |
| Lytton | fw_that | 0.030 | 0.030 | 0.037 | 0.069 | survived |
| Lytton | fw_to | 0.002 | 0.002 | 0.010 | 0.003 | survived |
| Meredith | avg_word_len | 0.002 | 0.002 | 0.002 | 0.003 | survived |
| Meredith | comma_rate | 0.007 | 0.006 | 0.004 | 0.009 | survived |
| Meredith | coordination_rate | 0.002 | 0.002 | 0.002 | 0.006 | survived |
| Meredith | fw_and | 0.002 | 0.002 | 0.002 | 0.003 | survived |
| Meredith | pct_monosyllabic | 0.002 | 0.002 | 0.002 | 0.003 | survived |
| Meredith | question_rate | 0.019 | 0.022 | 0.027 | 0.074 | survived |
| Meredith | subordination_rate | 0.016 | 0.019 | 0.017 | 0.071 | survived |
| Meredith | syll_per_word | 0.002 | 0.002 | 0.002 | 0.003 | survived |
| Stevenson | coordination_rate | 0.008 | 0.028 | 0.041 | 0.058 | survived |
| Stevenson | exclaim_rate | 0.045 | 0.025 | 0.027 | 0.180 | survived |
| Stevenson | fw_was | 0.033 | 0.052 | 0.027 | 0.074 | survived |
| Trollope | fw_and | 0.009 | 0.015 | 0.024 | 0.018 | survived |
| Trollope | fw_as | 0.019 | 0.025 | 0.031 | 0.029 | survived |
| Trollope | question_rate | 0.023 | 0.026 | 0.027 | 0.051 | survived |
| Trollope | sent_len_mean | 0.008 | 0.015 | 0.002 | 0.009 | survived |
| Woolf | sent_len_std | 0.003 | 0.004 | 0.027 | 0.126 | survived |
