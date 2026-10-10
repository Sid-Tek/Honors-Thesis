# Dialogue-share check (verification plan A4)

Run 2026-10-10. Dialogue = words inside quotation marks, per 300-word window. Quote styles: {'double': 84, 'single': 14, 'none': 2}; mean share 35% of words in books with detectable dialogue. Books with no detectable dialogue are left out of every statistic below: Galsworthy_River_1933.txt, Morris_Water_1897.txt.

## 1. Does dialogue dip in the opening?

First 1,500 words vs. words 6,000-12,000, authors as units (33 authors): mean difference -0.175 (a share of 0.10 = 10 percentage points), lower in the opening for 30 of 33 authors, sign-flip p = 0.0000.
The dip ends (fade-out fit) at **4,200 words** (95% 1,500-7,050); the combined opening register ended at 2,775 (step 3b).
If the two end points agree, "descriptive opening with little dialogue" is a good description of the register; if dialogue recovers much earlier or later, it is not the whole story.

## 2. Does dialogue start to rise where voice does?

Dialogue share starts to rise (flat-then-rising line, authors resampled) at **0 words** (95% 0-0). Voice onset (step 3): 5,700 words.
Paired gap (voice onset minus dialogue onset, same authors in each draw): +5,700 words, 95% interval +300 to +13,200.
Voice starts rising AFTER dialogue does.
(A "rise" fit is a crude description of a profile that dips and recovers; use the dip end point in section 1 alongside it.)

## 3. Which features move with dialogue?

Detrended within-book correlation with dialogue share (positive = more of the feature when there is more dialogue), averaged over books.

| feature | mean r | books r>0 | books |
|---|---|---|---|
| avg_word_len | -0.53 | 0 | 98 |
| pct_monosyllabic | +0.48 | 98 | 98 |
| syll_per_word | -0.45 | 0 | 98 |
| fw_the | -0.44 | 0 | 98 |
| fw_was | -0.37 | 2 | 98 |
| fw_of | -0.36 | 0 | 98 |
| sent_len_mean | -0.29 | 7 | 98 |
| fw_and | -0.24 | 5 | 98 |
| fw_in | -0.21 | 2 | 98 |
| dash_rate | +0.21 | 64 | 71 |
| exclaim_rate | +0.19 | 91 | 98 |
| question_rate | +0.17 | 93 | 98 |

Full table: dialogue_feature_corr.csv. Features above about |r| = 0.3 are mostly scene structure (who is talking), not voice.

## 4. Do the accumulators and the onset survive taking dialogue out?

33 authors with detectable dialogue in their test books. Accumulators among them: 69 originally; 59 after dialogue is removed from the curves; 53 in both.

| analysis | onset (words) | 95% interval | authors | accumulator pairs |
|---|---:|---:|---:|---:|
| same authors, original curves, original accumulators | 5,700 | 900-12,900 | 23 | 69 |
| dialogue removed from the curves, original accumulators | 6,900 | 900-15,900 | 23 | 69 |
| dialogue removed, accumulators re-selected on the adjusted curves | 6,900 | 900-12,060 | 22 | 59 |

By feature (the three flagged as scene structure first):

| feature | accumulators (original) | after dialogue removed | in both |
|---|---|---|---|
| dash_rate | 4 | 4 | 3 |
| question_rate | 3 | 3 | 3 |
| exclaim_rate | 2 | 2 | 1 |
| sent_len_mean | 6 | 4 | 4 |
| sent_len_std | 4 | 2 | 2 |
| fw_and | 4 | 4 | 4 |
| syll_per_word | 4 | 4 | 4 |
| comma_rate | 4 | 3 | 3 |
| fw_it | 4 | 3 | 3 |
| fw_that | 4 | 2 | 2 |
| ttr | 3 | 3 | 3 |
| hapax_rate | 3 | 3 | 3 |
| fw_as | 3 | 2 | 2 |
| coordination_rate | 3 | 2 | 2 |
| pct_monosyllabic | 3 | 4 | 3 |
| subordination_rate | 3 | 3 | 3 |
| fw_to | 3 | 2 | 2 |
| avg_word_len | 2 | 2 | 2 |
| fw_the | 2 | 3 | 1 |
| semicolon_rate | 2 | 2 | 2 |
| fw_in | 1 | 0 | 0 |
| fw_of | 1 | 1 | 0 |
| fw_was | 1 | 1 | 1 |
| fw_but | 0 | 0 | 0 |

How to read it: if the onset and most accumulators survive, dialogue is not what the onset is tracking. If many accumulators vanish, check section 3: the vanishing ones should be the dialogue-correlated features. Vanishing is ambiguous (see the top of this script): dialogue may be a confound or part of how voice shows up.
