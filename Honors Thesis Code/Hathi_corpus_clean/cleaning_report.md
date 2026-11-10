# Cleaning report: page numbers and running heads

55 texts from `..\Hathi_corpus_trimmed` written to `..\Hathi_corpus_clean`. The originals are untouched.

* Page-number lines removed: 18,191
* Running-head lines removed: 12,332
* Words: 5,997,178 before, 5,960,429 after (0.61% removed). Per book: median 0.77%, maximum 1.76%.
* Books with no running head found: 15 (page numbers may still have been removed; see cleaning_log.csv).

No book lost more than 3% of its words.

## What was removed

removed_heads.csv lists every repeated line that was removed, per book. The most frequent ones across the corpus:

| line (as first seen) | books | lines removed |
|---|---:|---:|
| THE OLD BUNCH | 1 | 474 |
| THE CASE OF MRS. WINGATE | 1 | 466 |
| THE STORY OF DOROTHY STANFIELD | 1 | 379 |
| THE KINGS OF BEACON HILL | 1 | 371 |
| A SIGN FOR CAIN | 1 | 365 |
| THE WIND FROM NOWHERE | 1 | 349 |
| THE FIREMAKERS | 1 | 345 |
| MASQUERADE | 1 | 342 |
| THE HILLIKIN | 1 | 340 |
| TOWARD ROMANCE | 1 | 338 |
| TUESDAY NEVER COMES | 1 | 300 |
| THE WIND BLOWS WEST | 1 | 298 |
| THE HARVEST MURDER | 1 | 286 |
| THE TALL HOUSE MYSTERY | 1 | 284 |
| NO LOVELIER SPRING | 1 | 283 |

## Left in, on purpose

Mixed-case repeated edge lines on 3-5 pages only (probably chapter titles on short chapters) are NOT removed; the full list is below so you can see what stayed. If one of these is a running head, say so and it can be added to the log; it will not change the rule.

| book | line | pages |
|---|---|---:|
| Brown_Hillikin.txt | with the coarse-voiced, profane thirteen-year-olds in No. 5 | 3 |
| Brown_Hillikin.txt | Company Row who wore oversized, half-buttoned dresses | 3 |
| Brown_Hillikin.txt | when needs were especially great. | 3 |
| Corbett_The-Langworthy-family-a-novel-of | Langworthy. | 3 |
| Corbett_The-house-across-the-river.txt | Farwell. | 4 |
| Corbett_The-young-Mrs-Meigs.txt | Mrs. Meigs. | 5 |
| Corbett_The-young-Mrs-Meigs.txt | Meigs. | 3 |
| Fisher_Orphans-in-Gethsemane-a-novel-of- | Neloa | 3 |
| Fisher_The-valley-of-vision-a-novel-of-K | he said. | 3 |
| Gregg_Inspector-Higgins-hurries-being-a- | Inspector Higgins listened with much interest to the | 3 |
| Gregg_Inspector-Higgins-hurries-being-a- | Higgins grinned. | 3 |
| Johnston_Miss-Delicia-Allen.txt | Lady Annesley to Mrs. Allen. | 5 |
| Lumpkin_A-sign-for-Cain_1935.txt | asked. | 3 |
| Street_Body-unidentified.txt | PROLOGUE | 3 |
| Street_Body-unidentified.txt | asked. | 3 |
| Thayer_Dead-end-street-no-outlet.txt | APARTMENT 426 | 5 |
| Thayer_Dead-end-street-no-outlet.txt | BELLEVUE | 5 |
| Thayer_Dead-end-street-no-outlet.txt | DETECTIVE | 4 |
| Thayer_Sudden-death.txt | detective. | 3 |

## Next

Run the test on the cleaned texts with `--starts` pointing at starts_clean.csv. The as-is run uses the original folder with starts_asis.csv. Both read the first story sentence from the file itself, so the loader does not trim again.
