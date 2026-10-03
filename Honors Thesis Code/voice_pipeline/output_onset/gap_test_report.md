# Gap test: opening register end vs. voice onset (step 3b)

Run 2026-10-03. Paired bootstrap: 2000 draws of 33 authors; register end and voice onset computed from the same draw each time (2000 valid draws). Voice onset uses the 23 authors with accumulators.

| quantity | estimate | 95% interval |
|---|---:|---:|
| register end (combined) | 2,775 words | 825 to 5,700 |
| voice onset (corrected) | 5,700 words | 900 to 14,100 |
| gap = onset - register end | 2,925 words | -3,075 to 12,002 |

Share of draws where voice starts after the register ends: 71%.

**No evidence of a gap: the interval includes 0, so voice may start right as the register ends -- the data cannot separate the two.**

Register features combined (13): sent_len_mean, sent_len_std, ttr, avg_word_len, syll_per_word, pct_monosyllabic, fw_the, fw_and, fw_of, fw_to, fw_in, fw_was, fw_but.
