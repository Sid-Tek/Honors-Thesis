# Book-by-book burst map

Every test book (66 books: 33 authors × 2) was checked on every feature (24). A book "bursts" on a feature when its opening ~600 words rank in the top 5% of every same-length stretch of that book (detrended; see `stats_tests.opening_rank_p`).

## Headline numbers

- Book-level bursts: **117** of 1560 book × feature checks (~78 expected by chance).
- Shared by both of an author's books: **9** author × feature pairs. If books were paired at random across authors, the average would be 5.5 (95th percentile 9); p = 0.0995.
- Authors with at least one shared burst: **7 of 33**. Authors with no burst in either book: 4.
- Features that burst in more books than chance across the corpus (q < 0.05): **4** (pct_monosyllabic, fw_of, avg_word_len, exclaim_rate).

## By feature: does any feature burst consistently?

| feature | books bursting (of ~66; ~3 by chance) | q (more than chance) | authors: both books | authors: one book |
|---|---:|---:|---|---|
| pct_monosyllabic | 11 | 0.005 | Trollope, Woolf | Braddon, Corelli, Ford, James, Meredith, Ward, Wcollins |
| fw_of | 11 | 0.005 | Thackeray | Blackmore, Chesterton, Corelli, Forster, Galsworthy, Lee, Morris, Wcollins, Woolf |
| avg_word_len | 10 | 0.013 | Trollope, Woolf | Braddon, Corelli, James, Meredith, Ward, Wcollins |
| exclaim_rate | 9 | 0.033 | – | Blackmore, Eliot, Ford, Gissing, James, Lawrence, Lee, Stevenson, Thackeray |
| syll_per_word | 8 | 0.081 | – | Burnett, Chesterton, Dickens, James, Trollope, Ward, Wcollins, Woolf |
| semicolon_rate | 7 | 0.186 | Ward | Burnett, Gissing, Lytton, Wcollins, Woolf |
| subordination_rate | 6 | 0.298 | Braddon | Bennet, Corelli, Lawrence, Trollope |
| comma_rate | 6 | 0.298 | – | Barclay, Bennet, Conrad, Gaskell, Lee, Trollope |
| fw_in | 6 | 0.298 | – | Conrad, Gaskell, Hardy, Lee, Thackeray, Woolf |
| fw_it | 5 | 0.468 | – | Braddon, Meredith, Stevenson, Thackeray, Trollope |
| question_rate | 5 | 0.468 | – | Blackmore, Burnett, Corelli, Eliot, Meredith |
| ttr | 5 | 0.468 | – | Braddon, Burnett, Thackeray, Wcollins, Woolf |
| fw_to | 4 | 0.778 | – | Dickens, Galsworthy, Hardy, Ward |
| fw_was | 3 | 0.885 | Burnett | Gaskell |
| fw_and | 3 | 0.885 | – | Dickens, Hardy, Stevenson |
| fw_that | 3 | 0.885 | – | Corelli, Gissing, Lawrence |
| sent_len_std | 2 | 0.885 | Blackmore | – |
| coordination_rate | 2 | 0.885 | – | Dickens, Hardy |
| dash_rate | 2 | 0.885 | – | Haggard, Lee |
| fw_as | 2 | 0.885 | – | Haggard, Lee |
| fw_but | 2 | 0.885 | – | Burnett, Lee |
| fw_the | 2 | 0.885 | – | Bennet, Thackeray |
| sent_len_mean | 2 | 0.885 | – | Lee, Thackeray |
| hapax_rate | 1 | 0.966 | – | Woolf |

## By author: do the two books agree?

"Expected" = shared bursts if the two books picked their burst features independently. p = chance of sharing at least this many.

| author | book A | book B | bursts A | bursts B | shared | expected | p | shared features | only A | only B |
|---|---|---|---:|---:|---:|---:|---:|---|---|---|
| Woolf | Lighthouse (1927) | Years (1937) | 4 | 6 | 2 | 1.04 | 0.270 | avg_word_len, pct_monosyllabic | hapax_rate, ttr | fw_in, fw_of, semicolon_rate, syll_per_word |
| Trollope | Phineas (1869) | Angel (1881) | 3 | 5 | 2 | 0.65 | 0.107 | avg_word_len, pct_monosyllabic | subordination_rate | comma_rate, fw_it, syll_per_word |
| Braddon | Quest (1871) | Phantom (1883) | 4 | 2 | 1 | 0.35 | 0.324 | subordination_rate | avg_word_len, fw_it, pct_monosyllabic | ttr |
| Burnett | Princess (1905) | Garden (1911) | 4 | 3 | 1 | 0.52 | 0.453 | fw_was | fw_but, syll_per_word, ttr | question_rate, semicolon_rate |
| Thackeray | Esmond (1852) | Virginians (1859) | 3 | 5 | 1 | 0.62 | 0.521 | fw_of | exclaim_rate, fw_it | fw_in, fw_the, sent_len_mean, ttr |
| Blackmore | Erema (1877) | Springhaven (1887) | 1 | 4 | 1 | 0.17 | 0.174 | sent_len_std | – | exclaim_rate, fw_of, question_rate |
| Ward | Ashe (1905) | Harvest (1920) | 1 | 5 | 1 | 0.21 | 0.208 | semicolon_rate | – | avg_word_len, fw_to, pct_monosyllabic, syll_per_word |
| Hardy | Tess (1891) | Jude (1895) | 4 | 0 | 0 | 0.00 | 1.000 | – | coordination_rate, fw_and, fw_in, fw_to | – |
| James | Tragic (1890) | Ambassadors (1903) | 4 | 0 | 0 | 0.00 | 1.000 | – | avg_word_len, exclaim_rate, pct_monosyllabic, syll_per_word | – |
| Dickens | Bleak (1853) | Expectations (1861) | 3 | 1 | 0 | 0.12 | 1.000 | – | coordination_rate, fw_and, syll_per_word | fw_to |
| Lee | Brown (1884) | Penelope (1903) | 3 | 5 | 0 | 0.62 | 1.000 | – | comma_rate, exclaim_rate, fw_as | dash_rate, fw_but, fw_in, fw_of, sent_len_mean |
| Meredith | Richmond (1871) | Marriage (1895) | 3 | 1 | 0 | 0.13 | 1.000 | – | avg_word_len, pct_monosyllabic, question_rate | fw_it |
| Wcollins | Woman (1860) | Legacy (1889) | 3 | 3 | 0 | 0.38 | 1.000 | – | fw_of, semicolon_rate, ttr | avg_word_len, pct_monosyllabic, syll_per_word |
| Bennet | Helen (1910) | Imperial (1930) | 2 | 1 | 0 | 0.08 | 1.000 | – | comma_rate, fw_the | subordination_rate |
| Conrad | Nostromo (1904) | Rover (1923) | 2 | 0 | 0 | 0.00 | 1.000 | – | comma_rate, fw_in | – |
| Ford | Soldier (1915) | Post (1926) | 2 | 0 | 0 | 0.00 | 1.000 | – | exclaim_rate, pct_monosyllabic | – |
| Gissing | Women (1893) | Warburton (1903) | 2 | 1 | 0 | 0.09 | 1.000 | – | exclaim_rate, semicolon_rate | fw_that |
| Lawrence | Women (1920) | Serpent (1926) | 2 | 1 | 0 | 0.09 | 1.000 | – | fw_that, subordination_rate | exclaim_rate |
| Corelli | Satan (1895) | Innocent (1914) | 1 | 5 | 0 | 0.21 | 1.000 | – | question_rate | avg_word_len, fw_of, fw_that, pct_monosyllabic, subordination_rate |
| Galsworthy | Saints (1919) | River (1933) | 1 | 1 | 0 | 0.04 | 1.000 | – | fw_to | fw_of |
| Gaskell | Lovers (1863) | Wives (1865) | 1 | 2 | 0 | 0.09 | 1.000 | – | comma_rate | fw_in, fw_was |
| Morris | Wood (1894) | Water (1897) | 1 | 0 | 0 | 0.00 | 1.000 | – | fw_of | – |
| Barclay | Postern (1911) | Ladies (1917) | 0 | 1 | 0 | 0.00 | 1.000 | – | – | comma_rate |
| Cbronte | Shirley (1849) | Villette (1853) | 0 | 0 | 0 | 0.00 | 1.000 | – | – | – |
| Chesterton | Thursday (1908) | Innocence (1911) | 0 | 2 | 0 | 0.00 | 1.000 | – | – | fw_of, syll_per_word |
| Doyle | Hound (1902) | Lost (1912) | 0 | 0 | 0 | 0.00 | 1.000 | – | – | – |
| Eliot | Felix (1866) | Daniel (1876) | 0 | 2 | 0 | 0.00 | 1.000 | – | – | exclaim_rate, question_rate |
| Forster | View (1908) | Howards (1910) | 0 | 1 | 0 | 0.00 | 1.000 | – | – | fw_of |
| Haggard | She (1887) | Sheallan (1921) | 0 | 2 | 0 | 0.00 | 1.000 | – | – | dash_rate, fw_as |
| Kipling | Captains (1896) | Kim (1901) | 0 | 0 | 0 | 0.00 | 1.000 | – | – | – |
| Lytton | What (1858) | Kenelm (1873) | 0 | 1 | 0 | 0.00 | 1.000 | – | – | semicolon_rate |
| Schreiner | Trooper (1897) | Undine (1929) | 0 | 0 | 0 | 0.00 | 1.000 | – | – | – |
| Stevenson | Arrow (1888) | Catriona (1893) | 0 | 3 | 0 | 0.00 | 1.000 | – | – | exclaim_rate, fw_and, fw_it |

Files: `book_feature_bursts.csv` (every book × feature, with rank p and percentile, so a different cut-off can be applied), `author_pairs.csv`, `feature_consistency.csv`, `burst_map.png`.
