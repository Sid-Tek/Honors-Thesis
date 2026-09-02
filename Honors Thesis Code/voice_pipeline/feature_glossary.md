# Feature glossary — voice pipeline

Plain-language description of each of the 24 features in `feature_summary.md`
/ `author_feature_results.csv`. Grouped by type; original computation is in
`features.py` if you need the exact formula.

## Vocabulary / word-choice features

| Feature | What it measures |
|---|---|
| `ttr` | Type-token ratio: unique words ÷ total words in the window. Higher = more varied vocabulary, less repetition. (Note: mechanically shrinks as window size grows — see README caveat.) |
| `hapax_rate` | Share of words that appear exactly once in the window ("hapax legomena"). Higher = more one-off, unusual word choices rather than a small set of words repeated often. |
| `avg_word_len` | Average word length in characters. Higher = longer, often more Latinate/formal vocabulary. |
| `syll_per_word` | Average syllables per word. A rough proxy for vocabulary complexity/formality. |
| `pct_monosyllabic` | Share of words that are exactly one syllable. Higher = plainer, blunter word choice. |

## Specific function words

These track how often one particular common word appears, as a rate per 100
words. Function words (articles, prepositions, conjunctions, pronouns) are a
classic stylometry signal because authors use them unconsciously and
consistently, unlike topic words which change with subject matter.

| Feature | Word tracked |
|---|---|
| `fw_the` | "the" |
| `fw_and` | "and" |
| `fw_of` | "of" |
| `fw_to` | "to" |
| `fw_in` | "in" |
| `fw_that` | "that" |
| `fw_it` | "it" |
| `fw_was` | "was" |
| `fw_but` | "but" |
| `fw_as` | "as" |

## Sentence structure

| Feature | What it measures |
|---|---|
| `sent_len_mean` | Average sentence length, in words. |
| `sent_len_std` | How much sentence length varies within the window — a mix of short punchy and long winding sentences vs. a more uniform rhythm. |
| `subordination_rate` | Rate (per 100 words) of subordinating conjunctions — "because," "although," "since," "while," "if," "when," "though," "unless," "whereas," "whenever," "after," "before," "until," "once." Higher = more dependent-clause, "although X, Y" style sentences. |
| `coordination_rate` | Rate (per 100 words) of coordinating conjunctions — "and," "but," "or," "nor," "for," "yet," "so." Higher = more clauses strung together as equals ("X, and Y, and Z") rather than subordinated. |

## Punctuation

| Feature | What it measures |
|---|---|
| `comma_rate` | Commas per 100 words. |
| `semicolon_rate` | Semicolons per 100 words. |
| `dash_rate` | Dashes per 100 words. (Flagged as unstable in the corpus-wide run — rare enough that some authors' reference books have very few, inflating their z-scores. Treat results on this one cautiously.) |
| `question_rate` | Share of sentences ending in "?". |
| `exclaim_rate` | Share of sentences ending in "!". |

## Reading the summary table

For each feature, `feature_summary.md` reports, out of the 33 authors run:
how many showed that feature as a significant **burst** (spikes in the
opening pages, then settles down), how many as a significant **accumulator**
(builds gradually across the whole book, no early spike), how many as
**both**, and how many as **null** (nothing significant either way) —
plus which of burst/accumulator/both was most common among the authors
that showed *any* significant pattern (`dominant_signal`).
