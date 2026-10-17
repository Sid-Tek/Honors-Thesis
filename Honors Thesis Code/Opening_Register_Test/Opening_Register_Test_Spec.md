# Opening Register Test: pre-registered spec

Written 2026-10-17, before any new dataset has been chosen or analysed. Nothing below may be changed after the first run on new data. Changes made later must be listed in a "Deviations" section.

## Claim being tested

Novels open in a shared "opening register" that differs from the rest of the same book. In the original corpus (100 Victorian/Edwardian novels, 33 authors x 3 books, step 1 of the verification work) 13 of 24 features shifted in the first 1,500 words, and dialogue share was lower in the opening (about 17 points below words 6,000-12,000, lower in 30 of 33 authors).

This is a generalization test: the new dataset will NOT be late 19th / early 20th century. Both outcomes are informative. Modern novels often open in dialogue or action, so the dialogue prediction is the one most at risk. It stays as written.

## Test (same as opening_register_scan.py)

- Trim front matter automatically; narrative start = first chapter heading followed by real prose.
- Opening = first 1,500 words of the narrative (the first 5 windows of 300 words).
- Reference = a straight line fitted to the same book's windows from word 6,000 onward (window 20+), exactly as in the original. The opening never influences its own baseline.
- Per book and feature, z = (window value - baseline) / SD of the baseline residuals, so z = +1 means "one typical window-to-window SD above what the rest of this book predicts". The opening score is the mean z over the first 5 windows.
- Author = unit of analysis (average z over the author's books). Sign-flip test across authors.
- Only the 13 pre-chosen features are tested, with BH-FDR across those 13 (q < 0.05, one-sided in the predicted direction).
- Dialogue is tested separately (not counted in the 13): words inside quotation marks per window, lower in the opening than in the rest of the book. Measured as the difference in share points between the opening and the same baseline line (not a z-score; see Clarifications), same sign-flip test and threshold (p < 0.05).

## Predicted directions (frozen)

- Higher in the opening (9): sentence length, word length (letters), "the", "of", sentence-length spread, syllables per word, "in", "and", "was".
- Lower in the opening (4): one-syllable words, "to", "but", type-token ratio (MATTR).
- Dialogue share: lower in the opening.

## Verdict rule (frozen)

- **Confirmed:** at least 9 of 13 features pass in the predicted direction AND dialogue is lower in the opening.
- **Partial:** 5 to 8 features pass, or 9+ pass but dialogue does not.
- **Not found:** 4 or fewer features pass.

The other 11 original features are reported in a separate table labelled exploratory. They do not count toward the verdict.

## Inclusion rules (frozen)

- Each book at least 20,000 words after trimming.
- Each author at least 2 books.
- Author = filename text before the first underscore, or a supplied authors.csv.
- Books whose narrative start could not be found with confidence ("uncertain" trims) are excluded and listed in the report. A book can be added back by listing its first sentence in starts.csv.
- Dialogue: quote style detected per book (double vs single, straight and curly). Books with no detectable dialogue are excluded from the dialogue test only. Detected style is listed per book.
- Every exclusion and its reason is listed in the report.

## Known limitations

- Plain text loses italics, so internal thoughts or dialogue in italics are not counted as dialogue.
- Translated authors mix translator style into author style.
- Smaller corpora have less power than the original (33 authors); "not found" may mean low power rather than absence.
- Placebo check (same test at random mid-book positions) is run on the new data as calibration.

## Out of scope (for now)

Attribution accuracy, voice-onset estimation, chapter-start control. These may follow if the register is confirmed.

## Code

All new code goes in a new folder, Opening_Register_Test/, directly under Honors Thesis Code/. The original voice_pipeline scripts stay untouched. The script takes --corpus-dir and --out.

## Dataset (to fill in before the first run)

Source, era, number of authors, number of books, date chosen: TBD.

## Clarifications (2026-10-17, made before any new data was touched)

- The first version said the reference was "the rest of the same book". Reading the original code showed it is a straight-line baseline fitted from word 6,000 onward. The spec now says that. A book therefore needs at least 30 windows (about 9,000 words) to have a baseline; the 20,000-word minimum already guarantees this.
- Dialogue share first got the same z-score treatment as the features. That was superseded on 2026-10-17 (see the next item).
- Dialogue is measured in share points, not z. A dry run on a corpus that is NOT the thesis dataset (the Gutenburg_corpus folder: plays, essays, overlap with the first corpus) showed why: one author's books (Ibsen's plays, 1-3% of words in quotes) have a near-zero baseline SD, so a small difference became z = +10.9 and hid a clear effect in the other 17 authors (mean z -0.37 without him). The dialogue score is now (opening share - baseline line), in share points, which is also the unit of the original 17-point result. The test, the threshold and the verdict rule are unchanged. This change was made before the thesis dataset was chosen or analysed.
- All frozen settings and predictions now live in config.py.

## Deviations

None yet.
