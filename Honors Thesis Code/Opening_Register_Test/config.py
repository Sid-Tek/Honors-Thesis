"""
Opening Register Test -- FROZEN settings and predictions.

Written 2026-10-17, before any new dataset was chosen or analysed. This file
is the code form of the pre-registered spec ("Opening Register Test -
Pre-registered Spec (2026-10-17)" in the project docs). Do not edit the
predictions, the thresholds or the measurement settings after the first run on
new data. If something has to change, record it in the spec's "Deviations"
section as well.

The measurement settings (300-word windows, first 1,500 words as the opening,
baseline from word 6,000 onward) are the ones used by the original
voice_pipeline/opening_register_scan.py, so results are comparable.
"""

# ---------------------------------------------------------------- measurement
WINDOW_SIZE = 300            # words per window (whole sentences, as in the original pipeline)
OPENING_WINDOWS = 5          # first 5 windows = about the first 1,500 words
BASELINE_START_WINDOW = 20   # "rest of book" baseline = straight line fitted to windows 20+ (word 6,000+)
MIN_WINDOWS = 30             # a book needs this many windows to have a baseline (30 windows = 9,000 words)
MATTR_WINDOW = 50            # sub-window for the moving-average type-token ratio
MIN_QUOTE_MARKS_PER_10K = 10 # fewer quote marks than this -> "no dialogue detectable"
MAX_OPEN_QUOTE_WORDS = 300   # a quote open longer than this is a stray mark and is closed by force

# ------------------------------------------------------------ inclusion rules
MIN_WORDS = 20000            # words per book after trimming
MIN_BOOKS = 2                # included books per author

# ------------------------------------------------------- the 13 tested features
# (feature key as in the original pipeline) -> predicted direction in the opening
TESTED = {
    'sent_len_mean':    'higher',   # sentence length
    'avg_word_len':     'higher',   # word length (letters)
    'fw_the':           'higher',   # "the"
    'fw_of':            'higher',   # "of"
    'sent_len_std':     'higher',   # sentence-length spread
    'syll_per_word':    'higher',   # syllables per word
    'fw_in':            'higher',   # "in"
    'fw_and':           'higher',   # "and"
    'fw_was':           'higher',   # "was"
    'pct_monosyllabic': 'lower',    # one-syllable words
    'fw_to':            'lower',    # "to"
    'fw_but':           'lower',    # "but"
    'ttr':              'lower',    # type-token ratio (moving average, MATTR)
}
DIALOGUE_PREDICTION = 'lower'       # share of words inside quotation marks

# The other 11 original features: reported separately, exploratory, NOT part of the verdict.
EXPLORATORY = ['hapax_rate', 'comma_rate', 'semicolon_rate', 'dash_rate', 'question_rate',
               'exclaim_rate', 'subordination_rate', 'coordination_rate',
               'fw_that', 'fw_it', 'fw_as']

# ---------------------------------------------------------------- decision rule
ALPHA = 0.05                 # BH-FDR level across the 13 tested features (one-sided, predicted direction)
CONFIRMED_MIN_PASS = 9       # confirmed: >= 9 of 13 pass AND dialogue is lower
PARTIAL_MIN_PASS = 5         # partial: 5-8 pass (or 9+ pass but dialogue does not); below this: not found

LABELS = {
    'sent_len_mean': 'sentence length', 'sent_len_std': 'sentence-length spread',
    'ttr': 'type-token ratio (MATTR)', 'hapax_rate': 'hapax rate (MA)',
    'avg_word_len': 'word length (letters)', 'comma_rate': 'commas',
    'semicolon_rate': 'semicolons', 'dash_rate': 'dashes',
    'question_rate': 'questions', 'exclaim_rate': 'exclamations',
    'subordination_rate': 'subordinators', 'coordination_rate': 'coordinators',
    'syll_per_word': 'syllables per word', 'pct_monosyllabic': 'one-syllable words',
    'fw_the': '"the"', 'fw_and': '"and"', 'fw_of': '"of"', 'fw_to': '"to"',
    'fw_in': '"in"', 'fw_that': '"that"', 'fw_it': '"it"', 'fw_was': '"was"',
    'fw_but': '"but"', 'fw_as': '"as"', 'dialogue': 'dialogue share',
}

assert len(TESTED) == 13 and len(EXPLORATORY) == 11

# --------------------------------------------------- statistics (frozen procedure)
N_PERM = 20000               # random sign flips when an exact test is too large
EXACT_MAX_AUTHORS = 16       # with this many authors or fewer, ALL 2^n sign flips are enumerated (exact p)
N_BOOT = 2000                # author resamples for the 95% intervals
N_PLACEBO = 50               # placebo draws (same test at random mid-book "openings")
SEED = 20261017
MIN_AUTHORS_WARN = 10        # fewer authors than this: the report warns that power is low
