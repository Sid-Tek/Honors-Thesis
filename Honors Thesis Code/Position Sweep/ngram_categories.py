"""
Character n-gram category extractor, reimplementing the 10 disjoint
category definitions from:

  Sapkota, Bethard, Montes-y-Gomez, Solorio (2015).
  "Not All Character N-grams Are Created Equal: A Study in Authorship
  Attribution." NAACL 2015.

No code was released alongside the paper; this is a from-spec
reimplementation of the category *definitions* (Section 2), which the
paper gives precisely enough to reproduce without guessing at the core
rules. A few genuinely unspecified tie-break edge cases are handled with
a documented, reasonable choice -- see NOTES below.

Categories (grouped by the paper's three "super categories"):

  affix : prefix, suffix, space-prefix, space-suffix
  word  : whole-word, mid-word, multi-word
  punct : beg-punct, mid-punct, end-punct

NOTES / assumptions where the paper doesn't spell out every tie-break:
  - Punctuation character set: Python's string.punctuation. The paper
    never gives an exact list.
  - Priority order when an n-gram could match more than one rule:
        1. mid-punct   (interior punctuation -- checked first per the
                         paper's own worked example: 'e, ' -> mid-punct,
                         NOT space-suffix, even though it also ends in a
                         space. This confirms punctuation is checked
                         before spaces.)
        2. beg-punct
        3. end-punct
        4. space-prefix
        5. space-suffix
        6. multi-word  (interior space)
        7. whole-word / prefix / suffix / mid-word (pure alphabetic,
           decided by which word-boundaries the n-gram touches)
    This exact order isn't spelled out for every collision the paper
    doesn't discuss (e.g. an n-gram that both starts AND ends with
    punctuation), but it keeps all categories disjoint and matches every
    worked example the paper gives.
  - The paper chose n=3 after preliminary experiments found it worked
    best on their data (news text). n is a parameter here, defaulting
    to 3, so you can revisit that choice on your own corpus later --
    see the earlier discussion on why arbitrary n has real downsides
    for short-window emergence testing.
"""

import string
from collections import Counter, defaultdict

PUNCTUATION = set(string.punctuation)

CATEGORIES = [
    "prefix", "suffix", "space-prefix", "space-suffix",
    "whole-word", "mid-word", "multi-word",
    "beg-punct", "mid-punct", "end-punct",
]


def _is_punct(ch):
    return ch in PUNCTUATION


def categorize_ngram(text, i, n):
    """
    Classify the n-gram text[i:i+n] into exactly one of the 10 categories
    (or 'other' for the rare non-alphabetic, non-space, non-punctuation
    character -- e.g. digits -- which the paper's scheme doesn't cover).

    `text` is the full raw document, so we can look just outside the
    n-gram itself to find word/space/punctuation boundaries.
    """
    s = text[i:i + n]
    middle = s[1:-1]  # characters strictly between the first and last

    # --- punctuation categories: checked first, regardless of spaces ---
    if any(_is_punct(c) for c in middle):
        return "mid-punct"
    if _is_punct(s[0]):
        return "beg-punct"
    if _is_punct(s[-1]):
        return "end-punct"

    # --- space categories: checked next ---
    if s[0] == " ":
        return "space-prefix"
    if s[-1] == " ":
        return "space-suffix"
    if " " in middle:
        return "multi-word"

    # --- pure alphabetic n-gram: decide by word boundaries ---
    if not s.isalpha():
        return "other"

    before_is_letter = (i > 0) and text[i - 1].isalpha()
    after_is_letter = (i + n < len(text)) and text[i + n].isalpha()

    if not before_is_letter and not after_is_letter:
        return "whole-word"
    if not before_is_letter and after_is_letter:
        return "prefix"
    if before_is_letter and not after_is_letter:
        return "suffix"
    return "mid-word"


def extract_ngram_categories(text, n=3):
    """
    Slides an n-character window across the ENTIRE raw text (letters,
    spaces, and punctuation all included -- nothing is stripped first)
    and classifies every n-gram into one of the 10 categories.

    Returns a dict: {category_name: Counter({ngram: count, ...})}
    All 10 categories are always present in the returned dict, even if
    empty, so downstream code doesn't need to special-case missing keys.
    """
    counts = defaultdict(Counter)
    for i in range(len(text) - n + 1):
        category = categorize_ngram(text, i, n)
        if category == "other":
            continue
        counts[category][text[i:i + n]] += 1

    for cat in CATEGORIES:
        counts.setdefault(cat, Counter())
    return dict(counts)


if __name__ == "__main__":
    # The paper's own worked example (Table 1/Table 2), n=3
    example = "The actors wanted to see if the pact seemed like an old-fashioned one."
    result = extract_ngram_categories(example, n=3)

    for cat in CATEGORIES:
        top = result[cat].most_common(8)
        print(f"{cat:14s} {top}")