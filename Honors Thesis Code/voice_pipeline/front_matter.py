"""
Front-matter trimming: find where each novel's narrative actually starts.

Why: the corpus files begin with whatever the source edition had before
chapter 1 -- title pages, tables of contents, dedications, prefaces,
publisher lines. Earlier opening tests (burst_sweep.py,
wordlen_direction_check.py) used the raw files, so the first window of many
books was partly or wholly front matter (e.g. Bleak House opens with
Dickens's Preface; Tess with a table of contents). A preface is expository
prose, which plausibly has longer words and more "of" -- the very
"opening register" those tests found. Trimming separates the two.

Rule: walk the file line by line and find the FIRST chapter-style heading
(CHAPTER I, BOOK I, PART ONE, a bare Roman numeral, a bare number or year,
PROLOGUE, ...) that is followed by real prose -- at least MIN_PROSE_WORDS
words before the next heading. Contents-page entries fail that test because
the next heading comes a line later. Up to MAX_TITLE_LINES short lines right
after the heading (chapter titles, e.g. "THE SUBTERRANEAN WAY") are skipped
too. The narrative starts at the first prose line.

A few books need a manual start (MANUAL_START below) because their first
narrative chapter has no heading the rule recognises; each is listed with
the reason. Every cut is written to output_register/front_matter_cuts.csv
with the text on both sides of it, so the cuts can be checked by eye.
"""
import re

HEADING_RE = re.compile(
    r"""^(
        (CHAPTER|Chapter|CHAP\.?|Chap\.?|BOOK|Book|PART|Part|PHASE|Phase|STAVE|Stave|VOLUME|Volume)
            [\s.]+([IVXLC]+|\d+|[A-Za-z]+)\b.*
      | [IVXLC]{1,7}\.?
      | \d{1,4}\.?
      | (PROLOGUE|Prologue|PRELUDE|Prelude|INTRODUCTORY|Introductory|OVERTURE|Overture)\b.*
    )$""", re.X)
MIN_PROSE_WORDS = 150
MAX_TITLE_LINES = 3
SEARCH_FRACTION = 0.2      # the narrative must start within the first 20% of the file

# filename -> (literal string the narrative starts with, reason)
MANUAL_START = {
    'Barclay_Postern_1911.txt': ('"But it was not your niece!', 'the rule stops on "1912" in the imprint; book uses day headings'),
    'Chesterton_Innocence_1911.txt': ('Between the silver ribbon of morning', 'story-collection title lines, no chapter heading'),
    'Dickens_Bleak_1853.txt': ('London. Michaelmas term', 'first line of ch. 1 has no terminal punctuation, so the rule took it for a chapter title'),
    'Dickens_Oliver_1839.txt': ('Among other public buildings', 'long chapter title (14 words) after the heading'),
    'Eliot_Daniel_1876.txt': ('Was she beautiful or not beautiful', 'skip the prose epigraph of ch. 1'),
    'Eliot_Felix_1866.txt': ('On the first of September, in the memorable year 1832', 'Introduction is a separate frame; ch. 1 starts after a verse epigraph'),
    'Ford_Girl_1907.txt': ('THE first thing he said when she came into the room', 'ch. 1 has no numeral heading'),
    'Hardy_Madding_1874.txt': ('When Farmer Oak smiled', 'same as Bleak House'),
    'James_Ambassadors_1903.txt': ("Strether's first question", "skip James's New York Edition preface"),
    'Kipling_Kim_1901.txt': ('He sat, in defiance of municipal orders', 'skip the verse epigraph of ch. 1'),
    'Kipling_Light_1891.txt': ("'WHAT do you think she'd do", 'skip the verse epigraph of ch. 1'),
    'Trollope_Phineas_1869.txt': ('Dr. Finn, of Killaloe, in county Clare', 'contents list uses "I." style entries'),
    'Wcollins_Woman_1860.txt': ("This is the story of what a Woman's patience", 'section I has no numeral heading'),
}


def _is_heading(line: str) -> bool:
    return bool(HEADING_RE.match(line.strip()))


def find_narrative_start(raw: str, filename: str = ''):
    """Return (char_offset, method). raw is the file text as read (newlines kept)."""
    if filename in MANUAL_START:
        needle, _ = MANUAL_START[filename]
        i = raw.find(needle)
        if i >= 0:
            return i, 'manual'
    lines = raw.split('\n')
    offsets, pos = [], 0
    for ln in lines:
        offsets.append(pos)
        pos += len(ln) + 1
    limit = len(raw) * SEARCH_FRACTION
    for i, ln in enumerate(lines):
        if offsets[i] > limit:
            break
        if not _is_heading(ln):
            continue
        # skip blank lines and up to MAX_TITLE_LINES short title lines
        j, titles = i + 1, 0
        while j < len(lines):
            s = lines[j].strip()
            if not s:
                j += 1
                continue
            if _is_heading(s):
                break
            if len(s.split()) <= 10 and titles < MAX_TITLE_LINES and not re.search(r'[a-z].*[.!?,;:]["\']?$', s):
                titles += 1
                j += 1
                continue
            break
        if j >= len(lines) or _is_heading(lines[j]):
            continue
        # count prose words until the next heading
        words, k = 0, j
        while k < len(lines) and words < MIN_PROSE_WORDS:
            if _is_heading(lines[k]):
                break
            words += len(lines[k].split())
            k += 1
        if words >= MIN_PROSE_WORDS:
            return offsets[j], 'heading'
    return 0, 'none_found'


def load_trimmed(path: str, filename: str = ''):
    """Same normalisation as windowing.load_text, applied after trimming."""
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        raw = f.read().lstrip('﻿')
    start, method = find_narrative_start(raw, filename)
    text = re.sub(r'\s+', ' ', raw[start:]).strip()
    return text, start, method, raw
