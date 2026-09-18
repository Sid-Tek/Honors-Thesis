"""
Stage A (part 1): text loading, sentence parsing, and windowing.

Design choice: windows are built by accumulating whole sentences until the
target word count is reached, rather than slicing raw word lists. This keeps
sentence-length and subordination features meaningful (a window never ends
mid-sentence) at the cost of window sizes being approximate rather than exact.
"""
import re
from dataclasses import dataclass, field
from typing import List


SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
WORD_RE = re.compile(r"[A-Za-z']+")

# A hyphen standing ALONE (whitespace or a string edge on both sides), e.g.
# "Well - I don't know" or "he stopped -". Counted as a dash because most
# authors' source files in this corpus write their dashes this way instead
# of with "--" / em / en dashes (a diagnosis on Sep 13 found 27 of 33 authors
# with zero true dash characters, including Dickens, Hardy, Woolf). Hyphens
# INSIDE words ("to-day", "mother-in-law") never match, since they have a
# non-space character on at least one side. Known limit: a dash written glued
# to both neighbours ("Well-I don't know") is indistinguishable from a
# compound and is not counted, so dash_rate is a lower bound for any author
# whose source flattened dashes that way.
STANDALONE_HYPHEN_RE = re.compile(r'(?<!\S)-(?!\S)')


@dataclass
class Sentence:
    words: List[str]           # lowercase alpha tokens
    n_words: int
    commas: int
    semicolons: int
    dashes: int
    terminal: str               # '.', '!', '?', or '' if undetermined
    raw: str = field(repr=False, default="")


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    # Collapse whitespace/newlines inside paragraphs so sentence splitting
    # doesn't trip on line breaks.
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def count_dashes(raw: str) -> int:
    """True dash characters (-- , em dash, en dash) plus standalone hyphens."""
    return (raw.count('--') + raw.count('—') + raw.count('–')
            + len(STANDALONE_HYPHEN_RE.findall(raw)))


def parse_sentences(text: str) -> List[Sentence]:
    raw_sentences = SENTENCE_SPLIT_RE.split(text)
    sentences = []
    for raw in raw_sentences:
        raw = raw.strip()
        if not raw:
            continue
        words = [w.lower() for w in WORD_RE.findall(raw)]
        if not words:
            continue
        terminal = raw[-1] if raw[-1] in '.!?' else ''
        sentences.append(Sentence(
            words=words,
            n_words=len(words),
            commas=raw.count(','),
            semicolons=raw.count(';'),
            dashes=count_dashes(raw),
            terminal=terminal,
            raw=raw,
        ))
    return sentences


def local_windows(sentences: List[Sentence], window_size: int = 300) -> List[List[Sentence]]:
    """Non-overlapping windows, each holding whole sentences, closed once the
    accumulated word count reaches window_size. The final partial window is
    dropped if it has fewer than half the target words (too noisy to compare)."""
    windows = []
    current = []
    count = 0
    for s in sentences:
        current.append(s)
        count += s.n_words
        if count >= window_size:
            windows.append(current)
            current = []
            count = 0
    if current and count >= window_size / 2:
        windows.append(current)
    return windows


def cumulative_windows(sentences: List[Sentence], step: int = 300, max_windows: int = None) -> List[List[Sentence]]:
    """Windows = first `step` words, first `2*step` words, etc., each a list
    of sentences from the very start of the text."""
    local = local_windows(sentences, window_size=step)
    if max_windows is not None:
        local = local[:max_windows]
    cumulative = []
    acc = []
    for w in local:
        acc = acc + w
        cumulative.append(list(acc))
    return cumulative


def window_word_count(window: List[Sentence]) -> int:
    return sum(s.n_words for s in window)


def window_start_offset(sentences: List[Sentence], window: List[Sentence]) -> int:
    """Word offset (from text start) where this window begins -- only
    meaningful for local windows where `window` is a contiguous slice."""
    first = window[0]
    offset = 0
    for s in sentences:
        if s is first:
            return offset
        offset += s.n_words
    return offset