"""
Opening Register Test -- stage 2: window features and dialogue share.

For every INCLUDED book (stage 1, loader.py) this cuts the trimmed narrative
into 300-word windows (whole sentences) and measures, per window:
  * the 24 stylometric features of the original pipeline (the 13 tested ones
    and the 11 exploratory ones; see config.py), and
  * the share of words that sit inside quotation marks (dialogue).

The feature and dialogue definitions below are copies of the original
voice_pipeline code (features.py, windowing.py, dialogue_share_check.py) so
this folder works on its own and the originals stay untouched. test_parity.py
checks that the copies give the same numbers as the originals on real novels.

Dialogue detection (per book, from the text itself):
  double  "..." or curly double quotes
  single  '...' or curly single quotes (British style); apostrophes are told
          apart from quotes by position, with a short list of elisions
  none    fewer than 10 quote marks per 10,000 words: no dialogue detectable.
          Such books stay in the 13-feature test but are left out of the
          dialogue test.
Words inside quotes are counted with a running open/closed state across
sentences; a quote left open for more than 300 words is treated as a stray
mark and closed. Treat the share as a good RELATIVE measure (opening vs rest),
not an exact count. Plain text loses italics, so italic thoughts are not counted.

Run (from the Opening_Register_Test folder):
    python windows.py --corpus-dir path\\to\\txt_folder --out output_run
    python windows.py --corpus-dir ... --authors authors.csv --starts starts.csv
    python windows.py --corpus-dir ... --rebuild        # ignore the cache

Writes, in --out: everything loader.py writes, plus
    dialogue_books.csv    quote style and dialogue share per book
    windows_summary.md    counts, books without dialogue, books with too few windows
    window_cache.pkl      per-book window features (read by the later stages)
"""
import argparse
import hashlib
import os
import pickle
import re
import time
from bisect import bisect_left
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Dict, List

import numpy as np
import pandas as pd

import config
from loader import WORD_RE, load_corpus, write_outputs

CACHE_NAME = 'window_cache.pkl'
CACHE_VERSION = 1

# ============================================================ sentences & windows
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
# A hyphen standing ALONE (spaces or a string edge on both sides) counts as a dash.
STANDALONE_HYPHEN_RE = re.compile(r'(?<!\S)-(?!\S)')


@dataclass
class Sentence:
    words: List[str]
    n_words: int
    commas: int
    semicolons: int
    dashes: int
    terminal: str
    raw: str = field(repr=False, default='')


def count_dashes(raw: str) -> int:
    return (raw.count('--') + raw.count('—') + raw.count('–')
            + len(STANDALONE_HYPHEN_RE.findall(raw)))


def parse_sentences(text: str) -> List[Sentence]:
    out = []
    for raw in SENTENCE_SPLIT_RE.split(text):
        raw = raw.strip()
        if not raw:
            continue
        words = [w.lower() for w in WORD_RE.findall(raw)]
        if not words:
            continue
        out.append(Sentence(words=words, n_words=len(words), commas=raw.count(','),
                            semicolons=raw.count(';'), dashes=count_dashes(raw),
                            terminal=raw[-1] if raw[-1] in '.!?' else '', raw=raw))
    return out


def local_windows(sentences: List[Sentence], window_size: int = config.WINDOW_SIZE) -> List[List[Sentence]]:
    """Non-overlapping windows of whole sentences, closed once window_size words are reached.
    A final partial window is kept only if it has at least half the target words."""
    windows, current, count = [], [], 0
    for s in sentences:
        current.append(s)
        count += s.n_words
        if count >= window_size:
            windows.append(current)
            current, count = [], 0
    if current and count >= window_size / 2:
        windows.append(current)
    return windows


# ================================================================== the features
FUNCTION_WORDS = ['the', 'and', 'of', 'to', 'in', 'that', 'it', 'was', 'but', 'as']
SUBORDINATORS = {'because', 'although', 'since', 'while', 'if', 'when', 'though', 'unless',
                 'whereas', 'whenever', 'after', 'before', 'until', 'once'}
COORDINATORS = {'and', 'but', 'or', 'nor', 'for', 'yet', 'so'}
VOWEL_GROUP_RE = re.compile(r'[aeiouy]+')

FEATURE_NAMES = (
    ['sent_len_mean', 'sent_len_std', 'ttr', 'hapax_rate', 'avg_word_len',
     'comma_rate', 'semicolon_rate', 'dash_rate', 'question_rate', 'exclaim_rate',
     'subordination_rate', 'coordination_rate', 'syll_per_word', 'pct_monosyllabic']
    + [f'fw_{fw}' for fw in FUNCTION_WORDS]
)


def count_syllables(word: str) -> int:
    word = word.lower()
    n = len(VOWEL_GROUP_RE.findall(word))
    if word.endswith('e') and not word.endswith('le') and n > 1:
        n -= 1
    return max(n, 1)


class SlidingLexicalStats:
    """Moving-average type-token ratio and hapax rate over a sliding sub-window of w words
    (Covington & McFall's MATTR), so the value does not depend on the window's length."""

    def __init__(self, w: int = config.MATTR_WINDOW):
        self.w = w
        self.buffer = deque()
        self.counts = Counter()
        self.ttr_sum = 0.0
        self.hapax_sum = 0.0
        self.n_measurements = 0

    def add_word(self, word: str):
        self.buffer.append(word)
        self.counts[word] += 1
        if len(self.buffer) > self.w:
            old = self.buffer.popleft()
            self.counts[old] -= 1
            if self.counts[old] == 0:
                del self.counts[old]
        if len(self.buffer) == self.w:
            n_hapax = sum(1 for c in self.counts.values() if c == 1)
            self.ttr_sum += len(self.counts) / self.w
            self.hapax_sum += n_hapax / self.w
            self.n_measurements += 1

    def snapshot(self):
        if self.n_measurements > 0:
            return self.ttr_sum / self.n_measurements, self.hapax_sum / self.n_measurements
        n = len(self.buffer)
        if n == 0:
            return 0.0, 0.0
        c = Counter(self.buffer)
        return len(c) / n, sum(1 for v in c.values() if v == 1) / n


def extract_features(window: List[Sentence]) -> Dict[str, float]:
    words = [w for s in window for w in s.words]
    n_words = len(words)
    if n_words == 0:
        return {}
    lengths = [s.n_words for s in window]
    n_sent = len(window)
    counts = Counter(words)

    f = {}
    f['sent_len_mean'] = mean(lengths)
    f['sent_len_std'] = pstdev(lengths) if len(lengths) > 1 else 0.0
    lex = SlidingLexicalStats(config.MATTR_WINDOW)
    for w in words:
        lex.add_word(w)
    f['ttr'], f['hapax_rate'] = lex.snapshot()
    f['avg_word_len'] = mean(len(w) for w in words)
    f['comma_rate'] = sum(s.commas for s in window) / n_words * 100
    f['semicolon_rate'] = sum(s.semicolons for s in window) / n_words * 100
    f['dash_rate'] = sum(s.dashes for s in window) / n_words * 100
    f['question_rate'] = sum(1 for s in window if s.terminal == '?') / n_sent
    f['exclaim_rate'] = sum(1 for s in window if s.terminal == '!') / n_sent
    f['subordination_rate'] = sum(1 for w in words if w in SUBORDINATORS) / n_words * 100
    f['coordination_rate'] = sum(1 for w in words if w in COORDINATORS) / n_words * 100
    syll = [count_syllables(w) for w in words]
    f['syll_per_word'] = mean(syll)
    f['pct_monosyllabic'] = sum(1 for s_ in syll if s_ == 1) / n_words
    for fw in FUNCTION_WORDS:
        f[f'fw_{fw}'] = counts.get(fw, 0) / n_words * 100
    return f


# ================================================================ dialogue share
DQ_RE = re.compile('["“”]')
SQ_RE = re.compile("['‘’]")
ELISION = r"(?:tis|twas|twere|twill|twould|em|cause|til|till|ud|ave|ere|alf|arry|ow|e|er|im)"
ELISION_RE = re.compile(ELISION + r'\b', re.I)
SQ_OPEN_RE = re.compile(r"(?<![\w'’])'(?=[A-Za-z])(?!" + ELISION + r'\b)', re.I)
OPENERS = '([{—-'
CLOSERS = '.,;:!?)]}—-'
OPENERS_SQ = OPENERS + '"“'


def detect_style(text: str):
    """('double' | 'single' | 'none', n_double_marks, n_single_openers)."""
    n_words = max(len(text.split()), 1)
    n_dq = len(DQ_RE.findall(text))
    n_sq = text.count('‘') + len(SQ_OPEN_RE.findall(text))
    if n_dq / n_words * 1e4 >= config.MIN_QUOTE_MARKS_PER_10K and n_dq >= n_sq:
        return 'double', n_dq, n_sq
    if n_sq / n_words * 1e4 >= config.MIN_QUOTE_MARKS_PER_10K:
        return 'single', n_dq, n_sq
    return 'none', n_dq, n_sq


def double_events(raw: str):
    ev, L = [], len(raw)
    for m in DQ_RE.finditer(raw):
        i, ch = m.start(), m.group()
        if ch == '“':
            ev.append((i, 'open'))
        elif ch == '”':
            ev.append((i, 'close'))
        else:                                   # straight quote: decide from the neighbours, else toggle
            prev = raw[i - 1] if i > 0 else ' '
            nxt = raw[i + 1] if i + 1 < L else ' '
            if (prev.isspace() or prev in OPENERS) and not nxt.isspace():
                ev.append((i, 'open'))
            elif not prev.isspace() and (nxt.isspace() or nxt in CLOSERS):
                ev.append((i, 'close'))
            else:
                ev.append((i, 'toggle'))
    return ev


def single_events(raw: str):
    ev, L = [], len(raw)
    for m in SQ_RE.finditer(raw):
        i, ch = m.start(), m.group()
        prev = raw[i - 1] if i > 0 else ' '
        nxt = raw[i + 1] if i + 1 < L else ' '
        if ch == '‘':
            ev.append((i, 'open'))
        elif nxt.isalnum():                     # inside a word (don't) or the start of one ('Hello)
            if ch == "'" and (prev.isspace() or prev in OPENERS_SQ) and not ELISION_RE.match(raw, i + 1):
                ev.append((i, 'open'))
        elif not prev.isspace():                # follows a word or punctuation, then a space/punctuation
            ev.append((i, 'close'))
    return ev


def dialogue_words(sents: List[Sentence], style: str) -> List[int]:
    """Words inside quotes, per sentence (same tokens as Sentence.n_words)."""
    if style == 'none':
        return [0] * len(sents)
    get_events = double_events if style == 'double' else single_events
    state, open_words, out = False, 0, []
    for s in sents:
        raw = s.raw
        starts = [m.start() for m in WORD_RE.finditer(raw)]
        count_in, last = 0, 0
        for idx, kind in get_events(raw):
            n_between = bisect_left(starts, idx) - bisect_left(starts, last)
            if state:
                count_in += n_between
            state = True if kind == 'open' else (False if kind == 'close' else not state)
            last = idx
        if state:
            count_in += len(starts) - bisect_left(starts, last)
        out.append(count_in)
        if state:
            open_words += s.n_words
            if open_words > config.MAX_OPEN_QUOTE_WORDS:    # stray unmatched mark: give up on this quote
                state, open_words = False, 0
        else:
            open_words = 0
    return out


# ============================================================== one book / all books
def process_book(args):
    """(filename, trimmed text) -> per-window feature matrix and dialogue shares."""
    fn, text = args
    sents = parse_sentences(text)
    wins = local_windows(sents, config.WINDOW_SIZE)
    X = np.array([[extract_features(w)[f] for f in FEATURE_NAMES] for w in wins], float)
    style, n_dq, n_sq = detect_style(text)
    inw = dialogue_words(sents, style)
    shares, pos = [], 0
    for w in wins:
        k = len(w)
        tot = sum(s.n_words for s in sents[pos:pos + k])
        shares.append(sum(inw[pos:pos + k]) / max(tot, 1))
        pos += k
    return fn, {'features': X, 'dialogue': np.array(shares, float), 'style': style,
                'n_double': n_dq, 'n_single_open': n_sq,
                'n_words': sum(s.n_words for s in sents), 'n_windows': len(wins)}


def _settings_key():
    return (CACHE_VERSION, config.WINDOW_SIZE, config.MATTR_WINDOW,
            config.MIN_QUOTE_MARKS_PER_10K, config.MAX_OPEN_QUOTE_WORDS)


def build_windows(books, out_dir: str, rebuild: bool = False, workers=None) -> Dict[str, dict]:
    """Window features for every book in `books` (loader.Book objects with .text). Cached by text hash."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, CACHE_NAME)
    cache = {}
    if os.path.exists(path) and not rebuild:
        with open(path, 'rb') as f:
            d = pickle.load(f)
        if d.get('settings') == _settings_key():
            cache = d['books']
    hashes = {b.filename: hashlib.md5(b.text.encode('utf-8', 'ignore')).hexdigest() for b in books}
    todo = [b for b in books if b.filename not in cache or cache[b.filename][0] != hashes[b.filename]]
    print(f'Window features: {len(books) - len(todo)} cached, {len(todo)} to compute.', flush=True)
    if todo:
        t0 = time.time()
        jobs = [(b.filename, b.text) for b in todo]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, (fn, res) in enumerate(ex.map(process_book, jobs), 1):
                cache[fn] = (hashes[fn], res)
                if i % 10 == 0 or i == len(jobs):
                    print(f'  [{i}/{len(jobs)}] {time.time() - t0:.0f}s', flush=True)
        with open(path, 'wb') as f:
            pickle.dump({'settings': _settings_key(), 'books': cache}, f)
    return {b.filename: cache[b.filename][1] for b in books}


def write_window_outputs(books, results: Dict[str, dict], out_dir: str) -> None:
    rows = []
    for b in books:
        r = results[b.filename]
        d = r['dialogue']
        has = r['style'] != 'none'
        k = config.OPENING_WINDOWS
        rows.append({
            'filename': b.filename, 'author': b.author, 'style': r['style'],
            'n_double_marks': r['n_double'], 'n_single_openers': r['n_single_open'],
            'n_windows': r['n_windows'], 'usable': r['n_windows'] >= config.MIN_WINDOWS,
            'dialogue_mean_all': float(d.mean()) if has else np.nan,
            'dialogue_mean_opening': float(d[:k].mean()) if has else np.nan,
            'dialogue_mean_from_baseline': float(d[config.BASELINE_START_WINDOW:].mean()) if has else np.nan,
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'dialogue_books.csv'), index=False)

    styles = df['style'].value_counts().to_dict()
    nodlg = df.loc[df['style'] == 'none', 'filename'].tolist()
    few = df.loc[~df['usable'], 'filename'].tolist()
    L = ['# Window features summary', '',
         f'{len(df)} included books, {df["n_windows"].sum():,} windows of about {config.WINDOW_SIZE} words.',
         f'Quote styles: {styles}. Mean dialogue share in books with dialogue: '
         f'{df.loc[df["style"] != "none", "dialogue_mean_all"].mean():.0%}.' if (df['style'] != 'none').any()
         else f'Quote styles: {styles}. No book has detectable dialogue.', '']
    if nodlg:
        L += ['No detectable dialogue (left out of the dialogue test only): ' + ', '.join(nodlg), '']
    if few:
        L += [f'Fewer than {config.MIN_WINDOWS} windows (too short for a baseline; lower --min-words was used?): '
              + ', '.join(few), '']
    L += ['Check dialogue_books.csv: a book whose style looks wrong (for example "none" in a book you know '
          'has dialogue) is the thing to catch before trusting the dialogue result.', '']
    with open(os.path.join(out_dir, 'windows_summary.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('Run (from')[0],
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--corpus-dir', required=True, help='folder of .txt novels')
    ap.add_argument('--out', default='output_run', help='output folder (default output_run)')
    ap.add_argument('--authors', default=None, help='optional authors.csv (filename,author)')
    ap.add_argument('--starts', default=None, help='optional starts.csv (filename,start_text)')
    ap.add_argument('--min-words', type=int, default=config.MIN_WORDS)
    ap.add_argument('--min-books', type=int, default=config.MIN_BOOKS)
    ap.add_argument('--workers', type=int, default=None, help='processes (default: all cores)')
    ap.add_argument('--rebuild', action='store_true', help='ignore the cache')
    args = ap.parse_args()

    t0 = time.time()
    books = load_corpus(args.corpus_dir, args.starts, args.authors, args.min_words, args.min_books)
    write_outputs(books, args.out, args)
    included = [b for b in books if b.included]
    print(f'Stage 1: {len(books)} files, {len(included)} included '
          f'({len({b.author for b in included})} authors), {len(books) - len(included)} excluded.', flush=True)
    if not included:
        raise SystemExit('No books passed the inclusion rules; see exclusions.csv.')
    results = build_windows(included, args.out, args.rebuild, args.workers)
    write_window_outputs(included, results, args.out)
    print(f'Wrote dialogue_books.csv, windows_summary.md and {CACHE_NAME} to {args.out}/ '
          f'({time.time() - t0:.0f}s total).')


if __name__ == '__main__':
    main()
