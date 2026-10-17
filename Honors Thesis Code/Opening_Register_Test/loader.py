"""
Opening Register Test -- stage 1: load a folder of plain-text novels.

Works on ANY folder of .txt files. It does not depend on the original
voice_pipeline scripts (those stay untouched); the front-matter rule here is
adapted from voice_pipeline/front_matter.py, minus its hand-typed table for
the first corpus.

What it does
------------
1. Finds every .txt file in --corpus-dir (not recursive).
2. Decides each book's AUTHOR: the filename text before the first underscore
   ("Austen_Emma.txt" -> "Austen"), unless authors.csv gives one.
3. Removes Project Gutenberg boilerplate if present ("*** START OF ..." /
   "*** END OF ..." markers).
4. Trims front matter: the narrative starts at the first chapter-style heading
   that is followed by real prose (contents pages fail that test). A book can
   be fixed by hand by listing its first sentence in starts.csv.
5. Flags UNCERTAIN starts (no heading found, start suspiciously late, or a
   starts.csv sentence that is not in the file). Uncertain books are excluded
   and listed; they are never silently included.
6. Applies the inclusion rules from the spec: at least --min-words words after
   trimming, and at least --min-books included books per author.

Files read
----------
  authors.csv  (optional)  columns: filename,author
  starts.csv   (optional)  columns: filename,start_text
                           start_text = the first sentence (or its first
                           ~10 words) of chapter 1; whitespace may differ.

Files written (in --out)
------------------------
  manifest.csv    one row per INCLUDED book
  exclusions.csv  one row per EXCLUDED book, with the reason
  cuts.csv        one row per book: how it was trimmed, with the text just
                  before and just after the cut so you can check it by eye
  loader_summary.md  counts and the exclusion list

Run (from the Opening_Register_Test folder):
    python loader.py --corpus-dir path\\to\\txt_folder --out output_loader
    python loader.py --corpus-dir path\\to\\txt_folder --authors authors.csv --starts starts.csv

Later stages import load_corpus() from this file.
"""
import argparse
import csv
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

WORD_RE = re.compile(r"[A-Za-z']+")          # same word rule as the original pipeline

HEADING_RE = re.compile(
    r"""^(
        (CHAPTER|Chapter|CHAP\.?|Chap\.?|BOOK|Book|PART|Part|PHASE|Phase|STAVE|Stave|VOLUME|Volume)
            [\s.]+([IVXLC]+|\d+|[A-Za-z]+)\b.*
      | [IVXLC]{1,7}\.?
      | \d{1,4}\.?
      | (PROLOGUE|Prologue|PRELUDE|Prelude|INTRODUCTORY|Introductory|OVERTURE|Overture)\b.*
    )$""", re.X)
MIN_PROSE_WORDS = 150        # real prose needed after a heading (contents pages fail this)
MAX_TITLE_LINES = 3          # short chapter-title lines skipped after a heading
SEARCH_FRACTION = 0.2        # the narrative must start within the first 20% of the file
LATE_START_FRACTION = 0.10   # a start later than 10% of the file is flagged uncertain

GUTENBERG_START_RE = re.compile(r"^\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG.*$", re.M | re.I)
GUTENBERG_END_RE = re.compile(r"^\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG.*$", re.M | re.I)

PREVIEW_BEFORE = 150         # characters shown before the cut in cuts.csv
PREVIEW_AFTER = 250          # characters shown after the cut


@dataclass
class Book:
    filename: str
    author: str = ''
    included: bool = False
    reason: str = ''                 # why excluded ('' if included)
    detail: str = ''
    method: str = ''                 # heading | manual | none_found | gutenberg-only ...
    start_char: int = 0
    words_trimmed: int = 0           # words removed before the narrative starts
    n_words: int = 0                 # words in the trimmed narrative
    text: str = field(default='', repr=False)   # trimmed, whitespace-normalised
    before_preview: str = field(default='', repr=False)
    after_preview: str = field(default='', repr=False)


# ------------------------------------------------------------------ helpers --
def read_csv_map(path: Optional[str], key: str, value: str) -> Dict[str, str]:
    """Read a two-column CSV into {filename: value}. Missing file -> {}."""
    if not path:
        return {}
    if not os.path.exists(path):
        raise SystemExit(f'File not found: {path}')
    out = {}
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            k = (row.get(key) or '').strip()
            v = (row.get(value) or '').strip()
            if k and v:
                out[k] = v
    return out


def author_from_filename(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    if '_' not in stem:
        return ''
    return stem.split('_', 1)[0].strip()


def strip_gutenberg(raw: str) -> Tuple[str, bool]:
    """Cut Project Gutenberg header/footer if the markers are present."""
    changed = False
    m = GUTENBERG_START_RE.search(raw)
    if m:
        raw = raw[m.end():]
        changed = True
    m = GUTENBERG_END_RE.search(raw)
    if m:
        raw = raw[:m.start()]
        changed = True
    return raw, changed


def _is_heading(line: str) -> bool:
    return bool(HEADING_RE.match(line.strip()))


def find_narrative_start(raw: str) -> Tuple[int, str]:
    """(char offset of the first narrative prose line, method). 'none_found' -> offset 0."""
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
        j, titles = i + 1, 0                    # skip blank lines and short title lines
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
        words, k = 0, j                          # real prose before the next heading?
        while k < len(lines) and words < MIN_PROSE_WORDS:
            if _is_heading(lines[k]):
                break
            words += len(lines[k].split())
            k += 1
        if words >= MIN_PROSE_WORDS:
            return offsets[j], 'heading'
    return 0, 'none_found'


def find_manual_start(raw: str, start_text: str) -> int:
    """Offset of start_text in raw, ignoring whitespace differences. -1 if absent."""
    toks = start_text.split()
    if not toks:
        return -1
    pat = r'\s+'.join(re.escape(t) for t in toks)
    m = re.search(pat, raw)
    return m.start() if m else -1


def normalise(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


# ------------------------------------------------------------- one book ----
def load_book(path: str, filename: str, starts: Dict[str, str], authors: Dict[str, str],
              min_words: int) -> Book:
    b = Book(filename=filename)
    b.author = authors.get(filename) or author_from_filename(filename)
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            raw = f.read().lstrip('﻿')
    except OSError as e:
        b.reason, b.detail = 'unreadable', str(e)
        return b
    raw, had_boiler = strip_gutenberg(raw)

    if filename in starts:
        start = find_manual_start(raw, starts[filename])
        if start < 0:
            b.reason = 'uncertain_start'
            b.detail = 'starts.csv sentence not found in the file'
            b.method = 'manual_not_found'
            return b
        method = 'manual'
    else:
        start, method = find_narrative_start(raw)

    b.method = method
    b.start_char = start
    b.before_preview = normalise(raw[max(0, start - 600):start])[-PREVIEW_BEFORE:]
    narrative = normalise(raw[start:])
    b.text = narrative
    b.after_preview = narrative[:PREVIEW_AFTER]
    b.words_trimmed = len(WORD_RE.findall(raw[:start]))
    b.n_words = len(WORD_RE.findall(narrative))

    if not b.author:
        b.reason, b.detail = 'no_author', 'no underscore in the filename and no authors.csv entry'
    elif method == 'none_found':
        b.reason, b.detail = 'uncertain_start', 'no chapter heading followed by prose found (add it to starts.csv to include)'
    elif method == 'heading' and start > LATE_START_FRACTION * max(len(raw), 1):
        b.reason = 'uncertain_start'
        b.detail = (f'narrative starts at {start / max(len(raw), 1):.0%} of the file '
                    f'(limit {LATE_START_FRACTION:.0%}); check the cut, then add to starts.csv')
    elif b.n_words < min_words:
        b.reason, b.detail = 'too_short', f'{b.n_words:,} words after trimming (minimum {min_words:,})'
    return b


# ---------------------------------------------------------------- corpus ----
def load_corpus(corpus_dir: str, starts_csv: Optional[str] = None, authors_csv: Optional[str] = None,
                min_words: int = 20000, min_books: int = 2) -> List[Book]:
    """Load every .txt in corpus_dir. Returns ALL books; those with .included True pass every rule."""
    if not os.path.isdir(corpus_dir):
        raise SystemExit(f'Not a folder: {corpus_dir}')
    starts = read_csv_map(starts_csv, 'filename', 'start_text')
    authors = read_csv_map(authors_csv, 'filename', 'author')
    files = sorted(fn for fn in os.listdir(corpus_dir) if fn.lower().endswith('.txt'))
    if not files:
        raise SystemExit(f'No .txt files in {corpus_dir}')
    known = set(files)
    for name, table in (('starts.csv', starts), ('authors.csv', authors)):
        missing = sorted(set(table) - known)
        if missing:
            print(f'NOTE: {name} lists files not in the folder: {", ".join(missing[:5])}'
                  + (' ...' if len(missing) > 5 else ''))
    books = [load_book(os.path.join(corpus_dir, fn), fn, starts, authors, min_words) for fn in files]

    passing = Counter(b.author for b in books if not b.reason)
    for b in books:
        if b.reason:
            continue
        if passing[b.author] < min_books:
            b.reason = 'author_too_few_books'
            b.detail = (f'{b.author} has {passing[b.author]} usable book(s) (minimum {min_books}); '
                        f'check its other books in exclusions.csv')
        else:
            b.included = True
    return books


# ---------------------------------------------------------------- outputs ---
def write_outputs(books: List[Book], out_dir: str, args) -> None:
    os.makedirs(out_dir, exist_ok=True)
    inc = [b for b in books if b.included]
    exc = [b for b in books if not b.included]

    with open(os.path.join(out_dir, 'manifest.csv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['filename', 'author', 'n_words', 'words_trimmed', 'start_method'])
        for b in inc:
            w.writerow([b.filename, b.author, b.n_words, b.words_trimmed, b.method])

    with open(os.path.join(out_dir, 'exclusions.csv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['filename', 'author', 'reason', 'detail'])
        for b in exc:
            w.writerow([b.filename, b.author, b.reason, b.detail])

    with open(os.path.join(out_dir, 'cuts.csv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['filename', 'author', 'status', 'start_method', 'words_trimmed', 'n_words',
                    'text_before_cut', 'text_after_cut'])
        for b in books:
            w.writerow([b.filename, b.author, 'included' if b.included else b.reason, b.method,
                        b.words_trimmed, b.n_words, b.before_preview, b.after_preview])

    by_author = Counter(b.author for b in inc)
    reasons = Counter(b.reason for b in exc)
    L = ['# Loader summary', '',
         f'Folder: `{args.corpus_dir}`. {len(books)} .txt files found; **{len(inc)} included** '
         f'({len(by_author)} authors), {len(exc)} excluded.',
         f'Rules: at least {args.min_words:,} words after trimming; at least {args.min_books} included '
         f'books per author; author = filename before the first underscore'
         + (' (with authors.csv overrides)' if args.authors else '') + '.', '']
    if exc:
        L += ['## Excluded', '', '| reason | books |', '|---|---:|']
        L += [f'| {r} | {n} |' for r, n in reasons.most_common()]
        L += ['', 'Details are in exclusions.csv. For `uncertain_start`, open cuts.csv, check the text '
              'before/after the cut, and add the real first sentence to starts.csv to bring the book in.', '']
    if inc:
        counts = Counter(by_author.values())
        L += ['## Included authors', '',
              'Books per author: ' + ', '.join(f'{k} books: {v} authors' for k, v in sorted(counts.items())), '']
        methods = Counter(b.method for b in inc)
        L += ['Start method: ' + ', '.join(f'{m} ({n})' for m, n in methods.most_common()), '',
              'Check cuts.csv by eye before trusting any result: a leftover preface or contents page '
              'inside the first 1,500 words would contaminate the opening measurement.', '']
    with open(os.path.join(out_dir, 'loader_summary.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('Files read')[0],
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--corpus-dir', required=True, help='folder of .txt novels')
    ap.add_argument('--out', default='output_loader', help='output folder (default output_loader)')
    ap.add_argument('--authors', default=None, help='optional authors.csv (filename,author)')
    ap.add_argument('--starts', default=None, help='optional starts.csv (filename,start_text)')
    ap.add_argument('--min-words', type=int, default=20000)
    ap.add_argument('--min-books', type=int, default=2)
    args = ap.parse_args()

    books = load_corpus(args.corpus_dir, args.starts, args.authors, args.min_words, args.min_books)
    write_outputs(books, args.out, args)
    inc = [b for b in books if b.included]
    print(f'{len(books)} files: {len(inc)} included ({len({b.author for b in inc})} authors), '
          f'{len(books) - len(inc)} excluded.')
    for r, n in Counter(b.reason for b in books if not b.included).most_common():
        print(f'  excluded - {r}: {n}')
    print(f'Wrote manifest.csv, exclusions.csv, cuts.csv, loader_summary.md to {args.out}/')


if __name__ == '__main__':
    main()
