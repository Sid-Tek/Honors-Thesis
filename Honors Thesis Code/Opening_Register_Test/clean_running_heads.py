"""
Opening Register Test -- corpus clean-up: strip page numbers and running heads.

Why
---
The Hathi scans keep the printed page furniture: a page number on about 29% of
pages and, in most books, the book title (or a chapter title) repeated at the
top of every page. Chapter-opening pages usually have none, so the opening of a
book would carry slightly fewer of these stray words than the rest of the book.
This script removes them BEFORE the test, with a fixed, content-blind rule.
It never looks at test results and uses nothing but the page layout.

The rule (decided 2026-11-10, before the first run on this corpus)
------------------------------------------------------------------
Pages are separated by form-feed characters in Hathi_corpus_trimmed. On each
page only the "edge" lines are considered: the first 4 and the last 4
non-blank lines. Body lines are never touched.

1. PAGE NUMBER: an edge line that is only a page number (1-3 digits,
   optionally with dashes, brackets, dots, or OCR look-alikes such as 4I or l2) is removed.
2. RUNNING HEAD: an edge line is removed if, ignoring digits, capitals and
   punctuation, the same line is an edge line on at least MIN_PAGES (6) pages of
   the SAME book, AND it has at least 5 letters, at most 8 words, no quotation
   marks, and does not start with CHAPTER / BOOK / PART / VOLUME. For lines
   printed entirely in capitals with at least 2 words (chapter titles used as
   running heads), 3 pages are enough (MIN_PAGES_CAPS).
3. SPLIT-TITLE FRAGMENTS: a shorter repeated edge line (under 5 letters, e.g.
   "THE" or "LAST" when the title is printed over several lines) is removed if
   it repeats on at least MIN_PAGES pages and all its words occur in a running
   head found by rule 2.

Not touched: hyphenated line breaks ("Ken-" / "tucky"), lost drop capitals,
chapter headings, time-stamp headings, dialogue, anything in the body of a page.

Run (from the Opening_Register_Test folder; standard library only):
    python -I clean_running_heads.py --in ..\\Hathi_corpus_trimmed --out ..\\Hathi_corpus_clean

Writes, in --out:
    <book>.txt             the cleaned texts (same file names as --in)
    cleaning_report.md     READ THIS FIRST: totals, flags, residual repeated lines
    cleaning_log.csv       per book: pages, lines removed, words removed, % removed
    removed_heads.csv      every removed repeated line (per book, with counts)
    starts_clean.csv       first 12 words of each cleaned text (for register_test.py --starts)
    starts_asis.csv        the same for the untouched --in texts (for the as-is run)
"""
import argparse
import csv
import os
import re
from collections import Counter

MIN_PAGES = 6            # a repeated edge line must appear on this many pages of one book
MIN_PAGES_CAPS = 3       # same, for ALL-CAPS lines of 2+ words (chapter titles used as running heads)
EDGE_LINES = 4           # first and last 4 non-blank lines of a page (titles can be printed over several lines)
MIN_LETTERS = 5
MAX_WORDS = 8
FLAG_PCT = 3.0           # flag a book if more than this % of its words are removed
RESIDUAL_MIN = 3         # report repeated edge lines on 3..MIN_PAGES-1 pages that were NOT removed
START_WORDS = 12

WORD_RE = re.compile(r"[A-Za-z']+")                       # same word rule as the test
# a page number: 1-3 digits (OCR look-alikes I l O o | ! allowed), optionally wrapped in [ ] ( ) dashes or dots
PAGE_NUMBER_RE = re.compile(r'^[\[\(\-–—.\s]*[0-9IlOo|!]{1,3}[\]\)\-–—.\s]*$')
PROTECTED_RE = re.compile(r'^(CHAPTER|BOOK|PART|VOLUME)\b')
QUOTE_CHARS = '"“”'


def head_key(line):
    s = re.sub(r'\d+', '', line.upper())
    s = re.sub(r'[^A-Z ]', ' ', s)
    return ' '.join(s.split())


def is_page_number(line):
    s = line.strip()
    return bool(s) and len(s) <= 10 and any(c in '0123456789' for c in s) and bool(PAGE_NUMBER_RE.match(s))


def edge_indices(lines):
    """Indices (into `lines`) of the first and last EDGE_LINES non-blank lines of a page."""
    nb = [i for i, l in enumerate(lines) if l.strip()]
    if len(nb) <= 2 * EDGE_LINES:
        return set(nb)
    return set(nb[:EDGE_LINES]) | set(nb[-EDGE_LINES:])


def head_candidate(line):
    """True if the line could be a running head (before the repetition test)."""
    if any(q in line for q in QUOTE_CHARS):
        return False
    k = head_key(line)
    return bool(k) and len(line.split()) <= MAX_WORDS


def clean_text(text):
    """Returns (cleaned_text, stats, removed Counter {key: (n_lines, example)}, residual Counter)."""
    pages = text.split('\f')
    page_lines = [p.split('\n') for p in pages]
    edges = [edge_indices(ls) for ls in page_lines]

    # --- count on how many pages each candidate key is an edge line
    pages_with_key = Counter()
    example = {}
    for ls, ed in zip(page_lines, edges):
        seen = set()
        for i in ed:
            if is_page_number(ls[i]) or not head_candidate(ls[i]):
                continue
            k = head_key(ls[i])
            if k not in seen:
                seen.add(k)
                pages_with_key[k] += 1
                example.setdefault(k, ls[i].strip())

    # --- rule 2: running heads; rule 3: split-title fragments
    def need(k):
        ex = example[k]
        return MIN_PAGES_CAPS if (not re.search(r'[a-z]', ex) and len(k.split()) >= 2) else MIN_PAGES

    heads = {k for k, n in pages_with_key.items()
             if n >= need(k) and len(k.replace(' ', '')) >= MIN_LETTERS and not PROTECTED_RE.match(k)}
    head_words = {w for k in heads for w in k.split()}
    fragments = {k for k, n in pages_with_key.items()
                 if n >= MIN_PAGES and k not in heads and len(k.replace(' ', '')) < MIN_LETTERS
                 and not PROTECTED_RE.match(k) and all(w in head_words for w in k.split())}
    remove_keys = heads | fragments

    removed = Counter()
    n_numbers = 0
    out_pages = []
    for ls, ed in zip(page_lines, edges):
        keep = []
        for i, l in enumerate(ls):
            if i in ed and l.strip():
                if is_page_number(l):
                    n_numbers += 1
                    continue
                if head_candidate(l) and head_key(l) in remove_keys:
                    removed[head_key(l)] += 1
                    continue
            keep.append(l)
        out_pages.append('\n'.join(keep))
    cleaned = '\f'.join(out_pages)

    residual = {k: n for k, n in pages_with_key.items()
                if k not in remove_keys and RESIDUAL_MIN <= n < MIN_PAGES
                and len(k.replace(' ', '')) >= MIN_LETTERS and not PROTECTED_RE.match(k)}
    stats = {'pages': len(pages), 'numbers_removed': n_numbers, 'heads_removed': sum(removed.values()),
             'distinct_heads': len(removed)}
    return cleaned, stats, removed, example, residual


def first_words(text, n=START_WORDS):
    return ' '.join(text.split()[:n])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('Run (from')[0],
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--in', dest='src', required=True, help='folder of trimmed .txt novels (Hathi_corpus_trimmed)')
    ap.add_argument('--out', required=True, help='NEW folder for the cleaned texts (must not be the --in folder)')
    args = ap.parse_args()
    if os.path.abspath(args.src) == os.path.abspath(args.out):
        raise SystemExit('--out must be a different folder from --in; the original texts are never overwritten.')
    if not os.path.isdir(args.src):
        raise SystemExit(f'Not a folder: {args.src}')
    files = sorted(f for f in os.listdir(args.src) if f.lower().endswith('.txt'))
    if not files:
        raise SystemExit(f'No .txt files in {args.src}')
    os.makedirs(args.out, exist_ok=True)

    log_rows, head_rows, starts_clean, starts_asis, flags, residual_rows = [], [], [], [], [], []
    tot_w0 = tot_w1 = tot_num = tot_head = 0
    for fn in files:
        with open(os.path.join(args.src, fn), 'r', encoding='utf-8', errors='ignore', newline='') as f:
            raw = f.read().lstrip('﻿')
        cleaned, st, removed, example, residual = clean_text(raw)
        with open(os.path.join(args.out, fn), 'w', encoding='utf-8', newline='') as f:
            f.write(cleaned)
        w0, w1 = len(WORD_RE.findall(raw)), len(WORD_RE.findall(cleaned))
        pct = 100.0 * (w0 - w1) / max(w0, 1)
        log_rows.append([fn, st['pages'], st['numbers_removed'], st['heads_removed'], st['distinct_heads'],
                         w0, w1, f'{pct:.2f}'])
        for k, n in removed.most_common():
            head_rows.append([fn, k, example.get(k, ''), n])
        for k, n in sorted(residual.items(), key=lambda kv: -kv[1]):
            residual_rows.append([fn, k, example.get(k, ''), n])
        starts_clean.append([fn, first_words(cleaned)])
        starts_asis.append([fn, first_words(raw)])
        if pct > FLAG_PCT:
            flags.append(f'{fn}: {pct:.1f}% of words removed')
        tot_w0 += w0
        tot_w1 += w1
        tot_num += st['numbers_removed']
        tot_head += st['heads_removed']

    def write_csv(name, header, rows):
        with open(os.path.join(args.out, name), 'w', encoding='utf-8', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    write_csv('cleaning_log.csv', ['filename', 'pages', 'page_number_lines_removed', 'running_head_lines_removed',
                                   'distinct_heads', 'words_before', 'words_after', 'pct_words_removed'], log_rows)
    write_csv('removed_heads.csv', ['filename', 'key', 'example_line', 'lines_removed'], head_rows)
    write_csv('starts_clean.csv', ['filename', 'start_text'], starts_clean)
    write_csv('starts_asis.csv', ['filename', 'start_text'], starts_asis)

    pcts = sorted(float(r[7]) for r in log_rows)
    L = ['# Cleaning report: page numbers and running heads', '',
         f'{len(files)} texts from `{args.src}` written to `{args.out}`. The originals are untouched.', '',
         f'* Page-number lines removed: {tot_num:,}',
         f'* Running-head lines removed: {tot_head:,}',
         f'* Words: {tot_w0:,} before, {tot_w1:,} after ({100.0 * (tot_w0 - tot_w1) / max(tot_w0, 1):.2f}% removed). '
         f'Per book: median {pcts[len(pcts) // 2]:.2f}%, maximum {pcts[-1]:.2f}%.',
         f'* Books with no running head found: {sum(1 for r in log_rows if r[3] == 0)} '
         f'(page numbers may still have been removed; see cleaning_log.csv).', '']
    if flags:
        L += [f'## Flags (more than {FLAG_PCT:.0f}% of words removed): CHECK THESE BY EYE', ''] + [f'* {x}' for x in flags] + ['']
    else:
        L += [f'No book lost more than {FLAG_PCT:.0f}% of its words.', '']
    L += ['## What was removed', '',
          'removed_heads.csv lists every repeated line that was removed, per book. The most frequent ones across the corpus:', '',
          '| line (as first seen) | books | lines removed |', '|---|---:|---:|']
    agg = Counter()
    nb = Counter()
    for fn, k, ex, n in head_rows:
        agg[(k, ex)] += n
        nb[(k, ex)] += 1
    for (k, ex), n in agg.most_common(15):
        L.append(f'| {ex} | {nb[(k, ex)]} | {n:,} |')
    L += ['', '## Left in, on purpose', '',
          'Mixed-case repeated edge lines on 3-5 pages only (probably chapter titles on short chapters) are NOT removed; '
          'the full list is below so you can see what stayed. If one of these is a running head, say so and it can '
          'be added to the log; it will not change the rule.', '']
    if residual_rows:
        L += ['| book | line | pages |', '|---|---|---:|']
        for fn, k, ex, n in residual_rows[:40]:
            L.append(f'| {fn[:40]} | {ex} | {n} |')
        if len(residual_rows) > 40:
            L.append(f'| ... | {len(residual_rows) - 40} more in the data | |')
    else:
        L.append('None.')
    L += ['', '## Next', '',
          'Run the test on the cleaned texts with `--starts` pointing at starts_clean.csv. The as-is run uses the original '
          'folder with starts_asis.csv. Both read the first story sentence from the file itself, so the loader does not '
          'trim again.', '']
    with open(os.path.join(args.out, 'cleaning_report.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print(f'Cleaned {len(files)} texts: {tot_num:,} page-number lines and {tot_head:,} running-head lines removed '
          f'({100.0 * (tot_w0 - tot_w1) / max(tot_w0, 1):.2f}% of words).')
    if flags:
        print('FLAGS:')
        for x in flags:
            print('  ' + x)
    print(f'Wrote cleaning_report.md (read this first), cleaning_log.csv, removed_heads.csv, starts_clean.csv, '
          f'starts_asis.csv and the cleaned texts to {args.out}')


if __name__ == '__main__':
    main()
