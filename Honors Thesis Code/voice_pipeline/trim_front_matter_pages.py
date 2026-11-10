"""
trim_front_matter_pages.py -- Step 6: cut front matter from the PAGED Hathi texts.

Input : ..\\Hathi_corpus_paged\\*.txt  (pages separated by form-feed, from assemble_volumes.py)
Output: ..\\Hathi_corpus_trimmed\\*.txt  (narrative only; page breaks kept)
        ..\\Hathi_corpus_trimmed\\front_matter_cuts.csv  (for checking every cut by eye)

Rule, page by page (the front matter of a scanned book is a run of short pages --
library stamp, half-title, publisher blurb, title page, copyright, dedication -- and the
story starts on the first full page of prose after them):
  1. Anchors: the LAST short page in the first 30 that carries copyright/printing text, and the
     LAST short page that is just the book's title (second half-title).
  2. From the page after the later anchor, skip: short pages, non-prose pages (lists, stamps),
     and pages that carry >= 2 legal phrases.
  3. If a heading word (CONTENTS, FOREWORD, PREFACE, AUTHOR'S NOTE, A WORD TO THE READER, ...) is
     seen, keep skipping until a page that begins with a chapter-style heading (CHAPTER, BOOK, PART,
     PROLOGUE, or a lone numeral), then take the first prose page at or after it.
  4. On the start page, drop library stamps ("Bequest ... 11-14-39") and a leading "CHAPTER I".
A few books are overridden in MANUAL_START (page number, reason); each is listed in the cuts CSV.
After trimming, pages that repeat an earlier page word for word (>600 letters) are dropped: whole-page
repeats are scan artifacts (re-scanned pages), not prose, and would double-count text in any window.
The first copy is kept; the count removed per book is in the cuts CSV (repeated_pages_removed).
Nothing is deleted from the input folder.

Usage (from voice_pipeline):  python -I trim_front_matter_pages.py [paged_dir] [out_dir]
"""
import csv
import difflib
import os
import re
import sys

COPY = re.compile(r"copyright|all rights reserved|first (edition|published|printing)|printed in (the )?u|manufactured in|published (in|by)|©", re.I)
LEGAL = re.compile(r"no part of this book|all rights reserved|copyright", re.I)
FRONT_CAPS = re.compile(r"\b(CONTENTS|Contents|ILLUSTRATIONS|Illustrations|FOREWORD|Foreword|PREFACE|Preface|"
                        r"INTRODUCTION|Introduction|AUTHOR'?S NOTE|Author'?s Note|ACKNOWLEDG\w*|Acknowledg\w*|"
                        r"A WORD TO THE READER|LEADING CHARACTERS|CAST OF CHARACTERS|DRAMATIS)\b")
CHAP_HEAD = re.compile(r"\b(CHAPTER|Chapter|chapter|BOOK|Book|PART|Part|PROLOGUE|Prologue)\b")
NUMWORD = r"(?:ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|ELEVEN|TWELVE|One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|one|two|three|four|five|six|seven|eight|nine|ten)"
HEADING_PREFIX = re.compile(r"^(?P<pre>.{0,70}?)\b(?:CHAPTER|Chapter|chapter)\s+(?:[IVXLC]+|\d+|" + NUMWORD + r")\b[ \t]*", re.S)
STAMP_PREFIX = re.compile(r"^.{0,60}?\b\d{1,2}[-.]\d{1,2}[-.]\d{2,4}\b[ \t]*(?:" + NUMWORD + r"\b[ \t]*)?", re.S)
SENT = re.compile(r"[.!?][\"'”’)]?(\s|$)")

# filename -> (1-based page where the narrative starts, reason)
MANUAL_START = {
    'Micheaux_The-story-of-Dorothy-Stanfield-based-on-a-great-insurance-sw.txt':
        (33, 'dust-jacket blurb pages repeat until p33; story opens at CHAPTER I'),
    'Fisher_The-island-of-the-innocent-a-novel-of-Greek-and-Jew-in-the-t.txt':
        (13, 'p11 is an expository note on the historical setting; story opens p13'),
    'Thayer_Last-trump.txt':
        (15, 'p10-11 describe a ship deck plan; story follows the second half-title (p13)'),
}


def words(p):
    return re.sub(r"\s+", " ", p).strip().split()


def is_prose(p, minw=100):
    w = words(p)
    if len(w) < minw:
        return False
    alpha = sum(1 for t in w if re.search(r"[A-Za-z]", t)) / len(w)
    return alpha > 0.8 and len(SENT.findall(p)) >= 3


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def title_only(p, title):
    w = words(p)
    if not (1 <= len(w) <= 8):
        return False
    a, b = norm(p), norm(re.split(r"[/:;,]", title)[0])
    if len(a) < 5 or not b:
        return False
    return difflib.SequenceMatcher(None, a, b[:len(a) + 3]).ratio() >= 0.85


def starts_with_heading(p):
    w = words(p)
    head = " ".join(w[:12])
    return bool(CHAP_HEAD.search(head)) or any(re.fullmatch(r"(?:[IVXLC]+\.?|\d{1,3}\.?|" + NUMWORD + r")", t) for t in w[:2])


def drop_repeated_pages(pages):
    """Keep the first copy of any page whose letters (>600) repeat an earlier page exactly."""
    seen, kept, removed = set(), [], 0
    for p in pages:
        key = re.sub(r"[^a-z]", "", p.lower())
        if len(key) > 600:
            if key in seen:
                removed += 1
                continue
            seen.add(key)
        kept.append(p)
    return kept, removed


def find_start(pages, title, fname=""):
    if fname in MANUAL_START:
        return MANUAL_START[fname][0] - 1, 'manual'
    n = min(len(pages), 30)
    copy_idx = title_idx = -1
    for i in range(n):
        w = words(pages[i])
        if w and COPY.search(pages[i]) and (len(w) < 120 or len(LEGAL.findall(pages[i])) >= 2):
            copy_idx = i
        if title_only(pages[i], title):
            title_idx = i
    base = max(copy_idx, title_idx)
    in_front = False
    for i in range(base + 1, n):
        head = " ".join(words(pages[i])[:16])
        if FRONT_CAPS.search(head):
            in_front = True
            continue
        if in_front:
            if starts_with_heading(pages[i]):
                for j in range(i, n):
                    if is_prose(pages[j]) and len(LEGAL.findall(pages[j])) < 2:
                        return j, 'heading_after_front'
            continue
        if is_prose(pages[i]) and len(LEGAL.findall(pages[i])) < 2:
            return i, 'page_rule'
    for i in range(0, min(len(pages), 60)):  # fallback: first prose page anywhere early
        if is_prose(pages[i]) and len(LEGAL.findall(pages[i])) < 2:
            return i, 'fallback_first_prose'
    return 0, 'none_found'


def clean_start(text):
    """Drop a library stamp and/or a leading 'CHAPTER I' from the start page. Returns (text, n_words_removed)."""
    m = HEADING_PREFIX.match(text.lstrip())
    if m:
        rest = text.lstrip()[m.end():]
        return rest, len(words(text)) - len(words(rest))
    m = STAMP_PREFIX.match(text.lstrip())
    if m:
        rest = text.lstrip()[m.end():]
        return rest, len(words(text)) - len(words(rest))
    return text, 0


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    src = args[0] if args else os.path.join(root, "Hathi_corpus_paged")
    out = args[1] if len(args) > 1 else os.path.join(root, "Hathi_corpus_trimmed")
    man = os.path.join(src, "manifest.csv")
    if not os.path.isfile(man):
        sys.exit(f"manifest.csv not found in {src}")
    os.makedirs(out, exist_ok=True)
    with open(man, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    cuts, new_man = [], []
    for r in rows:
        fn = r["file"]
        with open(os.path.join(src, fn), encoding="utf-8", newline="") as f:
            pages = f.read().split("\f")
        s, method = find_start(pages, r["title"], fn)
        pre_words = sum(len(words(p)) for p in pages[:s])
        first, removed = clean_start(pages[s])
        body, n_rep = drop_repeated_pages([first] + pages[s + 1:])
        text = "\f".join(body)
        with open(os.path.join(out, fn), "w", encoding="utf-8", newline="") as f:
            f.write(text)
        before = re.sub(r"\s+", " ", pages[s - 1] if s > 0 else "")[-120:]
        after = re.sub(r"\s+", " ", first)[:160].strip()
        total = len(text.split())
        cuts.append([fn, method, s + 1, pre_words + removed, total, before, after,
                     MANUAL_START.get(fn, (0, ""))[1], n_rep])
        nr = dict(r)
        nr["words"] = total
        new_man.append(nr)

    with open(os.path.join(out, "front_matter_cuts.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "method", "start_page", "words_cut", "words_kept", "end_of_cut_text", "start_of_story", "manual_reason", "repeated_pages_removed"])
        w.writerows(cuts)
    with open(os.path.join(out, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(new_man[0]))
        w.writeheader()
        w.writerows(new_man)

    cw = sorted(c[3] for c in cuts)
    print(f"Trimmed {len(cuts)} texts -> {out}")
    print(f"Words cut: min {cw[0]}  median {cw[len(cw)//2]}  max {cw[-1]}")
    from collections import Counter
    print("Methods:", dict(Counter(c[1] for c in cuts)))
    rep = [c for c in cuts if c[8]]
    print(f"Repeated pages removed: {sum(c[8] for c in rep)} pages across {len(rep)} books")
    odd = [c for c in cuts if c[1] in ("fallback_first_prose", "none_found") or c[3] > 700 or c[3] < 15]
    if odd:
        print("Check these by eye (fallback, very long or very short cut):")
        for c in odd:
            print(f"   {c[0][:50]:50s} {c[1]:20s} cut {c[3]}")
    print(f"Review every cut in {os.path.join(out, 'front_matter_cuts.csv')}")


if __name__ == "__main__":
    main()
