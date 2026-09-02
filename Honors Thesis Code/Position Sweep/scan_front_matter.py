"""
Corpus-wide scan for the table-of-contents / front-matter contamination
found in Blackmore_Erema_1877.txt: unusually dense repetition of a short
n-gram (e.g. "CHA", "TER", "HAP" from a run of "CHAPTER I ... CHAPTER II
... CHAPTER III ...") within the first stretch of a novel. That's a
structural/formatting artifact, not prose -- and inflates offset-0
accuracy in the position sweep for reasons that have nothing to do with
voice.

This doesn't fix anything. It tells you WHICH novels in your corpus have
the problem, how bad it looks, and roughly how many characters into the
file the contamination seems to run -- enough to decide between fixing
the source files, skipping a safe number of characters (uniformly, or
per-novel using the estimate this script gives you), or -- if only a
couple of novels are affected -- just excluding them.

Usage:
    python scan_front_matter.py /path/to/corpus_folder
    python scan_front_matter.py /path/to/corpus_folder --scan-chars 2000 --threshold 10

    python scan_front_matter.py 100_english_novels-master/corpus
"""

import argparse
from pathlib import Path

from ngram_categories import extract_ngram_categories, CATEGORIES


def scan_document(text, scan_chars=2000, n=3, threshold_ratio=8.0, min_front_count=8):
    """
    Compares the front `scan_chars` of a document against the REST of
    that same document. For each n-gram, computes its rate (occurrences
    per character) in the front window vs. the rest of the novel, and
    flags the n-gram with the highest front/rest ratio -- provided it
    also occurs at least `min_front_count` times up front (so a rare
    n-gram's noisy ratio, driven by a tiny denominator, doesn't win).

    This self-normalizes against ordinary common words: "the" or "he "
    are common EVERYWHERE in a novel, so their front/rest ratio stays
    close to 1 no matter how frequent they are in absolute terms. An
    artifact concentrated ONLY at the very front -- like a table of
    contents repeating "CHAPTER" dozens of times before the narrative
    even starts -- has a front rate many times its (near-zero) rate in
    the rest of the book, so the ratio spikes instead.

    Returns None if the document is too short to compare against, or if
    no n-gram's ratio clears `threshold_ratio`. Otherwise returns a dict
    with the offending n-gram, its front/rest counts and ratio, and the
    character position of its last occurrence in the front window (a
    rough proxy for how far the contamination extends).
    """
    front = text[:scan_chars]
    rest = text[scan_chars:]
    if len(rest) < scan_chars:
        return None  # not enough "rest" of the novel to compare against

    front_counts = extract_ngram_categories(front, n=n)
    rest_counts = extract_ngram_categories(rest, n=n)
    front_len, rest_len = len(front), len(rest)

    best = None  # (ngram, ratio, front_count, rest_count, category)
    for cat in CATEGORIES:
        for ngram, f_count in front_counts[cat].items():
            if f_count < min_front_count:
                continue
            r_count = rest_counts[cat].get(ngram, 0)
            front_rate = f_count / front_len
            rest_rate = (r_count + 1) / rest_len  # +1: avoid divide-by-zero
            ratio = front_rate / rest_rate
            if best is None or ratio > best[1]:
                best = (ngram, ratio, f_count, r_count, cat)

    if best is None:
        return None
    ngram, ratio, f_count, r_count, cat = best
    if ratio < threshold_ratio:
        return None

    last_pos = front.rfind(ngram) + len(ngram)
    return {
        "ngram": ngram,
        "front_count": f_count,
        "rest_count": r_count,
        "ratio": ratio,
        "category": cat,
        "last_position": last_pos,
    }


def scan_corpus(folder, scan_chars=2000, n=3, threshold_ratio=8.0,
                 min_front_count=8, verbose=True):
    """
    Scans every Author_Title.txt file in `folder`. Returns
    (flagged, total) where flagged is a list of (filename, scan_result)
    tuples for flagged novels only, sorted worst-offense-first (by
    ratio, since that's the actual anomaly signal here).
    """
    folder = Path(folder)
    files = sorted(folder.glob("*.txt"))
    flagged = []
    for i, path in enumerate(files):
        text = path.read_text(encoding="utf-8", errors="replace")
        result = scan_document(text, scan_chars=scan_chars, n=n,
                                threshold_ratio=threshold_ratio,
                                min_front_count=min_front_count)
        if result is not None:
            flagged.append((path.name, result))
        if verbose:
            print(f"  Scanned {i + 1}/{len(files)}...", end="\r")
    if verbose:
        print()
    flagged.sort(key=lambda pair: -pair[1]["ratio"])
    return flagged, len(files)


def main():
    ap = argparse.ArgumentParser(
        description="Scan a corpus for table-of-contents / front-matter "
                    "contamination at the start of each novel."
    )
    ap.add_argument("corpus_folder", help="Folder of Author_Title.txt novel files")
    ap.add_argument("--scan-chars", type=int, default=2000,
                     help="How many characters from the start of each novel "
                          "to treat as the 'front' window (default 2000)")
    ap.add_argument("--n", type=int, default=3, help="n-gram length (default 3)")
    ap.add_argument("--threshold-ratio", type=float, default=8.0,
                     help="Flag a novel if any n-gram's front-window rate is "
                          "at least this many times its rate in the rest of "
                          "the novel (default 8.0)")
    ap.add_argument("--min-front-count", type=int, default=8,
                     help="Minimum occurrences in the front window for an "
                          "n-gram to be considered at all -- avoids chasing "
                          "noisy ratios from rare n-grams (default 8)")
    args = ap.parse_args()

    print(f"Scanning {args.corpus_folder} (front {args.scan_chars} chars vs. "
          f"rest of each novel, flagging ratio >= {args.threshold_ratio})...")
    flagged, total = scan_corpus(args.corpus_folder, scan_chars=args.scan_chars,
                                  n=args.n, threshold_ratio=args.threshold_ratio,
                                  min_front_count=args.min_front_count)

    print(f"\n{len(flagged)}/{total} novels flagged:\n")
    for filename, result in flagged:
        print(f"  {filename:40s} {result['ngram']!r:8s} "
              f"front={result['front_count']:<4d} rest={result['rest_count']:<4d} "
              f"ratio={result['ratio']:<7.1f} ({result['category']:12s}) "
              f"-- contamination runs to ~char {result['last_position']}")

    if flagged:
        max_extent = max(r["last_position"] for _, r in flagged)
        print(f"\nWorst-case contamination extends to ~character {max_extent} "
              f"into the file (within the {args.scan_chars}-char scan window).")
        print("For a uniform skip-N-characters fix, that's a reasonable floor "
              "for N. For a per-novel fix, each 'last_position' above is a "
              "starting estimate of where to cut -- spot-check a few by eye "
              "before trusting it corpus-wide, since a repeated n-gram's last "
              "position is only a rough proxy for where real narrative starts.")
    else:
        print("\nNo novels flagged at this threshold.")


if __name__ == "__main__":
    main()