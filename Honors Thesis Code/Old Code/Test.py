"""
batch_vectors.py

Runs the voice-vector extractor from Vector.py over an entire folder of
author-labeled novels/excerpts and writes one CSV row per file, with an
`author` column attached. Optionally runs a quick sanity check: are
same-author vectors closer together than different-author ones?

Expects one of two folder layouts:

  1. FLAT, "Author_Title.txt"  (the `stylo`-package convention used by the
     Computational Stylistics Group's "100 English Novels" benchmark, and
     by most similar academic corpora)
        corpus/
          Austen_Emma.txt
          Austen_Persuasion.txt
          Dickens_OliverTwist.txt

  2. FOLDER-PER-AUTHOR
        corpus/
          Austen/
            emma.txt
            persuasion.txt
          Dickens/
            oliver_twist.txt

Both are auto-detected file by file, so a corpus can even mix the two.

IMPORTANT: this script imports from Vector.py, so it must sit in the same
folder as Vector.py (or Vector.py must be importable on your PYTHONPATH).

Usage:
    python Test.py path/to/corpus --out vectors.csv
    python Test.py path/to/corpus --out vectors.csv --n 3000 --model en_core_web_lg
    python Test.py path/to/corpus --out vectors.csv --analyze
    python Test.py 100_english_novels-master/corpus --out vectors.csv --analyze
    python Test.py 100_english_novels-master/corpus --out vectors.csv --n 3000 --analyze
"""

import argparse
import csv
import math
import sys
from itertools import combinations
from pathlib import Path
from statistics import mean, pstdev

from Vector import load_text, get_nlp, extract_voice_vector


# ---------------------------------------------------------------------------
# File discovery / author labeling
# ---------------------------------------------------------------------------

def discover_files(corpus_dir: Path):
    """Yield (filepath, author) pairs for every .txt file under corpus_dir.

    Layout A (flat, 'Author_Title.txt'): author = text before the first underscore.
    Layout B (folder-per-author):        author = immediate parent folder name.
    """
    files = sorted(corpus_dir.rglob("*.txt"))
    if not files:
        sys.exit(f"No .txt files found under {corpus_dir}")

    results = []
    for f in files:
        if f.parent != corpus_dir:
            # nested inside an author-named subfolder
            author = f.parent.name
        else:
            # flat folder -- split filename on first underscore
            stem = f.stem
            author = stem.split("_")[0] if "_" in stem else stem
        results.append((f, author))
    return results


# ---------------------------------------------------------------------------
# Distance sanity check
# ---------------------------------------------------------------------------

def run_distance_check(rows):
    """Are same-author vectors closer together than different-author ones,
    on average? Uses standardized (z-scored) Euclidean distance over the
    numeric feature columns, so features on different scales (percentages
    vs. raw counts) don't dominate the distance."""

    numeric_keys = [
        k for k in rows[0]
        if k not in ("filename", "author") and not k.startswith("_")
    ]

    stats = {}
    for k in numeric_keys:
        vals = [r[k] for r in rows]
        mu = mean(vals)
        sigma = pstdev(vals) or 1.0  # avoid div-by-zero if a feature is constant
        stats[k] = (mu, sigma)

    def vec(r):
        return [(r[k] - stats[k][0]) / stats[k][1] for k in numeric_keys]

    def dist(a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    same, diff = [], []
    for r1, r2 in combinations(rows, 2):
        d = dist(vec(r1), vec(r2))
        (same if r1["author"] == r2["author"] else diff).append(d)

    print("\n--- Distance check (standardized Euclidean) ---")
    if same:
        print(f"Same-author pairs:      {len(same):5d}   mean distance: {mean(same):.3f}")
    else:
        print("No same-author pairs found -- is there only one text per author?")
    print(f"Different-author pairs:  {len(diff):5d}   mean distance: {mean(diff):.3f}")

    if same:
        ratio = mean(diff) / mean(same)
        verdict = "same-author novels ARE closer -- good sign" if ratio > 1 else "no separation by author yet"
        print(f"\nRatio (different / same): {ratio:.2f}  ({verdict})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Batch-extract voice vectors for an author-labeled corpus.")
    ap.add_argument("corpus_dir", type=Path, help="Folder containing the corpus")
    ap.add_argument("--n", type=int, default=1500, help="Characters to analyze per file (default: 3000)")
    ap.add_argument("--model", default="en_core_web_lg",
                    help="spaCy model (default: en_core_web_lg -- en_core_web_trf is more accurate but much slower over 100 files)")
    ap.add_argument("--out", type=Path, default=Path("vectors.csv"), help="Output CSV path")
    ap.add_argument("--analyze", action="store_true", help="Print same-author vs different-author distance check")
    args = ap.parse_args()

    if not args.corpus_dir.exists():
        sys.exit(f"Folder not found: {args.corpus_dir}")

    pairs = discover_files(args.corpus_dir)

    n_authors = len(set(a for _, a in pairs))
    print(f"Found {len(pairs)} files across {n_authors} authors.")
    print("First few detected (file -> author) pairs -- check these look right before it runs for real:")
    for f, a in pairs[:8]:
        print(f"  {f.name:40s} -> {a}")
    print()

    print(f"Loading spaCy model '{args.model}'... (one-time cost)")
    nlp = get_nlp(args.model)

    rows = []
    fieldnames = None
    for i, (f, author) in enumerate(pairs, 1):
        text = load_text(f, args.n)
        vector = extract_voice_vector(text, nlp)
        row = {"filename": f.name, "author": author, **vector}
        rows.append(row)
        if fieldnames is None:
            fieldnames = list(row.keys())
        print(f"[{i}/{len(pairs)}] {f.name}")

    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} vectors to {args.out}")

    if args.analyze:
        run_distance_check(rows)


if __name__ == "__main__":
    main()