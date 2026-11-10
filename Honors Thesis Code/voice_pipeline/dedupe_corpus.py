"""
dedupe_corpus.py -- Step 5: keep ONE copy of each title in Hathi_corpus.

Groups texts by (author, title). Within a group:
  * word counts within --tol (default 10%) of the largest  -> same edition scanned more than once.
    KEEP the longest (most complete); MOVE the others to ../Hathi_corpus_duplicates/.
  * word counts that differ more than that -> possibly a multi-volume set split into volumes.
    NOT touched; listed under "REVIEW" so you can decide.
Nothing is deleted. A backup manifest_before_dedupe.csv is kept; manifest.csv is rewritten.

Usage (from voice_pipeline):
    python -I dedupe_corpus.py [corpus_dir] [--tol 0.10] [--dry-run]
corpus_dir defaults to ..\\Hathi_corpus
"""
import csv
import os
import re
import shutil
import sys
from collections import defaultdict


def author_key(a):
    a = re.sub(r"[\d\?\-]+", " ", a)
    a = re.sub(r"\s+", " ", a).strip(" ,.;").lower()
    return a


def title_key(t):
    t = t.split("/")[0]
    t = re.sub(r"[^a-z0-9 ]", "", t.lower())
    return re.sub(r"\s+", " ", t).strip()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    tol = 0.10
    if "--tol" in sys.argv:
        tol = float(sys.argv[sys.argv.index("--tol") + 1])
        args = [a for a in args if a != sys.argv[sys.argv.index("--tol") + 1]]
    here = os.path.dirname(os.path.abspath(__file__))
    corpus = args[0] if args else os.path.join(os.path.dirname(here), "Hathi_corpus")
    man_path = os.path.join(corpus, "manifest.csv")
    if not os.path.isfile(man_path):
        sys.exit(f"manifest.csv not found in {corpus}")
    dup_dir = corpus.rstrip("\\/") + "_duplicates"

    with open(man_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0].keys())

    for r in rows:  # fill word counts if the manifest lacks them
        if not str(r.get("words", "")).strip():
            with open(os.path.join(corpus, r["file"]), encoding="utf-8", errors="replace") as tf:
                r["words"] = len(tf.read().split())
        r["words"] = int(r["words"])

    groups = defaultdict(list)
    for r in rows:
        groups[(author_key(r["author"]), title_key(r["title"]))].append(r)

    move, review, keep = [], [], []
    for (a, t), rs in groups.items():
        if len(rs) == 1:
            keep.append(rs[0])
            continue
        rs = sorted(rs, key=lambda r: (-r["words"], r["zip_id"]))
        top = rs[0]["words"]
        if all(r["words"] >= top * (1 - tol) for r in rs):
            keep.append(rs[0])
            move.extend(rs[1:])
        else:
            review.append((a, t, rs))
            keep.extend(rs)

    print(f"{len(rows)} texts, {len(groups)} distinct (author, title) groups")
    print(f"Duplicate scans to move: {len(move)}   groups needing review: {len(review)}")
    for r in move:
        print(f"   move  {r['words']:>7}  {r['file']}")
    for a, t, rs in review:
        print(f"\nREVIEW (same title, very different lengths -> multi-volume set?): {a} | {t[:50]}")
        for r in rs:
            print(f"   {r['words']:>7} words  {r['pages']} pp  {r['zip_id']}  {r['file']}")

    if dry:
        print("\n--dry-run: nothing moved.")
        return
    os.makedirs(dup_dir, exist_ok=True)
    shutil.copy2(man_path, os.path.join(corpus, "manifest_before_dedupe.csv"))
    for r in move:
        src = os.path.join(corpus, r["file"])
        if os.path.exists(src):
            shutil.move(src, os.path.join(dup_dir, r["file"]))
    moved = {r["file"] for r in move}
    with open(man_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            if r["file"] not in moved:
                w.writerow(r)
    kept_n = len(rows) - len(moved)
    n_auth = len({author_key(r["author"]) for r in rows if r["file"] not in moved})
    print(f"\nKept {kept_n} texts from {n_auth} authors in {corpus}")
    print(f"Moved {len(moved)} duplicates to {dup_dir}")


if __name__ == "__main__":
    main()
