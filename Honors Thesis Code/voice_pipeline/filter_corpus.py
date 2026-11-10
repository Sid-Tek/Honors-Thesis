"""
filter_corpus.py -- Step 3: trial filter on metadata.csv (reads only; touches no zips).

Keeps volumes that:
  * have imprint_year in [--from, --to]              (default 1930-1969)
  * are not flagged as a reprint/facsimile
  * have a known author
  * belong to an author with >= --min-books DISTINCT titles (default 3)
Author birth year is NOT used to exclude. It is reported (early_author column) so you
can eyeball authors born before 1870, whose books may be reprints of older works.

Usage:
    python -I filter_corpus.py hathi_inventory_out/metadata.csv [out_dir]
        [--from 1930] [--to 1969] [--min-books 3] [--born-from 1870] [--keep-unknown-birth]

Writes out_dir (default: next to metadata.csv):
    filtered.csv       kept volumes
    author_counts.csv  authors kept, with number of distinct titles
"""
import argparse
import csv
import os
import re
from collections import Counter, defaultdict


def author_key(a):
    a = re.sub(r"[\d\?\-]+", " ", a)          # drop dates
    a = re.sub(r"[.,;]+\s*$", "", a.strip())
    a = re.sub(r"\s+", " ", a).strip(" ,.;").lower()
    return a


def title_key(t):
    t = t.split("/")[0]                          # drop "/ by Author" part
    t = re.sub(r"[^a-z0-9 ]", "", t.lower())
    return re.sub(r"\s+", " ", t).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metadata")
    ap.add_argument("out_dir", nargs="?")
    ap.add_argument("--from", dest="y0", type=int, default=1930)
    ap.add_argument("--to", dest="y1", type=int, default=1969)
    ap.add_argument("--min-books", type=int, default=3)
    ap.add_argument("--born-from", type=int, default=1870)
    ap.add_argument("--keep-unknown-birth", action="store_true")
    a = ap.parse_args()
    out_dir = a.out_dir or os.path.dirname(os.path.abspath(a.metadata))
    os.makedirs(out_dir, exist_ok=True)

    with open(a.metadata, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    n0 = len(rows)

    def ok_year(r):
        y = r["imprint_year"]
        return y.isdigit() and a.y0 <= int(y) <= a.y1

    stages = [("all volumes", rows)]
    rows = [r for r in rows if ok_year(r)]
    stages.append((f"imprint year {a.y0}-{a.y1}", rows))
    rows = [r for r in rows if r["reprint_flag"] != "1"]
    stages.append(("not flagged reprint", rows))
    rows = [r for r in rows if author_key(r["author"])]
    stages.append(("author known", rows))

    def ok_birth(r):
        b = r["author_birth"]
        if b.isdigit():
            return int(b) >= a.born_from
        return a.keep_unknown_birth

    n_unknown = sum(1 for r in rows if not r["author_birth"].isdigit())
    rows = [r for r in rows if ok_birth(r)]
    label = f"author born >= {a.born_from}" + (" (unknown kept)" if a.keep_unknown_birth else " (unknown dropped)")
    stages.append((label, rows))

    titles = defaultdict(set)
    for r in rows:
        r["author_key"] = author_key(r["author"])
        titles[r["author_key"]].add(title_key(r["title"]))
    keep_auth = {k for k, t in titles.items() if len(t) >= a.min_books}
    kept = [r for r in rows if r["author_key"] in keep_auth]
    stages.append((f"author has >= {a.min_books} distinct titles", kept))

    print("Filter funnel (volumes):")
    for name, rs in stages:
        print(f"  {len(rs):5d}  {name}")
    print(f"\nAuthors kept: {len(keep_auth)}   (of {len(titles)} authors passing earlier stages)")
    dist = Counter(min(len(titles[k]), 10) for k in keep_auth)
    print("Distinct titles per kept author:", sorted(dist.items()), "(10 = 10 or more)")

    unk = {k for k in keep_auth
           if all(not r["author_birth"].isdigit() for r in kept if r["author_key"] == k)}
    multi = sum(1 for r in kept if r.get("volume_desc", "").strip())
    print(f"Volumes with no birth year before the birth filter: {n_unknown}"
          f"  ({'kept' if a.keep_unknown_birth else 'dropped; use --keep-unknown-birth to keep'})")
    print(f"Kept authors with no birth year in catalog: {len(unk)}")
    print(f"Kept volumes with a volume description (possible multi-volume works): {multi}")

    cols = ["zip_id", "author_key", "author", "title", "imprint_year", "author_birth",
            "volume_desc"]
    with open(os.path.join(out_dir, "filtered.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in sorted(kept, key=lambda r: (r["author_key"], r["imprint_year"])):
            w.writerow([r["zip_id"], r["author_key"], r["author"], r["title"], r["imprint_year"],
                        r["author_birth"], r["volume_desc"]])
    with open(os.path.join(out_dir, "author_counts.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["author_key", "distinct_titles", "volumes", "no_birth_year"])
        for k in sorted(keep_auth, key=lambda k: -len(titles[k])):
            w.writerow([k, len(titles[k]), sum(1 for r in kept if r["author_key"] == k),
                        1 if k in unk else 0])
    print(f"\nWrote {out_dir}/filtered.csv and author_counts.csv")


if __name__ == "__main__":
    main()
