"""
exclude_authors.py -- move all texts by the named authors OUT of Hathi_corpus.

Moves files to ../Hathi_corpus_excluded (a sibling folder; nothing is deleted),
rewrites manifest.csv (backup: manifest_before_exclude.csv) and writes
manifest_excluded.csv in the excluded folder so you can see what went and why.

Usage (from voice_pipeline), surnames in any case; multi-word names in quotes:
    python -I exclude_authors.py cather "dos passos" dreiser --dry-run
    python -I exclude_authors.py cather "dos passos" dreiser ferber byrne burgess nathan

A name matches an author whose SURNAME equals it ("Nathan, Robert" matches nathan;
"Nathan Hale" would not). Optional: --corpus <dir> (default ..\\Hathi_corpus).
"""
import csv
import os
import re
import shutil
import sys


def author_key(a):
    a = re.sub(r"[\d\?\-]+", " ", a)
    a = re.sub(r"\s+", " ", a).strip(" ,.;[]").lower()
    return a


def surname(a):
    return author_key(a).split(",")[0].strip()


def main():
    argv = sys.argv[1:]
    dry = "--dry-run" in argv
    here = os.path.dirname(os.path.abspath(__file__))
    corpus = os.path.join(os.path.dirname(here), "Hathi_corpus")
    if "--corpus" in argv:
        i = argv.index("--corpus")
        corpus = argv[i + 1]
        del argv[i:i + 2]
    names = [a.lower().strip() for a in argv if not a.startswith("--")]
    if not names:
        print(__doc__)
        sys.exit(1)
    man_path = os.path.join(corpus, "manifest.csv")
    if not os.path.isfile(man_path):
        sys.exit(f"manifest.csv not found in {corpus}")
    out_dir = corpus.rstrip("\\/") + "_excluded"

    with open(man_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0].keys())

    out, stay = [], []
    for r in rows:
        (out if surname(r["author"]) in names else stay).append(r)

    found = {surname(r["author"]) for r in out}
    for n in names:
        if n not in found:
            print(f"WARNING: no texts matched surname '{n}'")
    print(f"{len(rows)} texts: moving {len(out)}, keeping {len(stay)}")
    by = {}
    for r in out:
        by.setdefault(surname(r["author"]), []).append(r)
    for n, rs in sorted(by.items()):
        print(f"   {n.title():12s} {len(rs):3d} texts")
    if dry:
        print("\n--dry-run: nothing moved.")
        return

    os.makedirs(out_dir, exist_ok=True)
    shutil.copy2(man_path, os.path.join(corpus, "manifest_before_exclude.csv"))
    for r in out:
        src = os.path.join(corpus, r["file"])
        if os.path.exists(src):
            shutil.move(src, os.path.join(out_dir, r["file"]))
    with open(man_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(stay)
    ex_path = os.path.join(out_dir, "manifest_excluded.csv")
    new = not os.path.exists(ex_path)
    with open(ex_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(out)
    n_auth = len({surname(r["author"]) for r in stay})
    print(f"\nKept {len(stay)} texts from {n_auth} authors in {corpus}")
    print(f"Moved {len(out)} texts to {out_dir}")


if __name__ == "__main__":
    main()
