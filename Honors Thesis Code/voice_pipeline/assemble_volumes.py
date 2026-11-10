"""
assemble_volumes.py -- Step 4: turn each kept Hathi volume into ONE raw text file.

For every row in filtered.csv: find <zip_id>.zip under the source folder, read its page
files in memory (no extraction), sort pages numerically, join them, and save as
Surname_Title.txt (same style as Gutenburg_corpus).

RAW: front matter, running heads and OCR noise are left as they are. Trimming is the next step.

Usage:
    python -I assemble_volumes.py <filtered.csv> <source_folder> [out_dir] [--overwrite]
  e.g. python -I assemble_volumes.py hathi_inventory_out\\filtered.csv "C:\\Users\\sidte\\Downloads\\HathiFlat\\HathiFlat"

out_dir defaults to ..\\Hathi_corpus (a sibling of voice_pipeline, next to Gutenburg_corpus).
Existing files are skipped (so you can re-run safely) unless --overwrite.

Also writes <out_dir>/manifest.csv: file, zip_id, author, title, imprint_year, pages, words.
"""
import csv
import os
import re
import sys
import zipfile


def page_no(name):
    m = re.search(r"(\d+)\.txt$", os.path.basename(name), re.I)
    return int(m.group(1)) if m else None


def clean(s, n):
    s = re.sub(r"[^A-Za-z0-9 ]", "", s)
    s = re.sub(r"\s+", "-", s.strip())
    return s[:n].strip("-")


def make_name(author_key, title, taken, year, zip_id):
    surname = clean(author_key.split(",")[0].title(), 30) or "Unknown"
    t = clean(title.split("/")[0].strip(" .:;,"), 60) or "Untitled"
    base = f"{surname}_{t}"
    name = base
    if name.lower() in taken:
        name = f"{base}_{year or 'x'}"
    if name.lower() in taken:
        name = f"{base}_{clean(zip_id, 20)}"
    taken.add(name.lower())
    return name + ".txt"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    overwrite = "--overwrite" in sys.argv
    if len(args) < 2:
        print(__doc__)
        sys.exit(1)
    filtered, source = args[0], args[1]
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = args[2] if len(args) > 2 else os.path.join(os.path.dirname(here), "Hathi_corpus")
    if not os.path.isfile(filtered):
        sys.exit(f"Not found: {filtered}")
    if not os.path.isdir(source):
        sys.exit(f"Source folder not found: {source}")
    os.makedirs(out_dir, exist_ok=True)

    zips = {}
    for root, _d, files in os.walk(source):
        for fn in files:
            if fn.lower().endswith(".zip"):
                zips[fn[:-4]] = os.path.join(root, fn)

    with open(filtered, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"{len(rows)} volumes to assemble -> {out_dir}", file=sys.stderr)

    taken, manifest, problems = set(), [], []
    for i, r in enumerate(rows, 1):
        zid = r["zip_id"]
        fname = make_name(r["author_key"], r["title"], taken, r["imprint_year"], zid)
        path = os.path.join(out_dir, fname)
        zp = zips.get(zid)
        if not zp:
            problems.append(f"zip not found for {zid}")
            continue
        if os.path.exists(path) and not overwrite:
            text_words = None
            pages = None
        else:
            with zipfile.ZipFile(zp) as z:
                pg = sorted(((page_no(x.filename), x) for x in z.infolist()
                             if not x.is_dir() and page_no(x.filename) is not None),
                            key=lambda t: t[0])
                if not pg:
                    problems.append(f"no page files in {zid}")
                    continue
                parts = [z.read(x).decode("utf-8", "replace").strip("\r\n") for _n, x in pg]
            text = "\n".join(parts) + "\n"
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
            pages, text_words = len(pg), len(text.split())
        manifest.append([fname, zid, r["author"], r["title"], r["imprint_year"],
                         "" if pages is None else pages, "" if text_words is None else text_words])
        if i % 20 == 0:
            print(f"  ...{i}/{len(rows)}", file=sys.stderr)

    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "zip_id", "author", "title", "imprint_year", "pages", "words"])
        w.writerows(manifest)

    words = sorted(int(m[6]) for m in manifest if m[6] != "")
    print(f"\nWrote {len(manifest)} files to {out_dir}   problems: {len(problems)}")
    for p in problems[:10]:
        print("  PROBLEM:", p)
    if words:
        print(f"Words per text: min {words[0]:,}  median {words[len(words)//2]:,}  max {words[-1]:,}")
        short = [m for m in manifest if m[6] != "" and int(m[6]) < 30000]
        print(f"Under 30,000 words (probably not novels): {len(short)}")
        for m in short[:15]:
            print(f"   {m[6]:>7}  {m[0]}")
    print(f"Manifest: {os.path.join(out_dir, 'manifest.csv')}")


if __name__ == "__main__":
    main()
