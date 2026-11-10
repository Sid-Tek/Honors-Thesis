"""
hathi_metadata.py -- Step 2: attach author/title/date to each Hathi volume ID.

Streams a HathiTrust "Hathifile" (hathi_full_YYYYMMDD.txt.gz, tab-separated, no
header row) and keeps only the rows whose htid is in your inventory.csv.
Nothing is loaded fully into memory; works on the .gz directly.

Usage:
    python -I hathi_metadata.py <hathi_full_....txt.gz> <hathi_inventory_out/inventory.csv> [out.csv]

Writes metadata.csv (default hathi_inventory_out/metadata.csv) and prints a summary:
match count, missing IDs, language mix, and date spreads.

NOTE: Hathi's rights_date_used is NOT the publication date (e.g. a 1902 book can
carry 1962). So we also derive:
  imprint_year   first 4-digit year in the imprint (edition date; reprints show reprint year)
  author_birth / author_death   from the author field ("Bangs, John Kendrick, 1862-1922.")
  reprint_flag   1 if title/imprint mentions facsimile/reprint/reproduction/reissue
Zip names encode '.' as ',' (also ':' as '+', '/' as '='); we decode them to match the catalog.

Hathifile columns (positional, per HathiTrust docs):
 0 htid  1 access  2 rights  3 ht_bib_key  4 description(volume)  5 source
 6 source_inst_bib_num  7 oclc_num  8 isbn  9 issn  10 lccn  11 title
 12 imprint  13 rights_reason_code  14 rights_timestamp  15 us_gov_doc_flag
 16 rights_date_used  17 pub_place  18 lang  19 bib_fmt  20 collection_code
 21 content_provider_code  22 responsible_entity_code  23 digitization_agency_code
 24 access_profile_code  25 author
"""
import csv
import gzip
import os
import re
import sys
from collections import Counter

YEAR_RE = re.compile(r"(?<!\d)(1[5-9]\d{2}|20\d{2})(?!\d)")
LIFE_RE = re.compile(r"(?<!\d)(\d{4})\??\s*-\s*(\d{4})?(?!\d)")
REPRINT_RE = re.compile(r"facsimile|fac-simile|reprint|reproduction|reissue|re-issue", re.I)


def decode_id(zip_id):
    """Hathi filename form -> catalog htid: namespace.rest with ',' '.', '+' ':', '=' '/'."""
    ns, dot, rest = zip_id.partition(".")
    return ns + dot + rest.replace(",", ".").replace("+", ":").replace("=", "/")


COLS = {"htid": 0, "rights": 2, "ht_bib_key": 3, "volume_desc": 4, "oclc": 7,
        "title": 11, "imprint": 12, "rights_date_used": 16, "pub_place": 17,
        "lang": 18, "bib_fmt": 19, "author": 25}


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    hf, inv = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(inv)), "metadata.csv")
    for p in (hf, inv):
        if not os.path.isfile(p):
            sys.exit(f"File not found: {p}")

    with open(inv, newline="", encoding="utf-8") as f:
        zip_ids = [r["id_guess"] for r in csv.DictReader(f) if r["id_guess"]]
    wanted = {decode_id(z): z for z in zip_ids}  # catalog htid -> zip id
    print(f"{len(wanted)} volume IDs wanted; scanning {hf} ...", file=sys.stderr)

    opener = gzip.open if hf.lower().endswith(".gz") else open
    found, seen = {}, 0
    with opener(hf, "rt", encoding="utf-8", errors="replace", newline="") as f:
        for line in f:
            seen += 1
            if seen % 2_000_000 == 0:
                print(f"  ...{seen:,} rows, {len(found)} matched", file=sys.stderr)
            htid = line.split("\t", 1)[0]
            if htid in wanted:
                parts = line.rstrip("\r\n").split("\t")
                found[htid] = {k: (parts[i] if i < len(parts) else "") for k, i in COLS.items()}
                if len(found) == len(wanted):
                    break

    derived = ["zip_id", "imprint_year", "author_birth", "author_death", "reprint_flag"]
    for htid, r in found.items():
        y = YEAR_RE.search(r["imprint"])
        m = LIFE_RE.search(r["author"])
        r["zip_id"] = wanted[htid]
        r["imprint_year"] = y.group(1) if y else ""
        r["author_birth"] = m.group(1) if m else ""
        r["author_death"] = (m.group(2) or "") if m else ""
        r["reprint_flag"] = 1 if REPRINT_RE.search(r["title"] + " " + r["imprint"]) else 0

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=derived + list(COLS))
        w.writeheader()
        for htid in sorted(wanted):
            if htid in found:
                w.writerow(found[htid])

    missing = sorted(wanted[h] for h in wanted if h not in found)
    print(f"\nRows scanned: {seen:,}")
    print(f"Matched: {len(found)} / {len(wanted)}   missing: {len(missing)}")
    if missing:
        print("First missing IDs:", missing[:10])
        print("(If many are missing, you may have an older/newer Hathifile than your download; try the full file.)")
    print("Languages:", Counter(r["lang"] for r in found.values()).most_common(6))
    print("Record formats (BK = book):", Counter(r["bib_fmt"] for r in found.values()).most_common(5))
    def by_decade(vals):
        c = Counter()
        for v in vals:
            c[f"{v[:3]}0s" if v.isdigit() and len(v) == 4 else "unknown"] += 1
        return sorted(c.items())

    print("rights_date_used by decade (NOT pub date):", by_decade(r["rights_date_used"] for r in found.values()))
    print("imprint_year by decade:", by_decade(r["imprint_year"] for r in found.values()))
    print("author birth by decade:", by_decade(r["author_birth"] for r in found.values()))
    print("Likely reprints/facsimiles:", sum(r["reprint_flag"] for r in found.values()))
    print("Distinct authors:", len({r["author"] for r in found.values() if r["author"]}))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
