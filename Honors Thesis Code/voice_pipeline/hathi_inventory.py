"""
hathi_inventory.py -- Step 1 of wrangling a zip-of-zips (HathiTrust novel dump).

Reads every inner zip IN MEMORY. Nothing is extracted to disk, so there is no
risk from odd member paths ("../x") and no 1360-folder mess.

Usage (run from anywhere):
    python -I hathi_inventory.py <outer.zip OR folder of .zip files> [out_dir]

Writes to out_dir (default: ./hathi_inventory_out):
    inventory.csv   one row per inner zip (volume)
    members.csv     one row per file inside any volume (name, size) -- for pattern spotting
    problems.txt    unreadable zips, empty volumes, nested zips, non-.txt members
and prints a summary.

Stdlib only.
"""
import csv
import io
import os
import re
import sys
import zipfile
from collections import Counter


def iter_inner_zips(source):
    """Yield (inner_name, zipfile.ZipFile or Exception) for each inner zip."""
    if os.path.isdir(source):
        for root, _dirs, files in os.walk(source):
            for fn in sorted(files):
                if fn.lower().endswith(".zip"):
                    p = os.path.join(root, fn)
                    try:
                        yield os.path.relpath(p, source), zipfile.ZipFile(p)
                    except Exception as e:  # noqa: BLE001
                        yield os.path.relpath(p, source), e
    else:
        with zipfile.ZipFile(source) as outer:
            for info in outer.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".zip"):
                    continue
                try:
                    data = outer.read(info)  # one inner zip at a time
                    yield info.filename, zipfile.ZipFile(io.BytesIO(data))
                except Exception as e:  # noqa: BLE001
                    yield info.filename, e


PAGE_RE = re.compile(r"(\d+)\.txt$", re.I)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    source = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "hathi_inventory_out"
    if not os.path.exists(source):
        sys.exit(f"Source not found: {source}\nFirst argument must be the big .zip (or a folder of .zip files).")
    if os.path.isfile(source) and not zipfile.is_zipfile(source):
        sys.exit(f"Not a zip file: {source}")
    if os.path.abspath(source) == os.path.abspath(out_dir):
        sys.exit("Source and output folder are the same path. Usage: python -I hathi_inventory.py <big.zip> [out_dir]")
    os.makedirs(out_dir, exist_ok=True)

    inv_rows, problems = [], []
    pat_counter = Counter()
    n_files_dist = Counter()

    with open(os.path.join(out_dir, "members.csv"), "w", newline="", encoding="utf-8") as mf:
        mw = csv.writer(mf)
        mw.writerow(["volume", "member", "bytes"])
        for i, (name, z) in enumerate(iter_inner_zips(source), 1):
            if isinstance(z, Exception):
                problems.append(f"UNREADABLE {name}: {z!r}")
                inv_rows.append([name, "", 0, 0, 0, "", "", "", "unreadable"])
                continue
            with z:
                infos = [x for x in z.infolist() if not x.is_dir()]
                txts = [x for x in infos if x.filename.lower().endswith(".txt")]
                others = [x for x in infos if not x.filename.lower().endswith(".txt")]
                for x in infos:
                    mw.writerow([name, x.filename, x.file_size])
                # numbered page files? (typical Hathi: 00000001.txt ...)
                nums = sorted(int(m.group(1)) for x in txts if (m := PAGE_RE.search(os.path.basename(x.filename))))
                paged = len(nums) == len(txts) and len(txts) > 1
                gaps = 0
                if paged:
                    gaps = (nums[-1] - nums[0] + 1) - len(nums)
                total = sum(x.file_size for x in txts)
                # shape of the first filename with digits collapsed, e.g. "########.txt"
                shape = re.sub(r"\d", "#", os.path.basename(txts[0].filename)) if txts else ""
                pat_counter[("paged" if paged else "single/other", shape)] += 1
                n_files_dist[len(txts)] += 1
                notes = []
                if not txts:
                    notes.append("no_txt")
                    problems.append(f"NO TXT {name}: members={[x.filename for x in infos][:5]}")
                if others:
                    notes.append(f"non_txt:{len(others)}")
                    if any(x.filename.lower().endswith(".zip") for x in others):
                        problems.append(f"NESTED ZIP inside {name}")
                if paged and gaps:
                    notes.append(f"page_gaps:{gaps}")
                if txts and total < 20_000:
                    notes.append("tiny")
                inv_rows.append([
                    name, os.path.splitext(os.path.basename(name))[0], len(txts), total,
                    gaps, "paged" if paged else "single/other", shape,
                    txts[0].filename if txts else "", ";".join(notes),
                ])
            if i % 100 == 0:
                print(f"  ...{i} volumes", file=sys.stderr)

    with open(os.path.join(out_dir, "inventory.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["inner_zip", "id_guess", "n_txt", "txt_bytes", "page_gaps", "layout",
                    "filename_shape", "first_member", "notes"])
        w.writerows(inv_rows)
    with open(os.path.join(out_dir, "problems.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(problems) + ("\n" if problems else ""))

    if not inv_rows:
        print("\nNo inner .zip files found in the source. If this is a folder, check it contains .zip files;"
              "\nif it is a zip, its contents may be one level deeper (extract once, then point at the folder).")
        return
    good = [r for r in inv_rows if r[8] != "unreadable"]
    sizes = sorted(r[3] for r in good)
    print(f"\nVolumes: {len(inv_rows)}   unreadable: {len(inv_rows) - len(good)}   problems logged: {len(problems)}")
    if sizes:
        q = lambda p: sizes[int(p * (len(sizes) - 1))]  # noqa: E731
        print(f"Text bytes per volume: min {sizes[0]:,}  p10 {q(.1):,}  median {q(.5):,}  p90 {q(.9):,}  max {sizes[-1]:,}")
    print("Layouts / filename shapes (count):")
    for (layout, shape), c in pat_counter.most_common(8):
        print(f"  {c:5d}  {layout:13s} {shape}")
    print("Files-per-volume (most common):", n_files_dist.most_common(6))
    print(f"\nWrote {out_dir}/inventory.csv, members.csv, problems.txt")


if __name__ == "__main__":
    main()
