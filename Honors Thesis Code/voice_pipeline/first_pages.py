import csv, re, sys, collections
mn = {}
for r in csv.DictReader(open(sys.argv[1], encoding="utf-8")):
    m = re.search(r"(\d+)\.txt$", r["member"])
    if m:
        v = r["volume"]; n = int(m.group(1)); mn[v] = min(mn.get(v, n), n)
c = collections.Counter(mn.values())
print("volumes:", len(mn), " starting at page 1:", c.get(1, 0))
print("most common first-page numbers:", c.most_common(8))
