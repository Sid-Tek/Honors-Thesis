"""
Checks how many Gutenberg entries each candidate author has.
Prints total count + up to 10 titles, so you can eyeball which are
real distinct novels vs. duplicates/translations/short-story collections.

TIP: search on bare surnames where possible (e.g. "Balzac" not "Honore
de Balzac") -- Gutendex's search requires every word you give it to
match, and accented names (Honoré, Émile, etc.) typed without the accent
will silently under-match and give you a near-zero, misleading count.

Requires: pip install requests
Run:      python CheckAuthorCounts_Gutenburg.py
"""

import requests
import time

GUTENDEX_URL = "https://gutendex.com/books"

CANDIDATES = [
    # Safe anchors -- likely 20+
    "Balzac", "Dumas", "Verne", "Zola", "Trollope", "Dickens",
    "Wallace", "Oliphant", "Yonge", "Braddon", "Baum", "Alger", "Henty",

    # Prolific but likely under 20 -- worth checking
    "Tolstoy", "Turgenev", "Tagore",
    "Zangwill", "Cather",
    "Chopin", "Gilman",

    # Likely 1-5 novels only -- historically important, lower the bar to include
    "Oscar Wilde"
]

for name in CANDIDATES:
    try:
        resp = requests.get(GUTENDEX_URL, params={"search": name, "languages": "en"}, timeout=30)
        data = resp.json()
        titles = [b["title"] for b in data["results"]]
        print(f"\n{name}: {data['count']} entries")
        for t in titles[:10]:
            print(f"   - {t}")
    except requests.RequestException as e:
        print(f"\n{name}: request failed ({e})")
    time.sleep(0.5)