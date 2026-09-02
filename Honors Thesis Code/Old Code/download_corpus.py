"""
Builds an `Author_Title.txt` novel corpus from Project Gutenberg, using the
free Gutendex API (https://gutendex.com) -- no API key, no scraping.

For each author in AUTHORS below, this:
  1. Searches Gutendex for that author's books.
  2. Keeps only English-language entries with a plain-text download format.
  3. Double-checks the author's surname actually appears in the book's
     author field (Gutendex's search also loosely matches titles).
  4. Filters out anything under MIN_CHARS characters -- a rough way to drop
     short stories, poems, and essays and keep novel-length texts. This is
     a heuristic, not a guarantee -- skim the output folder afterward and
     prune anything that slipped through (story collections, duplicate
     volumes of a multi-part novel, non-fiction, etc).
  5. Strips Project Gutenberg's standard boilerplate header/footer.
  6. Saves as corpus/Author_Title.txt -- matching the "everything before
     the first underscore is the author label" convention your
     run_pipeline.py's load_corpus() already expects. No code changes
     needed on that end.

Requires: pip install requests
Run:      python download_corpus.py
"""

import re
import time
import unicodedata
from pathlib import Path

import requests

GUTENDEX_URL = "https://gutendex.com/books"
OUTPUT_DIR = Path("corpus")
MIN_CHARS = 150_000          # rough novel-length floor (~25k+ words)
MAX_PER_AUTHOR = 20          # stop after this many saved novels for one author
REQUEST_DELAY = 1.0          # seconds between downloads -- be polite to servers

# (search query for Gutendex, filename label to use before the underscore)
# Using bare surnames for both -- avoids the accent/multi-word matching
# problem full names can trigger (see: Balzac).
AUTHORS = [
    # Safe anchors -- likely 20+
    ("Balzac", "Balzac"),
    ("Dumas", "Dumas"),
    ("Verne", "Verne"),
    ("Zola", "Zola"),
    ("Trollope", "Trollope"),
    ("Dickens", "Dickens"),
    ("Wallace", "Wallace"),
    ("Oliphant", "Oliphant"),
    ("Yonge", "Yonge"),
    ("Braddon", "Braddon"),
    ("Baum", "Baum"),
    ("Alger", "Alger"),
    ("Henty", "Henty"),

    # Prolific but likely under 20 -- worth checking
    ("Tolstoy", "Tolstoy"),
    ("Turgenev", "Turgenev"),
    ("Tagore", "Tagore"),
    ("Zangwill", "Zangwill"),
    ("Cather", "Cather"),
    ("Chopin", "Chopin"),
    ("Gilman", "Gilman"),
    ("Ibsen", "Ibsen"),

    # Likely 1-5 novels only
    ("Wilde", "Wilde"),
]

START_MARKER = re.compile(
    r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)
END_MARKER = re.compile(
    r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)


def safe_filename(text):
    """Strips accents/punctuation so a title is safe to use in a filename."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "-", text.strip())
    return text[:60]  # keep filenames from getting absurdly long


def strip_boilerplate(text):
    """Cuts Project Gutenberg's standard header/footer, keeping just the book."""
    start_match = START_MARKER.search(text)
    end_match = END_MARKER.search(text)
    start = start_match.end() if start_match else 0
    end = end_match.start() if end_match else len(text)
    return text[start:end].strip()


def find_text_format(formats):
    """Picks a plain-text download URL out of Gutendex's `formats` dict."""
    for key, url in formats.items():
        if key.startswith("text/plain") and "images" not in key:
            return url
    return None


def author_matches(surname, authors_field):
    """Checks the target surname actually appears in the book's author list."""
    return any(surname.lower() in a.get("name", "").lower() for a in authors_field)


def fetch_books_for_author(query):
    """Pages through Gutendex results for a given author search query."""
    books = []
    url = GUTENDEX_URL
    params = {"search": query, "languages": "en"}
    while url:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        books.extend(data.get("results", []))
        url = data.get("next")
        params = None  # `next` already has query params baked in
    return books


def download_author(query, label):
    print(f"\n=== {query} ===")
    try:
        books = fetch_books_for_author(query)
    except requests.RequestException as e:
        print(f"  Skipping: search failed ({e})")
        return 0

    saved = 0
    for book in books:
        if saved >= MAX_PER_AUTHOR:
            print(f"  Reached {MAX_PER_AUTHOR}-novel cap, moving on.")
            break

        title = book.get("title", "untitled")
        if not author_matches(label, book.get("authors", [])):
            continue

        text_url = find_text_format(book.get("formats", {}))
        if not text_url:
            continue

        try:
            resp = requests.get(text_url, timeout=60)
            resp.raise_for_status()
            resp.encoding = resp.encoding or "utf-8"
            raw = resp.text
        except requests.RequestException as e:
            print(f"  Skipping '{title}': download failed ({e})")
            continue

        body = strip_boilerplate(raw)
        if len(body) < MIN_CHARS:
            print(f"  Skipping '{title}': only {len(body)} chars, below novel-length floor")
            continue

        filename = f"{label}_{safe_filename(title)}.txt"
        out_path = OUTPUT_DIR / filename
        out_path.write_text(body, encoding="utf-8")
        print(f"  Saved: {filename} ({len(body):,} chars)")
        saved += 1
        time.sleep(REQUEST_DELAY)

    return saved


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    total = 0
    for query, label in AUTHORS:
        total += download_author(query, label)
    print(f"\nDone. {total} novels saved.")
    print(f"Corpus saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()