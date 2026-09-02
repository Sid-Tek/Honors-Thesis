"""
Turns scan_front_matter.py's detection into an actual fix: for each
novel, figure out where its front-matter contamination (table of
contents, chapter-heading block) ends, and trim just that novel to
start after it. Clean novels are left completely untouched.

Two things this adds beyond a single scan_document() call:

  1. ADAPTIVE WINDOW: a fixed scan window can under-shoot -- if the
     contamination runs right up to the edge of the window, you can't
     tell whether it ends there or just past it (this happened for 3
     novels in the corpus scan: Blackmore_Lorna, Trollope_Phineas,
     Braddon_Quest, all flagged almost exactly at the 2000-char edge).
     This module re-scans with a doubled window whenever that happens,
     so the reported cutoff is trustworthy instead of a guess.

  2. CLEAN CUT POINT: the raw detection gives a character position
     inside the contamination, not a good place to actually cut. This
     adds a padding buffer and then snaps forward to the next
     paragraph/line break, so trimming lands on a real text boundary
     instead of mid-sentence.

Known limitation this does NOT fix: for novels where chapter headings
repeat throughout the whole body (not just in a front-of-book table of
contents -- e.g. Blackmore_Erema and Blackmore_Springhaven both still
had nonzero "rest" counts), trimming the front block removes the big
concentrated spike but small versions of the same artifact remain at
each later chapter break. This module only addresses the front-of-book
concentration, which is what was distorting the position sweep's
offset-0 measurements.

Usage (as a library -- see run_pipeline.py's --trim-front-matter flag
for the CLI-integrated version):

    from trim_front_matter import list_corpus_filenames, compute_trim_map, apply_trim

    filenames = list_corpus_filenames(corpus_folder)
    trim_map = compute_trim_map(documents, filenames)
    documents = apply_trim(documents, filenames, trim_map)
"""

from pathlib import Path

from scan_front_matter import scan_document


def list_corpus_filenames(folder):
    """
    Sorted list of filenames in `folder`, in the SAME order
    run_pipeline.load_corpus() produces its documents list (both sort
    the same glob), so zip(documents, list_corpus_filenames(folder))
    lines up correctly -- as long as this is called on the FULL,
    unfiltered corpus folder, before any author-count/max-authors
    filtering has dropped documents out of that order.
    """
    return [path.name for path in sorted(Path(folder).glob("*.txt"))]


def snap_to_boundary(text, pos, prefer="\n\n"):
    """
    Moves `pos` forward to the next paragraph break (or, failing that,
    the next line break) at or after `pos`, so a trim cut lands on a
    real text boundary instead of mid-sentence. Falls back to `pos`
    unchanged if no boundary is found before the end of the text.
    """
    idx = text.find(prefer, pos)
    if idx != -1:
        return idx + len(prefer)
    idx = text.find("\n", pos)
    if idx != -1:
        return idx + 1
    return pos


def adaptive_scan(text, initial_scan_chars=2000, max_scan_chars=8000, n=3,
                   threshold_ratio=8.0, min_front_count=8, edge_margin=0.85):
    """
    Calls scan_document() with a growing window until the flagged
    contamination's last_position sits comfortably inside the window
    (below edge_margin * scan_chars) -- i.e. we're confident we've seen
    where it actually ends -- or the window hits max_scan_chars, or
    nothing gets flagged at all.

    Returns None if clean. Otherwise returns scan_document()'s result
    dict, with an added "uncertain" key (True if max_scan_chars was hit
    before the contamination cleared the edge margin -- meaning this
    cutoff may still be an underestimate).
    """
    scan_chars = initial_scan_chars
    while True:
        result = scan_document(text, scan_chars=scan_chars, n=n,
                                threshold_ratio=threshold_ratio,
                                min_front_count=min_front_count)
        if result is None:
            return None
        if result["last_position"] < edge_margin * scan_chars:
            result["uncertain"] = False
            return result
        if scan_chars >= max_scan_chars:
            result["uncertain"] = True
            return result
        scan_chars *= 2


def compute_trim_map(documents, filenames, initial_scan_chars=2000,
                      max_scan_chars=8000, n=3, threshold_ratio=8.0,
                      min_front_count=8, padding=200, verbose=True):
    """
    documents, filenames: matched lists (see list_corpus_filenames()).

    Returns dict {filename: cutoff_position} for every FLAGGED novel
    only -- clean novels simply aren't in the dict. cutoff_position is
    padded past the detected contamination and snapped to the next
    paragraph/line break.
    """
    trim_map = {}
    for text, filename in zip(documents, filenames):
        result = adaptive_scan(text, initial_scan_chars=initial_scan_chars,
                                max_scan_chars=max_scan_chars, n=n,
                                threshold_ratio=threshold_ratio,
                                min_front_count=min_front_count)
        if result is None:
            continue
        raw_cutoff = result["last_position"] + padding
        cutoff = snap_to_boundary(text, raw_cutoff)
        trim_map[filename] = cutoff
        if verbose:
            tag = "  (uncertain -- hit max scan width, may underestimate)" \
                if result["uncertain"] else ""
            print(f"  {filename:40s} trimming first {cutoff:5d} chars "
                  f"(detected {result['ngram']!r} ratio={result['ratio']:.1f}){tag}")
    return trim_map


def apply_trim(documents, filenames, trim_map):
    """
    Returns a NEW documents list with each flagged novel's front matter
    removed. Novels not in trim_map (i.e. not flagged) are returned
    completely unchanged -- same string, same object.
    """
    return [text[trim_map[filename]:] if filename in trim_map else text
            for text, filename in zip(documents, filenames)]