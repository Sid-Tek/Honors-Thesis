"""
Command-line driver that ties the whole emergence pipeline together and
runs it against a real folder of novels.

Expects a folder of flat text files named:

    Author_Title.txt

(the stylo-package convention -- author is everything before the first
underscore in the filename. Same convention your earlier k-NN pipeline
used for the "100 English Novels" corpus.)

Usage:
    python run_pipeline.py /path/to/corpus_folder
    python run_pipeline.py /path/to/corpus_folder --lengths 500,1000,1500,2000,3000
    python run_pipeline.py /path/to/corpus_folder --n 3 --min-count 5 --folds 5
    python run_pipeline.py /path/to/corpus_folder --out results.png

python run_pipeline.py 100_english_novels-master/corpus --lengths 500,1000,1500,2000,3000 --n 3 --folds 3 --min-novels-per-author 3
python run_pipeline.py 100_english_novels-master/corpus --lengths 500,1000,1500,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,21000,22000,23000,24000,25000,26000,27000,28000,29000,30000,31000,32000,33000 --n 3 --folds 3 --min-novels-per-author 3 --max-authors 6


python run_pipeline.py 100_english_novels-master/corpus --lengths 500,1000,1500,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,21000,22000,23000,24000,25000,26000,27000,28000,29000,30000,31000,32000,33000 --n 3 --folds 3 --min-novels-per-author 3 --max-authors 6 --epsilon 0.06

python run_pipeline.py 100_english_novels-master/corpus --lengths 500,1000,1500,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,21000,22000,23000,24000,25000,26000,27000,28000,29000,30000,31000,32000,33000 --n 3 --folds 3 --min-novels-per-author 3 --max-authors 6 --epsilon-sweep 0.05,0.075,0.1 --out plateau_sweep_test.png

python run_pipeline.py 100_english_novels-master/corpus --lengths 1000,5000,10000,15000,20000,25000,30000 --n 3 --folds 3 --min-novels-per-author 3 --max-authors 6 --epsilon-sweep 0.05,0.075,0.1 --out plateau_sweep_test.png

python run_pipeline.py 100_english_novels-master/corpus --lengths 500,1000,1500,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,21000,22000,23000,24000,25000,26000,27000,28000,29000,30000,31000,32000,33000 --n 3 --folds 3 --min-novels-per-author 3 --max-authors 6 --repeats 100 --epsilon 0.06 --out plateau_repeats100.png

python run_pipeline.py Gutenburg_corpus --lengths 1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,21000,22000,23000,24000,25000,26000,27000,28000,29000,30000,31000,32000,33000 --n 3 --folds 3 --min-novels-per-author 3 --max-authors 10 --repeats 20 --epsilon 0.06 --out Gutenburg_plateau_repeats20_max10_FineQuick.png
USE 18 NEXT TIME FOR FULL ANALYSIS


--- --mode position: fixed window size(s), swept across different STARTING
POSITIONS in each novel instead of growing from the start. Tests whether
accuracy depends on how much text you have (a sample-size artifact) or on
how far into the narrative you are (real emergence) -- see the position_sweep.py
module docstring for the full rationale.

python run_pipeline.py 100_english_novels-master/corpus --mode position --position-sizes 500,1000,2000,4000,8000 --offsets 0,5000,10000,15000,20000,25000 --n 3 --folds 3 --min-novels-per-author 3 --max-authors 6 --ceiling-length 30000 --out position_sweep_test.png

python run_pipeline.py Gutenburg_corpus --mode position --position-sizes 250,500,1000,2000,4000,8000 --offsets 0,5000,10000,15000,20000,25000 --n 3 --folds 3 --min-novels-per-author 3 --max-authors 10 --repeats 20 --ceiling-length 30000 --alpha 0.05 --out Gutenburg_position_sweep.png

--- --trim-front-matter: detect and strip table-of-contents/chapter-heading
contamination from the start of each novel before running EITHER mode. Only
flagged novels are touched. See scan_front_matter.py's corpus scan output
first to sanity-check what will get trimmed.

python run_pipeline.py 100_english_novels-master/corpus --mode position --trim-front-matter --position-sizes 500,1000,2000,4000,8000 --offsets 0,4000,8000,12000,16000 --n 3 --folds 3 --min-novels-per-author 3 --repeats 20 --ceiling-length 25000 --alpha 0.05 --out position_sweep_trimmed.png


python run_pipeline.py 100_english_novels-master/corpus --mode position --trim-front-matter --position-sizes 500,1000,2000,4000,8000 --offsets 0,4000,8000,12000,16000 --n 3 --folds 3 --min-novels-per-author 3 --repeats 20 --ceiling-length 25000 --alpha 0.05 --out position_sweep_trimmed.png


python run_pipeline.py 100_english_novels-master/corpus --mode position --position-sizes 500,1000,2000,4000,8000 --offsets 0,4000,8000,12000,16000 --n 3 --folds 3 --min-novels-per-author 3 --repeats 20 --ceiling-length 25000 --alpha 0.05 --out position_sweep_untrimmed_full.png

python run_pipeline.py Gutenburg_corpus --mode position --position-sizes 1000,5000,10000,16000 --offsets 0,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000 --n 3 --folds 3 --min-novels-per-author 3 --max-authors 10 --repeats 20 --ceiling-length 120000 --alpha 0.05 --out position_sweep_100k.png

"""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from window_sweep import run_window_sweep
from plot_emergence import plot_emergence
from position_sweep import (
    run_position_sweep_multi_size,
    find_fast_acting_size,
    print_fast_acting_summary,
)
from plot_position_sweep import plot_position_sweep
from trim_front_matter import list_corpus_filenames, compute_trim_map, apply_trim


def load_corpus(folder):
    """
    Walks `folder` for .txt files named Author_Title.txt, returns
    (documents, labels) lists in matching order.
    """
    folder = Path(folder)
    documents, labels = [], []
    for path in sorted(folder.glob("*.txt")):
        author = path.stem.split("_")[0]
        text = path.read_text(encoding="utf-8", errors="replace")
        documents.append(text)
        labels.append(author)
    return documents, labels


def filter_by_author_count(documents, labels, min_count):
    """
    Drops any author with fewer than `min_count` novels in the corpus.

    This matters because n_folds-fold cross-validation can't split an
    author with fewer than n_folds novels into non-empty folds -- that
    failure has nothing to do with window length, so it's better to
    catch it once here than have every window in the sweep report it.

    Returns (filtered_documents, filtered_labels, dropped_authors) where
    dropped_authors is a sorted list of (author, count) tuples for
    authors that got removed.
    """
    counts = Counter(labels)
    keep_authors = {a for a, c in counts.items() if c >= min_count}
    dropped_authors = sorted(
        ((a, c) for a, c in counts.items() if a not in keep_authors),
        key=lambda pair: pair[1]
    )

    kept = [(d, l) for d, l in zip(documents, labels) if l in keep_authors]
    if not kept:
        return [], [], dropped_authors
    documents_f, labels_f = zip(*kept)
    return list(documents_f), list(labels_f), dropped_authors


def limit_authors(documents, labels, max_authors, seed=0):
    """
    Randomly selects `max_authors` distinct authors (fixed seed, so the
    same subset is picked every time you rerun) and drops all novels by
    every other author.

    This is a diagnostic tool: with a fixed number of novels per author,
    fewer authors makes the classification problem easier, since the
    SVM only has to draw a handful of boundaries instead of dozens. It
    helps tell apart two different explanations for a flat accuracy
    curve -- "voice genuinely saturates fast" vs. "too many authors for
    too few examples each is swamping the signal, regardless of window
    length."

    Returns (filtered_documents, filtered_labels, selected_authors).
    """
    unique_authors = sorted(set(labels))
    if max_authors >= len(unique_authors):
        return documents, labels, unique_authors  # nothing to limit

    rng = random.Random(seed)
    selected = set(rng.sample(unique_authors, max_authors))

    kept = [(d, l) for d, l in zip(documents, labels) if l in selected]
    documents_f, labels_f = zip(*kept) if kept else ([], [])
    return list(documents_f), list(labels_f), sorted(selected)


def main():
    ap = argparse.ArgumentParser(
        description="Run the full authorial-voice emergence pipeline on a corpus folder."
    )
    ap.add_argument("corpus_folder", help="Folder of Author_Title.txt novel files")
    ap.add_argument("--mode", choices=["length", "position"], default="length",
                     help="'length' (default): original cumulative-from-start sweep. "
                          "'position': NEW -- fixed window size(s) swept across different "
                          "starting offsets in each novel, to separate real narrative-time "
                          "emergence from a sample-size/noise-reduction artifact. See "
                          "position_sweep.py for the rationale.")
    ap.add_argument("--lengths", default="500,1000,1500,2000,3000,4000,6000",
                     help="[--mode length] Comma-separated window lengths in characters")
    ap.add_argument("--position-sizes", default="500,1000,2000,4000,8000",
                     help="[--mode position] Comma-separated FIXED window sizes (chars) "
                          "to test, e.g. 500,1000,2000,4000,8000")
    ap.add_argument("--offsets", default="0,5000,10000,15000,20000,25000",
                     help="[--mode position] Comma-separated starting offsets (chars) to "
                          "test at each fixed size, e.g. 0,5000,10000,15000,20000,25000")
    ap.add_argument("--alpha", type=float, default=0.05,
                     help="[--mode position] Significance threshold for the "
                          "position-flatness test. p_value >= alpha counts as flat -- "
                          "no detectable slope between accuracy and offset (default 0.05)")
    ap.add_argument("--ceiling-length", type=int, default=None,
                     help="[--mode position] Optional character length to use as each "
                          "line's reference 'ceiling' accuracy, measured with one "
                          "cumulative-from-start run at this length (reuses --mode length's "
                          "machinery). When given, a size only counts as fast-acting if it's "
                          "BOTH position-flat AND within --ceiling-fraction of this ceiling -- "
                          "otherwise a line flat at a low, useless accuracy would still count. "
                          "Omit to check flatness only.")
    ap.add_argument("--ceiling-fraction", type=float, default=0.9,
                     help="[--mode position] Fraction of the ceiling accuracy a size must "
                          "reach, if --ceiling-length is given (default 0.9)")
    ap.add_argument("--n", type=int, default=3,
                     help="n-gram length (default 3, matching the paper)")
    ap.add_argument("--min-count", type=int, default=5,
                     help="Minimum vocabulary frequency (default 5, matching the paper)")
    ap.add_argument("--folds", type=int, default=5, help="Cross-validation folds")
    ap.add_argument("--repeats", type=int, default=1,
                 help="Number of times to repeat k-fold cross-validation per "
                      "window, each with a different random fold split, averaging "
                      "the results (default 1, i.e. no repeating). Doesn't add new "
                      "novels -- smooths out the luck of any one fold split, which "
                      "matters most on small corpora where accuracy can only land "
                      "on coarse rungs (e.g. multiples of 1/18).")
    ap.add_argument("--min-novels-per-author", type=int, default=None,
                     help="Drop authors with fewer novels than this before running "
                          "(default: same as --folds, since that's the actual "
                          "requirement for cross-validation to work)")
    ap.add_argument("--max-authors", type=int, default=None,
                     help="Randomly keep only this many authors (diagnostic tool -- "
                          "see if a flat accuracy curve is caused by too many "
                          "authors swamping the signal, not by voice saturating fast)")
    ap.add_argument("--seed", type=int, default=0,
                     help="Random seed, used for --max-authors selection and for "
                          "cross-validation shuffling (default 0, for reproducibility)")
    ap.add_argument("--epsilon", type=float, default=0.01,
                     help="Plateau detection sensitivity: minimum accuracy gain "
                          "between windows to still count as climbing (default 0.01). "
                          "Raise this on small test sets, where accuracy can only move "
                          "in coarse steps (e.g. 1/18 ~= 0.056), or 1%% looks like a "
                          "plateau even mid-climb. Ignored if --epsilon-sweep is given.")
    ap.add_argument("--epsilon-sweep", default=None,
                     help="Comma-separated list of epsilon values (e.g. 0.06,0.08,0.1). "
                          "Runs the (slow) window sweep ONCE, then produces one plot per "
                          "epsilon value -- useful for checking how sensitive the plateau "
                          "diamonds are to your epsilon choice without recomputing "
                          "accuracies from scratch each time. Overrides --epsilon and --out.")
    ap.add_argument("--out", default="emergence_plot.png", help="Output plot filename")
    ap.add_argument("--trim-front-matter", action="store_true",
                     help="Detect and strip table-of-contents/chapter-heading "
                          "contamination from the start of each novel before "
                          "running the sweep. Only flagged novels are touched; "
                          "clean ones pass through unchanged. See "
                          "scan_front_matter.py / trim_front_matter.py.")
    ap.add_argument("--trim-scan-chars", type=int, default=2000,
                     help="[--trim-front-matter] Initial front-window size for "
                          "detection; doubles automatically (up to a limit) if "
                          "contamination runs close to the window edge, so a "
                          "too-small window doesn't silently under-trim "
                          "(default 2000)")
    ap.add_argument("--trim-threshold-ratio", type=float, default=8.0,
                     help="[--trim-front-matter] Flag threshold for the "
                          "front-window-vs-rest-of-novel n-gram rate ratio "
                          "(default 8.0)")
    ap.add_argument("--trim-padding", type=int, default=200,
                     help="[--trim-front-matter] Extra characters trimmed past "
                          "the detected contamination, then snapped forward to "
                          "the next paragraph/line break (default 200)")
    args = ap.parse_args()

    min_novels = args.min_novels_per_author if args.min_novels_per_author is not None else args.folds

    print(f"Loading corpus from {args.corpus_folder} ...")
    documents, labels = load_corpus(args.corpus_folder)
    print(f"Loaded {len(documents)} novels across {len(set(labels))} authors.")

    if len(documents) == 0:
        print("No .txt files found -- check the folder path.")
        sys.exit(1)

    if args.trim_front_matter:
        # Must run BEFORE any author-count/max-authors filtering below,
        # while documents is still in the same order as the full,
        # unfiltered corpus_folder listing that list_corpus_filenames()
        # produces -- otherwise filenames and documents would drift out
        # of alignment with each other.
        print(f"\nScanning for front-matter contamination (table of contents, "
              f"chapter-heading blocks) to trim before analysis...")
        filenames = list_corpus_filenames(args.corpus_folder)
        trim_map = compute_trim_map(documents, filenames,
                                     initial_scan_chars=args.trim_scan_chars,
                                     threshold_ratio=args.trim_threshold_ratio,
                                     padding=args.trim_padding)
        if trim_map:
            documents = apply_trim(documents, filenames, trim_map)
            print(f"Trimmed {len(trim_map)}/{len(documents)} novel(s).")
        else:
            print("No front-matter contamination detected -- nothing trimmed.")

    documents, labels, dropped = filter_by_author_count(documents, labels, min_novels)
    if dropped:
        print(f"\nDropping {len(dropped)} author(s) with fewer than {min_novels} "
              f"novel(s) (needed for --folds {args.folds}):")
        for author, count in dropped:
            print(f"  {author}: {count} novel(s)")
    print(f"\nProceeding with {len(documents)} novels across "
          f"{len(set(labels))} authors.")

    if args.max_authors is not None:
        documents, labels, selected = limit_authors(documents, labels,
                                                      args.max_authors, seed=args.seed)
        print(f"\n--max-authors {args.max_authors}: keeping only "
              f"{len(selected)} randomly selected author(s) (seed={args.seed}):")
        for author in selected:
            print(f"  {author}")
        print(f"\nProceeding with {len(documents)} novels across "
              f"{len(set(labels))} authors.")

    print()
    if len(set(labels)) < 2:
        print("Fewer than 2 authors remain after filtering -- nothing to classify. "
              "Try lowering --folds, --min-novels-per-author, or raising --max-authors.")
        sys.exit(1)

    if args.mode == "length":
        run_length_mode(args, documents, labels)
    else:
        run_position_mode(args, documents, labels)


def run_length_mode(args, documents, labels):
    """Original cumulative-from-start sweep -- unchanged behavior."""
    window_lengths = [int(x) for x in args.lengths.split(",")]

    results = run_window_sweep(documents, labels, window_lengths,
                            n=args.n, min_count=args.min_count,
                            n_folds=args.folds, seed=args.seed,
                            n_repeats=args.repeats, verbose=True)

    if args.epsilon_sweep:
        # The window sweep (the expensive part -- fitting classifiers at
        # every window length) already happened above and doesn't depend
        # on epsilon at all. Epsilon only affects which points plot_emergence
        # marks as plateaus, so re-plotting per epsilon here is cheap --
        # no need to re-run run_window_sweep for each value.
        epsilons = [float(e) for e in args.epsilon_sweep.split(",")]
        out_base = Path(args.out)
        print()
        for eps in epsilons:
            out_path = out_base.with_name(f"{out_base.stem}_eps{eps}{out_base.suffix}")
            fig = plot_emergence(results, window_lengths, labels, epsilon=eps)
            fig.savefig(out_path, dpi=150)
            print(f"Saved plot (epsilon={eps}) to {out_path}")
    else:
        fig = plot_emergence(results, window_lengths, labels, epsilon=args.epsilon)
        fig.savefig(args.out, dpi=150)
        print(f"\nSaved plot to {args.out}")


def run_position_mode(args, documents, labels):
    """
    NEW: fixed window size(s), swept across different starting offsets in
    each novel. See position_sweep.py for the full rationale.
    """
    window_sizes = [int(x) for x in args.position_sizes.split(",")]
    offsets = [int(x) for x in args.offsets.split(",")]

    print(f"\nRunning position sweep: sizes={window_sizes}, offsets={offsets}")
    results_by_size = run_position_sweep_multi_size(
        documents, labels, window_sizes, offsets,
        n=args.n, min_count=args.min_count, n_folds=args.folds,
        seed=args.seed, n_repeats=args.repeats, verbose=True,
    )

    ceiling = None
    if args.ceiling_length is not None:
        print(f"\nMeasuring ceiling accuracy at {args.ceiling_length} characters "
              f"(cumulative-from-start, for comparison)...")
        ceiling_results = run_window_sweep(
            documents, labels, [args.ceiling_length],
            n=args.n, min_count=args.min_count, n_folds=args.folds,
            seed=args.seed, n_repeats=args.repeats, verbose=False,
        )
        ceiling = {line: vals[0] for line, vals in ceiling_results.items()}
        print("Ceiling accuracies:")
        for line, acc in ceiling.items():
            print(f"  {line:14s} {acc:.3f}" if not np.isnan(acc) else f"  {line:14s} n/a")

    fast_acting = find_fast_acting_size(window_sizes, offsets, results_by_size,
                                         alpha=args.alpha, ceiling=ceiling,
                                         ceiling_fraction=args.ceiling_fraction)
    print_fast_acting_summary(fast_acting, ceiling=ceiling)

    fig = plot_position_sweep(results_by_size, offsets, alpha=args.alpha)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {args.out}")


if __name__ == "__main__":
    main()