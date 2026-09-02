"""
python Vector.py sample.txt --model en_core_web_lg

voice_vector.py

Extracts a "fast-tier" voice vector from the first N characters of a text file,
based on four structurally fast-emerging, genre-independent stylistic dimensions
established for the Variables in Voice Emergence project:

  1. Punctuation patterns          -- habitual, present from sentence one
  2. Sentence length & rhythm      -- converges within a paragraph or two
  3. Syntactic packaging           -- clause-level word order / dependency structure
  4. Sentence type distribution    -- simple/compound/complex/compound-complex
                                       (second-tier: needs ~10+ sentences to settle)

All features are computed from a spaCy dependency parse -- deterministic,
reproducible, no LLM involved.

Requires:
    pip install spacy
    python -m spacy download en_core_web_trf   # best accuracy (needs spacy-transformers)
    # or, faster / no extra dependency:
    python -m spacy download en_core_web_lg

Usage:
    python voice_vector.py path/to/text.txt
    python Vector.py path/to/text.txt --n 3000
    python voice_vector.py path/to/text.txt --n 3000 --model en_core_web_lg --json out.json
"""

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean, pstdev

import spacy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PUNCT_CHARS = {
    "comma": ",",
    "semicolon": ";",
    "colon": ":",
    "exclamation": "!",
    "question": "?",
}
DASH_CHARS = {"-", "\u2013", "\u2014"}  # hyphen, en dash, em dash
ELLIPSIS_PATTERN = re.compile(r"\.\.\.|\u2026")

# Dependency labels whose presence marks a subordinate clause head
SUBORD_CLAUSE_DEPS = {"advcl", "ccomp", "xcomp", "relcl", "acl"}


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_text(path: Path, n_chars: int) -> str:
    """Read a text file and return roughly the first n_chars, extended to the
    next sentence-ending punctuation so spaCy isn't handed a truncated
    final sentence."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if len(raw) <= n_chars:
        return raw
    window = raw[:n_chars]
    tail = raw[n_chars : n_chars + 300]
    m = re.search(r"[.!?]", tail)
    if m:
        window += tail[: m.end()]
    return window


def get_nlp(model_name: str):
    try:
        return spacy.load(model_name)
    except OSError:
        sys.exit(
            f"spaCy model '{model_name}' isn't downloaded.\n"
            f"Run: python -m spacy download {model_name}"
        )


# ---------------------------------------------------------------------------
# 1. Punctuation patterns
# ---------------------------------------------------------------------------

def punctuation_features(doc) -> dict:
    n_words = sum(1 for t in doc if not t.is_space and not t.is_punct)
    n_words = max(n_words, 1)
    text = doc.text

    counts = {name: text.count(ch) for name, ch in PUNCT_CHARS.items()}
    counts["dash"] = sum(text.count(ch) for ch in DASH_CHARS)
    counts["ellipsis"] = len(ELLIPSIS_PATTERN.findall(text))

    # rate per 100 words normalizes for passage length so N can vary
    return {f"{k}_per_100w": round(v / n_words * 100, 3) for k, v in counts.items()}


# ---------------------------------------------------------------------------
# 2. Sentence length & rhythm
# ---------------------------------------------------------------------------

def sentence_length_features(doc) -> dict:
    lengths = [
        sum(1 for t in sent if not (t.is_punct or t.is_space))
        for sent in doc.sents
    ]
    lengths = [l for l in lengths if l > 0]

    if len(lengths) < 2:
        return {
            "mean_sentence_length": float(lengths[0]) if lengths else 0.0,
            "sentence_length_stdev": 0.0,
            "sentence_length_cv": 0.0,
            "mean_consecutive_delta": 0.0,
        }

    mu = mean(lengths)
    sigma = pstdev(lengths)
    deltas = [abs(lengths[i] - lengths[i - 1]) for i in range(1, len(lengths))]

    return {
        "mean_sentence_length": round(mu, 3),
        "sentence_length_stdev": round(sigma, 3),
        "sentence_length_cv": round(sigma / mu, 3) if mu else 0.0,  # length-independent rhythm
        "mean_consecutive_delta": round(mean(deltas), 3),           # short/long alternation
    }


# ---------------------------------------------------------------------------
# 3. Syntactic packaging
# ---------------------------------------------------------------------------

def _token_depth(tok) -> int:
    """Hops from tok up to its sentence root."""
    depth = 0
    seen = set()
    cur = tok
    while cur.head is not cur and cur.i not in seen:
        seen.add(cur.i)
        cur = cur.head
        depth += 1
    return depth


def syntactic_packaging_features(doc) -> dict:
    arc_lengths = []
    sent_max_depths = []

    for sent in doc.sents:
        tokens = [t for t in sent if not t.is_space]
        if not tokens:
            continue
        for t in tokens:
            if t.head is not t:
                arc_lengths.append(abs(t.i - t.head.i))
        sent_max_depths.append(max(_token_depth(t) for t in tokens))

    return {
        "mean_dependency_arc_length": round(mean(arc_lengths), 3) if arc_lengths else 0.0,
        "mean_parse_tree_depth": round(mean(sent_max_depths), 3) if sent_max_depths else 0.0,
    }


# ---------------------------------------------------------------------------
# 4. Sentence type distribution
# ---------------------------------------------------------------------------

def classify_sentence(sent) -> str:
    """Rule-based simple/compound/complex/compound-complex classifier
    from independent- and subordinate-clause counts in the dependency parse.

    Heuristic, not a full grammar -- adequate for a fast-tier structural
    signal, not intended as a gold-standard syntactic classifier."""
    root = sent.root
    independent = 1  # the root clause itself
    subordinate = 0

    for t in sent:
        if t.dep_ == "conj" and t.head == root and t.pos_ in {"VERB", "AUX"}:
            independent += 1
        elif t.dep_ in SUBORD_CLAUSE_DEPS:
            subordinate += 1

    if independent > 1 and subordinate > 0:
        return "compound-complex"
    if independent > 1:
        return "compound"
    if subordinate > 0:
        return "complex"
    return "simple"


def sentence_type_features(doc) -> dict:
    labels = [classify_sentence(sent) for sent in doc.sents]
    n = len(labels) or 1
    dist = {
        "simple": labels.count("simple") / n,
        "compound": labels.count("compound") / n,
        "complex": labels.count("complex") / n,
        "compound_complex": labels.count("compound-complex") / n,
    }
    return {f"pct_{k}": round(v, 3) for k, v in dist.items()}


# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------

def extract_voice_vector(text: str, nlp) -> dict:
    doc = nlp(text)
    vector = {}
    vector.update(punctuation_features(doc))
    vector.update(sentence_length_features(doc))
    vector.update(syntactic_packaging_features(doc))
    vector.update(sentence_type_features(doc))
    vector["_n_sentences_parsed"] = sum(1 for _ in doc.sents)
    vector["_n_chars_parsed"] = len(text)
    return vector


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Extract a fast-tier voice vector from a text file.")
    ap.add_argument("file", type=Path, help="Path to a .txt file")
    ap.add_argument("--n", type=int, default=1500, help="Number of characters to analyze (default: 3000)")
    ap.add_argument("--model", default="en_core_web_trf", help="spaCy model to use")
    ap.add_argument("--json", type=Path, default=None, help="Optional path to write result as JSON")
    args = ap.parse_args()

    if not args.file.exists():
        sys.exit(f"File not found: {args.file}")

    text = load_text(args.file, args.n)
    nlp = get_nlp(args.model)
    vector = extract_voice_vector(text, nlp)

    print(f"\nVoice vector for: {args.file.name}  (first ~{args.n} chars)\n" + "-" * 50)
    for k, v in vector.items():
        print(f"{k:28s} {v}")

    if args.json:
        args.json.write_text(json.dumps(vector, indent=2))
        print(f"\nSaved to {args.json}")


if __name__ == "__main__":
    main()