"""
Parity check: do loader.py / windows.py reproduce the ORIGINAL voice_pipeline numbers?

Runs a few real novels through both the original code (voice_pipeline/) and the
copies in this folder and compares:
  1. the 24 window features (original features.extract_features vs windows.py),
  2. the per-window dialogue share (original dialogue_share_check.py vs windows.py),
  3. where the narrative starts (original front_matter.py vs loader.py).
The original files are only read, never changed.

Run (from the Opening_Register_Test folder):
    python test_parity.py
    python test_parity.py --original-dir ..\\voice_pipeline --corpus-dir ..\\voice_pipeline\\real_corpus
    python test_parity.py --books Dickens_Oliver_1839.txt Eliot_Adam_1859.txt
"""
import argparse
import os
import re
import sys
from bisect import bisect_left

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BOOKS = ['Dickens_Oliver_1839.txt', 'Eliot_Adam_1859.txt', 'Hardy_Tess_1891.txt',
                 'Kipling_Kim_1901.txt', 'Woolf_Lighthouse_1927.txt', 'Stevenson_Treasure_1883.txt']


def load_original_dialogue(original_dir, windowing):
    """Pull the dialogue functions out of the original script WITHOUT importing it
    (it imports half the pipeline). Executes only its parser block."""
    with open(os.path.join(original_dir, 'dialogue_share_check.py'), encoding='utf-8') as f:
        lines = f.read().split('\n')
    start = next(i for i, ln in enumerate(lines) if ln.startswith('MIN_MARKS_PER_10K'))
    end = next(i for i, ln in enumerate(lines) if ln.startswith('def _book('))
    ns = {'re': re, 'bisect_left': bisect_left, 'WORD_RE': windowing.WORD_RE}
    exec('\n'.join(lines[start:end]), ns)
    return ns


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('Run (from')[0],
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--original-dir', default=os.path.join(HERE, '..', 'voice_pipeline'))
    ap.add_argument('--corpus-dir', default=None, help='default: <original-dir>/real_corpus')
    ap.add_argument('--books', nargs='*', default=DEFAULT_BOOKS)
    args = ap.parse_args()
    orig = os.path.abspath(args.original_dir)
    corpus = args.corpus_dir or os.path.join(orig, 'real_corpus')

    sys.path.insert(0, HERE)
    import config                                   # noqa: E402
    import loader                                   # noqa: E402
    import windows as new                           # noqa: E402
    sys.path.insert(1, orig)
    import windowing as ow                          # noqa: E402  (original)
    import features as of                           # noqa: E402  (original)
    import front_matter as ofm                      # noqa: E402  (original)
    assert ow.__file__.startswith(orig) and of.__file__.startswith(orig), 'original modules not picked up'
    od = load_original_dialogue(orig, ow)

    all_ok = True
    print(f'{"book":28s} {"features":>9s} {"dialogue":>9s}  {"original cut":>12s} {"loader cut":>11s}  note')
    for fn in args.books:
        path = os.path.join(corpus, fn)
        if not os.path.exists(path):
            print(f'{fn:28s} missing in {corpus}')
            all_ok = False
            continue
        text, o_start, o_method, raw = ofm.load_trimmed(path, fn)

        # ---- 1. features, on the ORIGINAL's trimmed text so only the feature code differs
        o_wins = ow.local_windows(ow.parse_sentences(text), 300)
        o_X = np.array([[of.extract_features(w)[f] for f in of.FEATURE_NAMES] for w in o_wins])
        _, res = new.process_book((fn, text))
        feat_ok = (new.FEATURE_NAMES == list(of.FEATURE_NAMES) and o_X.shape == res['features'].shape
                   and np.allclose(o_X, res['features'], rtol=1e-9, atol=1e-12))

        # ---- 2. dialogue share, same text
        sents = ow.parse_sentences(text)
        style, n_dq, n_sq = od['detect_style'](text)
        inw = od['dialogue_words'](sents, style)
        shares, pos = [], 0
        for w in o_wins:
            k = len(w)
            tot = sum(s.n_words for s in sents[pos:pos + k])
            shares.append(sum(inw[pos:pos + k]) / max(tot, 1))
            pos += k
        dlg_ok = (style == res['style'] and len(shares) == len(res['dialogue'])
                  and np.allclose(shares, res['dialogue'], rtol=1e-9, atol=1e-12))

        # ---- 3. narrative start, loader vs original (original has a hand-typed table for some books)
        with open(path, encoding='utf-8', errors='ignore') as f:
            raw2 = f.read().lstrip('﻿')
        l_start, l_method = loader.find_narrative_start(raw2)
        manual = fn in ofm.MANUAL_START
        same = (l_start == o_start)
        note = ''
        if manual:
            note = 'original uses a hand-typed start' + ('' if same else '; loader would flag or cut elsewhere')
        elif not same:
            note = 'CUT DIFFERS'
        print(f'{fn:28s} {"OK" if feat_ok else "MISMATCH":>9s} {"OK" if dlg_ok else "MISMATCH":>9s}  '
              f'{o_start:>12,d} {l_start:>11,d}  {note}  [{style}, {len(o_wins)} windows]')
        all_ok &= feat_ok and dlg_ok and (same or manual)
    print('\nPARITY OK' if all_ok else '\nPARITY FAILED: see above')
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
