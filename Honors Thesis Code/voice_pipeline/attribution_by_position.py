"""
Verification test 1: ATTRIBUTION BY POSITION.

Question: does an author's text become easier to attribute to them as a novel
goes on -- and does it keep getting easier after the shared opening register
has faded? Unlike the pooled-curve onset, this cannot be produced by general
narrative drift: a passage can only be attributed to the right author if it
carries something specific to that author.

Design
------
* Authors: every author with at least K books (default 3) whose books are all
  long enough for the position range tested (shorter ones are dropped and
  listed). Candidates = those authors, so chance = 1 / number of authors.
* Folds: fold f tests every author's f-th book (by year) and trains on that
  author's other books. Every candidate therefore has the same amount of
  training text. Each fold is a different held-out book.
* Test passages: consecutive passages of PASSAGE words (default 2,000) from
  the start of the held-out book (positions 0, 2,000, 4,000, ... up to
  MAX_WORDS, default 30,000). Front matter is trimmed first (front_matter.py),
  so position 0 is the first word of the narrative.
* Method: Burrows' Delta on the most frequent words (default 500) -- each
  word's frequency is z-scored across the candidates' profiles, and the
  passage goes to the author with the smallest mean absolute z-difference.
  Cosine Delta (cosine of the z-score vectors) is run alongside as a check.
  The word list is chosen from the training books of each fold only.

Training conditions (what the author profile is built from)
-----------------------------------------------------------
  all         every word of the training books           (the practical case)
  all_noreg   as 'all', but the opening-register function words found in
              step 1 (the / of / in / and / was / to / but ...) are removed
              from the word list
  noopen      training books without their first 3,000 words
  matched     only text from the SAME position in the training books
              (MATCH_WIDTH words around the test passage's position)
  unmatched   an equal amount of text from random positions after word 3,000
              that do not overlap the matched range
'matched' versus 'unmatched' is the fair test of "is there author signal at
this position": same amount of training text, differing only in whether it
comes from the same part of the book.

What each outcome means
-----------------------
  accuracy low in the opening, higher later (condition 'all')
        the opening is harder to attribute. This alone is expected from the
        shared register, whether or not voice is present in the opening.
  accuracy keeps rising AFTER the register fades (early -> middle -> later)
        author-specific signal strengthens with position: the claim.
  noopen beats all
        the opening is misleading training text for an author profile.
  matched beats unmatched in the opening
        there IS author signal in the opening, visible when you compare like
        with like (voice underneath the register).
  the same patterns without the register words (all_noreg)
        the result is not just the register function words.
Uncertainty: authors are resampled (a 95% interval from 2,000 draws), because
an author's three books are not independent.

Caveats (read before interpreting)
----------------------------------
  * Opening passages also have less dialogue (dialogue_share_check.py), and
    dialogue changes which common words appear; some of "opening is harder"
    may be that, not voice.
  * Burrows' Delta is a standard baseline, not the best attribution method;
    it is used because the claim should not depend on a tuned classifier.
  * 2,000-word passages and ~30 authors give roughly 90 test passages per
    position: single-position accuracy is noisy, the grouped comparisons are
    the ones to read.

Run (from the voice_pipeline folder):
    python attribution_by_position.py
    python attribution_by_position.py --max-words 20000       # keeps more authors if some books are short
    python attribution_by_position.py --corpus-dir some_other_corpus --out output_attrib_other
    # held-out replication (after prepare_heldout_corpus.py):
    python attribution_by_position.py --corpus-dir heldout_corpus --pretrimmed --out output_attrib_heldout_all

Outputs (output_attrib/ unless --out): attribution_report.md, trials.csv,
  accuracy_by_position.csv, contrasts.csv, plots/attribution_by_position.png
(book_counts_*.pkl caches the word counts; delete it if the corpus changes.)
"""
import argparse
import os
import pickle
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from windowing import WORD_RE
from front_matter import load_trimmed

FNAME_RE = re.compile(r'^([A-Za-z]+)_(.+)_(\d{4})\.txt$')
REGISTER_CSV = os.path.join('output_register', 'register_tests.csv')
GAP_TEST_CSV = os.path.join('output_onset', 'gap_test.csv')
OPEN_CUT = 3000          # words dropped from training in the 'noopen' / 'unmatched' conditions
ALPHA_Q = 0.05

CONDS = ['all', 'all_noreg', 'noopen', 'matched', 'unmatched']
INK, MUTED, GRID = '#333333', '#8a8a8a', '#e6e6e6'
COLORS = {'all': '#2a78d6', 'all_noreg': '#1b4f8f', 'noopen': '#2a9d8f', 'matched': '#eb6834',
          'unmatched': '#b5b5b5'}


# ------------------------------------------------------------ corpus ----
def discover_books(corpus_dir, k_books):
    by = defaultdict(list)
    for fn in sorted(os.listdir(corpus_dir)):
        m = FNAME_RE.match(fn)
        if m:
            a, _, y = m.groups()
            by[a].append((int(y), fn))
    return {a: [fn for _, fn in sorted(v)][:k_books] for a, v in by.items() if len(v) >= k_books}


def _count_book(args):
    path, fn, limit_words, block, pretrimmed = args
    if pretrimmed:                       # file already starts at the narrative (prepare_heldout_corpus.py)
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            text = re.sub(r'\s+', ' ', f.read().lstrip('\ufeff')).strip()
    else:
        text, _, _, _ = load_trimmed(path, fn)
    toks = WORD_RE.findall(text.lower())
    n = len(toks)
    nb = min(limit_words // block, n // block)
    return fn, n, Counter(toks), [Counter(toks[i * block:(i + 1) * block]) for i in range(nb)]


def load_counts(corpus_dir, files, limit_words, block, cache, pretrimmed=False):
    sig = (tuple(sorted(files)), limit_words, block, bool(pretrimmed))
    if os.path.exists(cache):
        with open(cache, 'rb') as f:
            d = pickle.load(f)
        if d.get('sig') == sig:
            print(f'Loading cached word counts from {cache}', flush=True)
            return d['raw']
    print(f'Counting words in {len(files)} books (one-time; cached afterwards)...', flush=True)
    t0 = time.time()
    raw = {}
    jobs = [(os.path.join(corpus_dir, fn), fn, limit_words, block, pretrimmed) for fn in files]
    with ProcessPoolExecutor() as ex:
        for i, (fn, n, full, blocks) in enumerate(ex.map(_count_book, jobs), 1):
            raw[fn] = (n, full, blocks)
            if i % 10 == 0 or i == len(jobs):
                print(f'  [{i}/{len(jobs)}] {time.time()-t0:.0f}s', flush=True)
    with open(cache, 'wb') as f:
        pickle.dump({'sig': sig, 'raw': raw}, f)
    return raw


def build_matrices(raw, files, n_vocab, nb):
    tot = Counter()
    for fn in files:
        tot.update(raw[fn][1])
    vocab = [w for w, _ in tot.most_common(n_vocab)]
    idx = {w: i for i, w in enumerate(vocab)}
    full, blocks, ntok = {}, {}, {}
    for fn in files:
        n, fc, bl = raw[fn]
        v = np.zeros(len(vocab), np.int64)
        for w, c in fc.items():
            j = idx.get(w)
            if j is not None:
                v[j] = c
        B = np.zeros((nb, len(vocab)), np.int32)
        for b, cnt in enumerate(bl[:nb]):
            for w, c in cnt.items():
                j = idx.get(w)
                if j is not None:
                    B[b, j] = c
        full[fn], blocks[fn], ntok[fn] = v, B, n
    return vocab, full, blocks, ntok


# ----------------------------------------------------------- classifier ----
def scores(test_rel, prof_rel, method):
    """Higher = closer. test_rel (T x M), prof_rel (A x M) relative frequencies."""
    mu = prof_rel.mean(axis=0)
    sd = prof_rel.std(axis=0, ddof=1)
    sd = np.where(sd < 1e-12, 1e-12, sd)
    zp, zt = (prof_rel - mu) / sd, (test_rel - mu) / sd
    if method == 'delta':
        return -np.abs(zt[:, None, :] - zp[None, :, :]).mean(axis=2)
    num = zt @ zp.T
    den = np.linalg.norm(zt, axis=1)[:, None] * np.linalg.norm(zp, axis=1)[None, :]
    return num / np.where(den < 1e-12, 1e-12, den)


def true_ranks(S):
    """S (A x A): row i = passage by author i, columns = candidates. Rank 1 = best."""
    d = np.diag(S)
    return 1 + (S > d[:, None]).sum(axis=1)


# ------------------------------------------------------------ main run ----
def run_trials(authors, books, vocab, full, blocks, ntok, args, reg_idx):
    A, K = len(authors), args.books
    B, P, W, M = args.block, args.passage, args.match_width, args.mfw
    n0 = OPEN_CUT // B
    nb_total = next(iter(blocks.values())).shape[0]
    starts = list(range(0, args.max_words, P))
    rows = []
    t0 = time.time()
    for f in range(K):
        train = [[books[a][j] for j in range(K) if j != f] for a in authors]
        test = [books[a][f] for a in authors]
        tr_full = np.array([sum(full[b] for b in tb) for tb in train])                    # A x V
        tr_tok = np.array([sum(ntok[b] for b in tb) for tb in train], float)
        tr_first = np.array([sum(blocks[b][:n0].sum(axis=0) for b in tb) for tb in train])  # A x V
        order = np.argsort(-tr_full.sum(axis=0), kind='stable')
        mfw = order[:M]
        mfw_noreg = np.array([j for j in order if j not in reg_idx][:M])
        rel = lambda counts, tok, m: counts[:, m] / tok[:, None]
        prof_all = rel(tr_full, tr_tok, mfw)
        prof_allnr = rel(tr_full, tr_tok, mfw_noreg)
        prof_noopen = rel(tr_full - tr_first, tr_tok - OPEN_CUT * (K - 1), mfw)
        for si, s in enumerate(starts):
            b0, b1 = s // B, (s + P) // B
            tcount = np.array([blocks[t][b0:b1].sum(axis=0) for t in test], float)         # A x V
            a0 = max(0, s - (W - P) // 2)
            m0, m1 = a0 // B, (a0 + W) // B
            rng = np.random.default_rng([args.seed, f, si])
            allowed = np.array([b for b in range(n0, nb_total) if b < m0 or b >= m1])
            mcount = np.zeros_like(tr_full, float)
            ucount = np.zeros_like(tr_full, float)
            for i, tb in enumerate(train):
                for b in tb:
                    mcount[i] += blocks[b][m0:m1].sum(axis=0)
                    pick = rng.choice(allowed, size=m1 - m0, replace=False)
                    ucount[i] += blocks[b][pick].sum(axis=0)
            mtok = np.full(A, float(W * (K - 1)))
            profiles = {'all': (prof_all, tcount[:, mfw] / P),
                        'all_noreg': (prof_allnr, tcount[:, mfw_noreg] / P),
                        'noopen': (prof_noopen, tcount[:, mfw] / P),
                        'matched': (rel(mcount, mtok, mfw), tcount[:, mfw] / P),
                        'unmatched': (rel(ucount, mtok, mfw), tcount[:, mfw] / P)}
            for cond, (prof, trel) in profiles.items():
                for method in ('delta', 'cosine'):
                    rk = true_ranks(scores(trel, prof, method))
                    for i, a in enumerate(authors):
                        rows.append((a, test[i], f, s, cond, method, int(rk[i] == 1), 1.0 / rk[i], int(rk[i])))
        print(f'  fold {f + 1}/{K} done ({time.time()-t0:.0f}s)', flush=True)
    return pd.DataFrame(rows, columns=['author', 'book', 'fold', 'pos', 'cond', 'method', 'correct', 'rr', 'rank'])


# ------------------------------------------------------------ analysis ----
def make_cube(df, value, conds, groups, authors):
    aidx = {a: i for i, a in enumerate(authors)}
    cidx = {c: i for i, c in enumerate(conds)}
    S = np.zeros((len(authors), len(conds), len(groups)))
    N = np.zeros_like(S)
    ai = df['author'].map(aidx).values
    ci = df['cond'].map(cidx).values
    v, pos = df[value].values, df['pos'].values
    for g, (_, fn) in enumerate(groups):
        m = fn(pos)
        np.add.at(S, (ai[m], ci[m], g), v[m])
        np.add.at(N, (ai[m], ci[m], g), 1)
    return S, N


def boot_cube(S, N, rng, n_boot):
    A = S.shape[0]
    W = rng.multinomial(A, np.full(A, 1.0 / A), size=n_boot).astype(float)
    num = np.einsum('ba,acg->bcg', W, S)
    den = np.einsum('ba,acg->bcg', W, N)
    point = S.sum(axis=0) / np.where(N.sum(axis=0) == 0, np.nan, N.sum(axis=0))
    return point, num / np.where(den == 0, np.nan, den)


def verdict(lo, hi):
    if np.isnan(lo):
        return 'n/a'
    return 'higher' if lo > 0 else ('lower' if hi < 0 else 'inconclusive')


def group_defs(edges, max_words):
    e = [0] + list(edges) + [max_words]
    names = ['opening', 'early', 'middle', 'later'][:len(e) - 1]
    groups = [(f'{n} ({e[i] // 1000}-{e[i + 1] // 1000}k)', (lambda p, lo=e[i], hi=e[i + 1]: (p >= lo) & (p < hi)))
              for i, n in enumerate(names)]
    groups.append(('all positions', lambda p: p >= 0))
    return groups


def contrasts_for(groups):
    gn = [g[0] for g in groups]
    op, ea, mi, la, ov = gn[0], gn[1], gn[2], gn[3], gn[-1]
    return [
        ('P1  opening is harder than the middle (all training)', ('all', mi), ('all', op)),
        ('P2  accuracy keeps rising after the register: later vs early (all training)', ('all', la), ('all', ea)),
        ('    middle vs early (all training)', ('all', mi), ('all', ea)),
        ('    later vs middle (all training)', ('all', la), ('all', mi)),
        ('P3  dropping openings from training helps: all positions', ('noopen', ov), ('all', ov)),
        ('    ... for the opening passage', ('noopen', op), ('all', op)),
        ('    ... for later passages', ('noopen', la), ('all', la)),
        ('P4  position-matched training beats equal-size unmatched: opening', ('matched', op), ('unmatched', op)),
        ('    ... early', ('matched', ea), ('unmatched', ea)),
        ('    ... middle', ('matched', mi), ('unmatched', mi)),
        ('    ... later', ('matched', la), ('unmatched', la)),
        ('P5  rise after the register without the register words: later vs early', ('all_noreg', la), ('all_noreg', ea)),
        ('    removing the register words changes opening accuracy by', ('all_noreg', op), ('all', op)),
        ('    removing the register words changes later accuracy by', ('all_noreg', la), ('all', la)),
    ]


def analyse(trials, authors, args, rng):
    groups = group_defs(args.edges, args.max_words)
    gnames = [g[0] for g in groups]
    out_tab, out_con = [], []
    for method in ('delta', 'cosine'):
        df = trials[trials['method'] == method]
        for value, vname in (('correct', 'accuracy'), ('rr', 'mean reciprocal rank')):
            S, N = make_cube(df, value, CONDS, groups, authors)
            point, boot = boot_cube(S, N, rng, args.boot)
            lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
            for ci, c in enumerate(CONDS):
                for gi, g in enumerate(gnames):
                    out_tab.append({'method': method, 'measure': vname, 'cond': c, 'group': g,
                                    'value': point[ci, gi], 'ci_lo': lo[ci, gi], 'ci_hi': hi[ci, gi],
                                    'n_passages': int(N[:, ci, gi].sum())})
            for name, (c1, g1), (c2, g2) in contrasts_for(groups):
                i1, j1, i2, j2 = CONDS.index(c1), gnames.index(g1), CONDS.index(c2), gnames.index(g2)
                d0 = point[i1, j1] - point[i2, j2]
                db = boot[:, i1, j1] - boot[:, i2, j2]
                l, h = np.nanpercentile(db, [2.5, 97.5])
                out_con.append({'method': method, 'measure': vname, 'contrast': name.strip(),
                                'label': name, 'diff': d0, 'ci_lo': l, 'ci_hi': h, 'verdict': verdict(l, h)})
    return pd.DataFrame(out_tab), pd.DataFrame(out_con), gnames


def per_position(trials, authors, args, rng):
    starts = sorted(trials['pos'].unique())
    groups = [(str(s), (lambda p, s=s: p == s)) for s in starts]
    rows = []
    for method in ('delta', 'cosine'):
        df = trials[trials['method'] == method]
        S, N = make_cube(df, 'correct', CONDS, groups, authors)
        point, boot = boot_cube(S, N, rng, min(args.boot, 1000))
        lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
        for ci, c in enumerate(CONDS):
            for gi, s in enumerate(starts):
                rows.append({'method': method, 'cond': c, 'pos': s, 'accuracy': point[ci, gi],
                             'ci_lo': lo[ci, gi], 'ci_hi': hi[ci, gi], 'n': int(N[:, ci, gi].sum())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------- outputs ----
def style_axes(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)


def plot(pp, n_cand, reg_end, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    d = pp[pp['method'] == 'delta']
    for ax, conds, title in ((axes[0], ['all', 'noopen', 'all_noreg'], 'Trained on whole books'),
                             (axes[1], ['matched', 'unmatched'], 'Trained on equal amounts of text: same position vs. random later position')):
        for c in conds:
            x = d[d['cond'] == c]
            ax.plot(x['pos'] / 1000, x['accuracy'], color=COLORS[c], lw=2, label=c)
            ax.fill_between(x['pos'] / 1000, x['ci_lo'], x['ci_hi'], color=COLORS[c], alpha=0.12, linewidth=0)
        ax.axhline(1.0 / n_cand, color=MUTED, lw=1, ls=':')
        if reg_end:
            ax.axvline(reg_end / 1000, color='#eb6834', lw=1, ls='--')
        ax.set_xlabel('start of the 2,000-word test passage (thousand words into the book)', fontsize=8, color=INK)
        ax.set_title(title, loc='left', fontsize=9, color=INK)
        ax.legend(frameon=False, fontsize=8)
        style_axes(ax)
    axes[0].set_ylabel('share of passages attributed to the right author', fontsize=8, color=INK)
    fig.text(0.01, 0.01, 'Burrows\' Delta; bands = 95% over authors; dotted = chance; dashed = end of the opening register (step 3b)',
             fontsize=7, color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fmt(v, ci=None):
    if np.isnan(v):
        return '-'
    s = f'{v:.2f}'
    return s if ci is None else f'{s} ({ci[0]:.2f}-{ci[1]:.2f})'


def write_report(tab, con, gnames, authors, dropped, args, n_trials, path):
    n_cand = len(authors)
    L = ['# Attribution by position (verification test 1)', '',
         f'Run {time.strftime("%Y-%m-%d")}. Corpus: `{args.corpus_dir}`. {n_cand} candidate authors '
         f'(chance {1 / n_cand:.1%}); {args.books} folds (each fold tests every author\'s next book, trained on '
         f'the other {args.books - 1}); {args.passage:,}-word passages from the first {args.max_words:,} words; '
         f'{args.mfw} most frequent words; {n_trials:,} classifications. Intervals: authors resampled, '
         f'{args.boot:,} draws.']
    if dropped:
        L.append(f'Left out because a book is shorter than the range tested: {", ".join(sorted(dropped))}.')
    L.append('')
    for method, mname in (('delta', "Burrows' Delta (primary)"), ('cosine', 'Cosine Delta (check)')):
        t = tab[(tab['method'] == method) & (tab['measure'] == 'accuracy')]
        L += [f'## Accuracy by position: {mname}', '',
              '| training | ' + ' | '.join(gnames) + ' |', '|---|' + '---:|' * len(gnames)]
        for c in CONDS:
            row = []
            for g in gnames:
                r = t[(t['cond'] == c) & (t['group'] == g)].iloc[0]
                row.append(fmt(r['value'], (r['ci_lo'], r['ci_hi'])))
            L.append(f'| {c} | ' + ' | '.join(row) + ' |')
        L.append('')
    L += ['## Predictions', '',
          'Difference = first minus second; "higher" means the 95% interval is above 0, "lower" below 0.', '']
    for method, mname in (('delta', "Burrows' Delta (primary)"), ('cosine', 'Cosine Delta (check)')):
        c = con[(con['method'] == method) & (con['measure'] == 'accuracy')]
        m = con[(con['method'] == method) & (con['measure'] == 'mean reciprocal rank')].set_index('contrast')
        L += [f'### {mname}', '', '| contrast | accuracy difference | verdict | mean reciprocal rank difference | verdict |',
              '|---|---:|---|---:|---|']
        for _, r in c.iterrows():
            mr = m.loc[r['contrast']]
            L.append(f"| {r['label']} | {r['diff']:+.3f} ({r['ci_lo']:+.3f} to {r['ci_hi']:+.3f}) | {r['verdict']} | "
                     f"{mr['diff']:+.3f} ({mr['ci_lo']:+.3f} to {mr['ci_hi']:+.3f}) | {mr['verdict']} |")
        L.append('')
    p = con[(con['method'] == 'delta') & (con['measure'] == 'accuracy')].set_index('label')
    def v(prefix):
        return p[p.index.str.startswith(prefix)]['verdict'].iloc[0]
    L += ['## Plain reading (primary method, accuracy)', '',
          f'* P1, the opening is harder than the middle: **{v("P1")}**.',
          f'* P2, accuracy keeps rising after the register has faded (later vs. early): **{v("P2")}**. '
          'This is the one that bears on "voice builds after the opening"; P1 alone is expected from the register.',
          f'* P3, leaving the openings out of training helps: **{v("P3")}** (all positions).',
          f'* P4, same-position training beats equal-size random-position training in the opening: **{v("P4")}**. '
          '"higher" means the opening carries author-specific signal that shows up when compared like with like.',
          f'* P5, the rise after the register survives without the register words: **{v("P5")}**.', '',
          'Read these together; none alone settles the hypothesis. A single "inconclusive" is not a failure to '
          'replicate, it is a lack of power at this sample size.', '']
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L))


# ---------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('Run (from')[0], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--corpus-dir', default='real_corpus')
    ap.add_argument('--out', default='output_attrib')
    ap.add_argument('--books', type=int, default=3, help='books per author used (first K by year); also the number of folds')
    ap.add_argument('--passage', type=int, default=2000)
    ap.add_argument('--max-words', type=int, default=30000, help='test passages start before this position')
    ap.add_argument('--match-width', type=int, default=4000, help="words of same-position training text per book ('matched')")
    ap.add_argument('--block', type=int, default=500)
    ap.add_argument('--mfw', type=int, default=500)
    ap.add_argument('--vocab', type=int, default=2000)
    ap.add_argument('--edges', type=int, nargs=3, default=[2000, 6000, 12000],
                    help='group boundaries (word positions) for opening / early / middle / later')
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20261010)
    ap.add_argument('--authors', nargs='*', default=None, help='only these authors (e.g. --authors Baum Gilman)')
    ap.add_argument('--pretrimmed', action='store_true',
                    help='files already start at the narrative (heldout_corpus/ from prepare_heldout_corpus.py): skip front-matter trimming')
    args = ap.parse_args()
    B, P, W = args.block, args.passage, args.match_width
    assert P % B == 0 and W % B == 0 and (W - P) % (2 * B) == 0 and args.max_words % P == 0, \
        'passage, match-width and max-words must be multiples of the block; (match-width - passage) a multiple of 2 x block'
    t0 = time.time()
    os.makedirs(os.path.join(args.out, 'plots'), exist_ok=True)
    limit = args.max_words + W

    books_all = discover_books(args.corpus_dir, args.books)
    if args.authors:
        books_all = {a: bs for a, bs in books_all.items() if a in set(args.authors)}
    if not books_all:
        raise SystemExit(f'No author with {args.books}+ books found in {args.corpus_dir}/ (expected Author_Title_YYYY.txt).')
    files = [fn for bs in books_all.values() for fn in bs]
    raw = load_counts(args.corpus_dir, files, limit, B, os.path.join(args.out, f'book_counts_b{B}_l{limit}.pkl'),
                      pretrimmed=args.pretrimmed)
    need = limit
    dropped = [a for a, bs in books_all.items() if any(raw[fn][0] < need for fn in bs)]
    authors = sorted(a for a in books_all if a not in dropped)
    books = {a: books_all[a] for a in authors}
    print(f'{len(authors)} candidate authors' + (f'; left out (a book is shorter than {need:,} words): '
                                                 f'{", ".join(sorted(dropped))}' if dropped else '') + '.', flush=True)
    if len(authors) < 4:
        raise SystemExit('Fewer than 4 authors left; lower --max-words or --match-width.')
    if len(authors) < 8:
        print(f'NOTE: only {len(authors)} candidate authors (chance {1/len(authors):.0%}); intervals will be wide.', flush=True)

    keep = [fn for a in authors for fn in books[a]]
    vocab, full, blocks, ntok = build_matrices({fn: raw[fn] for fn in keep}, keep, args.vocab, limit // B)
    reg_words = []
    if os.path.exists(REGISTER_CSV):
        r = pd.read_csv(REGISTER_CSV)
        r = r[(r['q_1500'] < ALPHA_Q) & r['feature'].str.startswith('fw_')]
        reg_words = [f[3:] for f in r['feature']]
    reg_idx = {vocab.index(w) for w in reg_words if w in vocab}
    print(f'Opening-register words removed in "all_noreg": {", ".join(reg_words) if reg_words else "none (register_tests.csv not found)"}',
          flush=True)

    print('Classifying passages...', flush=True)
    trials = run_trials(authors, books, vocab, full, blocks, ntok, args, reg_idx)
    trials.to_csv(os.path.join(args.out, 'trials.csv'), index=False)

    rng = np.random.default_rng(args.seed)
    tab, con, gnames = analyse(trials, authors, args, rng)
    tab.to_csv(os.path.join(args.out, 'accuracy_by_group.csv'), index=False)
    con.to_csv(os.path.join(args.out, 'contrasts.csv'), index=False)
    pp = per_position(trials, authors, args, rng)
    pp.to_csv(os.path.join(args.out, 'accuracy_by_position.csv'), index=False)
    reg_end = None
    if os.path.exists(GAP_TEST_CSV):
        g = pd.read_csv(GAP_TEST_CSV)
        reg_end = float(g.loc[g['quantity'] == 'register end (combined)', 'estimate_words'].iloc[0])
    plot(pp, len(authors), reg_end, os.path.join(args.out, 'plots', 'attribution_by_position.png'))
    write_report(tab, con, gnames, authors, dropped, args, len(trials), os.path.join(args.out, 'attribution_report.md'))

    t = tab[(tab['method'] == 'delta') & (tab['measure'] == 'accuracy') & (tab['cond'] == 'all')]
    print('\nAccuracy by position (Delta, trained on whole books):')
    print(t[['group', 'value', 'ci_lo', 'ci_hi', 'n_passages']].to_string(index=False, float_format=lambda v: f'{v:.2f}'))
    print(f'\nOutputs written to {args.out}/ (report: attribution_report.md) -- total {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
