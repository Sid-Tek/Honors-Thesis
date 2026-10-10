"""
Verification plan A4: DIALOGUE-SHARE CHECK.

Step 1 guessed that the opening register (longer sentences, noun-phrase
function words, few verbs-of-speech) is descriptive scene-setting with little
dialogue, and the Sep 28 analysis flagged exclaim_rate / question_rate /
dash_rate as scene structure rather than voice. If dialogue is what changes
as a book gets going, the "voice onset" could just be the point where
dialogue picks up. This script measures the share of each window that is
inside quotation marks and asks four questions.

  1. Does dialogue share dip in the opening, and when does the dip end?
     (profile over position, authors as units; a fade-out end point like the
     register's, so the two can be compared.)
  2. When does dialogue share start to RISE, and is that at the same place as
     the voice onset? (paired bootstrap over authors: gap = voice onset minus
     dialogue onset; an interval around 0 means "same place".)
  3. Which of the 24 features move with dialogue inside a book? (detrended
     within-book correlation, averaged over books.)
  4. Do the accumulators, and the onset, survive taking dialogue out? Each
     author's feature curves are residualised on that author's dialogue-share
     curve (same books, same positions); the accumulator test and the onset
     are redone on the residuals.
     "Survives" is strong evidence that dialogue is not the explanation.
     "Disappears" is ambiguous: the feature may be driven by dialogue, or
     dialogue may simply rise with voice (a mediator, not a confound).

Detecting dialogue
------------------
Per book, the quote style is detected from the text itself:
  double  "..." or curly double quotes (the majority of the corpus)
  single  '...' or curly single quotes (British style); apostrophes are told
          apart from quotes by position (an opening quote starts a word, a
          closing one ends it) with a short list of elisions ('tis, 'em, ...)
  none    fewer than 10 quote marks per 10,000 words: no dialogue detectable.
          Those books are left out of every dialogue statistic (the plan
          expected about four).
Words inside quotes are counted with a running open/closed state across
sentences. Stray unmatched marks are the main failure mode, so a quote that
stays open for more than 300 words is closed by force. Treat the share as a
good relative measure (position, author, book), not an exact count.
Windows are exactly the 300-word windows of the rest of the pipeline.

Run (from the voice_pipeline folder, after steps 1-3):
    python dialogue_share_check.py --quick     # 2,000 permutations, 500 bootstraps
    python dialogue_share_check.py             # 20,000 permutations, 2,000 bootstraps
    python dialogue_share_check.py --rebuild   # re-detect dialogue (after changing the parser)

Outputs (output_verify/): dialogue_books.csv, dialogue_profile.csv,
  dialogue_feature_corr.csv, dialogue_report.md, plots/dialogue_profile.png
  (dialogue_share.pkl is the cache of per-window shares)
"""
import argparse
import os
import pickle
import re
import time
from bisect import bisect_left
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from features import FEATURE_NAMES
from windowing import parse_sentences, local_windows, WORD_RE
from front_matter import load_trimmed
from run_full_corpus import discover_authors, CORPUS_DIR
from accumulator_retest import build_curves
from verify_common import (
    OUT, COMPARISON_CSV, DIALOGUE_FEATURES, Cfg, load_features, test_curves, acc_dict, acc_from_comp,
    n_pairs, opening_residuals, estimate, fit_grand, fit_shape, shape_design, fade_grid, fade_design,
    fit_fade, pad_to, onset_words, check_baseline, style_axes, md_table, INK, MUTED, GRID, C_COR, C_RAW,
    C_BAND)

DIALOGUE_CSV_CACHE = os.path.join(OUT, 'dialogue_share.pkl')
GAP_TEST_CSV = os.path.join('output_onset', 'gap_test.csv')
WINDOW = 300
MIN_MARKS_PER_10K = 10     # fewer quote marks than this -> "no dialogue detectable"
MAX_OPEN_WORDS = 300       # a quote open longer than this is treated as a stray mark and closed

DQ_RE = re.compile('["\u201c\u201d]')
SQ_RE = re.compile("['\u2018\u2019]")
ELISION = r"(?:tis|twas|twere|twill|twould|em|cause|til|till|ud|ave|ere|alf|arry|ow|e|er|im)"
ELISION_RE = re.compile(ELISION + r'\b', re.I)
SQ_OPEN_RE = re.compile(r"(?<![\w'\u2019])'(?=[A-Za-z])(?!" + ELISION + r'\b)', re.I)
OPENERS = '([{\u2014-'
CLOSERS = '.,;:!?)]}\u2014-'
OPENERS_SQ = OPENERS + '"\u201c'


# ------------------------------------------------------- dialogue parser ----
def detect_style(text):
    """('double' | 'single' | 'none', n_double_marks, n_single_openers)."""
    n_words = max(len(text.split()), 1)
    n_dq = len(DQ_RE.findall(text))
    n_sq = text.count('\u2018') + len(SQ_OPEN_RE.findall(text))
    if n_dq / n_words * 1e4 >= MIN_MARKS_PER_10K and n_dq >= n_sq:
        return 'double', n_dq, n_sq
    if n_sq / n_words * 1e4 >= MIN_MARKS_PER_10K:
        return 'single', n_dq, n_sq
    return 'none', n_dq, n_sq


def double_events(raw):
    ev, L = [], len(raw)
    for m in DQ_RE.finditer(raw):
        i, ch = m.start(), m.group()
        if ch == '\u201c':
            ev.append((i, 'open'))
        elif ch == '\u201d':
            ev.append((i, 'close'))
        else:                                   # straight " : decide from the neighbours, else toggle
            prev = raw[i - 1] if i > 0 else ' '
            nxt = raw[i + 1] if i + 1 < L else ' '
            if (prev.isspace() or prev in OPENERS) and not nxt.isspace():
                ev.append((i, 'open'))
            elif not prev.isspace() and (nxt.isspace() or nxt in CLOSERS):
                ev.append((i, 'close'))
            else:
                ev.append((i, 'toggle'))
    return ev


def single_events(raw):
    ev, L = [], len(raw)
    for m in SQ_RE.finditer(raw):
        i, ch = m.start(), m.group()
        prev = raw[i - 1] if i > 0 else ' '
        nxt = raw[i + 1] if i + 1 < L else ' '
        if ch == '\u2018':
            ev.append((i, 'open'))
        elif nxt.isalnum():                     # inside a word (don't) or the start of one ('Hello)
            if ch == "'" and (prev.isspace() or prev in OPENERS_SQ) and not ELISION_RE.match(raw, i + 1):
                ev.append((i, 'open'))
        elif not prev.isspace():                # follows a word or punctuation, then a space/punctuation
            ev.append((i, 'close'))
    return ev


def dialogue_words(sents, style):
    """Words inside quotes, per sentence (same tokens as Sentence.n_words)."""
    if style == 'none':
        return [0] * len(sents)
    get_events = double_events if style == 'double' else single_events
    state, open_words, out = False, 0, []
    for s in sents:
        raw = s.raw
        starts = [m.start() for m in WORD_RE.finditer(raw)]
        count_in, last = 0, 0
        for idx, kind in get_events(raw):
            n_between = bisect_left(starts, idx) - bisect_left(starts, last)
            if state:
                count_in += n_between
            state = True if kind == 'open' else (False if kind == 'close' else not state)
            last = idx
        if state:
            count_in += len(starts) - bisect_left(starts, last)
        out.append(count_in)
        if state:
            open_words += s.n_words
            if open_words > MAX_OPEN_WORDS:     # stray unmatched mark: give up on this quote
                state, open_words = False, 0
        else:
            open_words = 0
    return out


def _book(fn):
    text, _, _, _ = load_trimmed(os.path.join(CORPUS_DIR, fn), fn)
    sents = parse_sentences(text)
    wins = local_windows(sents, WINDOW)
    style, n_dq, n_sq = detect_style(text)
    inw = dialogue_words(sents, style)
    shares, pos = [], 0
    for w in wins:
        k = len(w)
        tot = sum(s.n_words for s in sents[pos:pos + k])
        shares.append(sum(inw[pos:pos + k]) / max(tot, 1))
        pos += k
    n_words = sum(s.n_words for s in sents)
    return fn, {'style': style, 'shares': np.array(shares), 'n_double': n_dq, 'n_single_open': n_sq,
                'n_words': n_words}


def load_dialogue(by_author, rebuild=False):
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(DIALOGUE_CSV_CACHE) and not rebuild:
        print(f'Loading cached dialogue shares from {DIALOGUE_CSV_CACHE}', flush=True)
        with open(DIALOGUE_CSV_CACHE, 'rb') as f:
            return pickle.load(f)
    files = [fn for books in by_author.values() for _, fn in books]
    print(f'Measuring dialogue share in {len(files)} books (one-time; cached afterwards)...', flush=True)
    t0 = time.time()
    out = {}
    with ProcessPoolExecutor() as ex:
        for i, (fn, d) in enumerate(ex.map(_book, files), 1):
            out[fn] = d
            print(f'  [{i}/{len(files)}] {fn}: {d["style"]}, mean share {d["shares"].mean():.0%} '
                  f'({time.time()-t0:.0f}s)', flush=True)
    with open(DIALOGUE_CSV_CACHE, 'wb') as f:
        pickle.dump(out, f)
    return out


# ------------------------------------------------------------ analysis ----
def signflip_p(v, rng, n=20000):
    v = v[~np.isnan(v)]
    obs = abs(v.mean())
    signs = rng.choice([-1.0, 1.0], size=(n, len(v)))
    return float((1 + np.sum(np.abs((signs * v).mean(axis=1)) >= obs - 1e-12)) / (n + 1))


def books_table(by_author, dlg, feats):
    rows = []
    for a, books in by_author.items():
        for y, fn in books:
            d = dlg[fn]
            sh = d['shares']
            rows.append({'author': a, 'book': fn, 'year': y, 'style': d['style'],
                         'n_windows': len(sh), 'windows_match_features': len(sh) == len(feats.get(fn, [])),
                         'marks_double_per_10k': d['n_double'] / max(d['n_words'], 1) * 1e4,
                         'marks_single_open_per_10k': d['n_single_open'] / max(d['n_words'], 1) * 1e4,
                         'mean_share': sh.mean() if len(sh) else np.nan,
                         'opening_share_first5': sh[:5].mean() if len(sh) >= 5 else np.nan})
    return pd.DataFrame(rows)


def author_profiles(by_author, authors, dlg, L):
    """authors x L: each author's dialogue share by window (mean over their books with detectable dialogue)."""
    D = np.full((len(authors), L), np.nan)
    for i, a in enumerate(authors):
        rows = [pad_to(dlg[fn]['shares'], L) for _, fn in by_author[a] if dlg[fn]['style'] != 'none']
        if rows:
            D[i] = np.nanmean(rows, axis=0)
    return D


def dialogue_curve(a, n, by_author, dlg):
    """Dialogue share aligned with an author's curves: mean over their TEST books
    (same books, same truncation to n windows), books with detectable dialogue only."""
    shs = [dlg[fn]['shares'][:n] for _, fn in by_author[a][1:]
           if dlg[fn]['style'] != 'none' and len(dlg[fn]['shares']) >= n]
    return np.mean(shs, axis=0) if shs else None


def residualise(c, d):
    dc = d - d.mean()
    var = (dc ** 2).sum()
    if var < 1e-12:
        return np.asarray(c, float)
    c = np.asarray(c, float)
    b = (dc * (c - c.mean())).sum() / var
    return c - b * dc


def detrend(x):
    x = np.asarray(x, float)
    t = np.arange(len(x))
    return x - np.polyval(np.polyfit(t, x, 1), t)


def feature_dialogue_corr(by_author, dlg, feats):
    """Detrended within-book correlation of each feature with dialogue share."""
    per = {f: [] for f in FEATURE_NAMES}
    for books in by_author.values():
        for _, fn in books:
            d = dlg[fn]
            if d['style'] == 'none' or len(d['shares']) != len(feats.get(fn, [])) or len(d['shares']) < 30:
                continue
            ds = detrend(d['shares'])
            if ds.std() < 1e-9:
                continue
            for f in FEATURE_NAMES:
                x = np.array([w.get(f, np.nan) for w in feats[fn]], float)
                if np.isnan(x).any():
                    continue
                xs = detrend(x)
                if xs.std() < 1e-9:
                    continue
                per[f].append(np.corrcoef(xs, ds)[0, 1])
    rows = []
    for f in FEATURE_NAMES:
        r = np.array(per[f])
        if len(r) == 0:
            continue
        z = np.arctanh(np.clip(r, -0.999, 0.999))
        rows.append({'feature': f, 'mean_r': float(np.tanh(z.mean())), 'books': len(r),
                     'books_positive': int((r > 0).sum()), 'dialogue_sensitive_guess': f in DIALOGUE_FEATURES})
    return pd.DataFrame(rows).sort_values('mean_r', key=lambda s: -s.abs())


# --------------------------------------------------------------- plot ----
def smooth(y, k=5):
    out = np.full(len(y), np.nan)
    for i in range(len(y)):
        lo, hi = max(0, i - k // 2), min(len(y), i + k // 2 + 1)
        out[i] = np.nanmean(y[lo:hi])
    return out


def plot_profile(D, prof_lo, prof_hi, G_cor, T_d, T_v, E_reg, E_d, path):
    L = D.shape[1]
    x = (np.arange(L) + 0.5) * WINDOW / 1000
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.5, 6.2), sharex=True)
    m = np.nanmean(D, axis=0)
    a1.fill_between(x, prof_lo, prof_hi, color=C_COR, alpha=0.15, linewidth=0)
    a1.plot(x, m, color=C_RAW, lw=1)
    a1.plot(x, smooth(m), color=C_COR, lw=2)
    a1.set_ylabel('share of words inside quotes', fontsize=8, color=INK)
    a1.set_title('Dialogue share by position (authors averaged; band = 95% over authors)', loc='left',
                 fontsize=10, color=INK)
    yc = np.nanmean(G_cor, axis=0) if len(G_cor) else np.full(L, np.nan)
    a2.plot(x, smooth(yc), color=C_COR, lw=2)
    a2.axhline(0, color=MUTED, lw=0.8)
    a2.set_ylabel('pooled voice signal (SD units)', fontsize=8, color=INK)
    a2.set_title('Voice signal, opening register removed (step 3)', loc='left', fontsize=10, color=INK)
    a2.set_xlabel('thousand words into the narrative', fontsize=8, color=INK)
    for ax in (a1, a2):
        style_axes(ax)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        if np.isfinite(T_d):
            ax.axvline(T_d * WINDOW / 1000, color=C_BAND, lw=1.5, ls='--')
        if np.isfinite(T_v):
            ax.axvline(T_v * WINDOW / 1000, color=INK, lw=1.5, ls='--')
        if E_reg is not None:
            ax.axvline(E_reg / 1000, color=MUTED, lw=1, ls=':')
        if np.isfinite(E_d):
            ax.axvline(E_d * WINDOW / 1000, color=C_BAND, lw=1, ls=':')
    a1.text(0.99, 0.04, 'orange dashed = dialogue starts to rise; orange dotted = dialogue dip ends;\n'
            'black dashed = voice onset; grey dotted = register end (step 1)', transform=a1.transAxes,
            ha='right', va='bottom', fontsize=7, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('Run (from')[0], formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--rebuild', action='store_true', help='re-measure dialogue share even if cached')
    ap.add_argument('--n-perm', type=int, default=None)
    ap.add_argument('--n-boot', type=int, default=None)
    ap.add_argument('--seed', type=int, default=20261010)
    args = ap.parse_args()
    n_perm = args.n_perm or (2000 if args.quick else 20000)
    n_boot = args.n_boot or (500 if args.quick else 2000)

    t0 = time.time()
    os.makedirs(os.path.join(OUT, 'plots'), exist_ok=True)
    if not os.path.exists(COMPARISON_CSV):
        raise SystemExit(f'{COMPARISON_CSV} not found -- run accumulator_retest.py first.')
    rng = np.random.default_rng(args.seed)
    cfg = Cfg(300)
    by_author = discover_authors()
    feats = load_features(by_author, 300)
    dlg = load_dialogue(by_author, rebuild=args.rebuild)
    bt = books_table(by_author, dlg, feats)
    bt.to_csv(os.path.join(OUT, 'dialogue_books.csv'), index=False)
    n_none = int((bt['style'] == 'none').sum())
    print(f'\nQuote styles: {bt["style"].value_counts().to_dict()}; mean dialogue share '
          f'{bt.loc[bt["style"] != "none", "mean_share"].mean():.0%} (books with dialogue).', flush=True)
    if n_none:
        print('  no dialogue detectable in: ' + ', '.join(bt.loc[bt['style'] == 'none', 'book']), flush=True)
    bad = bt[~bt['windows_match_features']]
    if len(bad):
        print(f'  !! window counts differ from the cached features for {len(bad)} books '
              f'(front-matter or window settings changed?): ' + ', '.join(bad['book']), flush=True)

    curves, habit = build_curves(by_author, feats)
    comp = pd.read_csv(COMPARISON_CSV)
    acc_full = acc_from_comp(comp, comp['sig_trimmed_full'])
    resid_all = opening_residuals(by_author, feats, cfg, 'all')
    resid_ref = opening_residuals(by_author, feats, cfg, 'ref')
    base = estimate(curves, habit, acc_full, cfg, mode='corpus_loo', resid_all=resid_all, resid_ref=resid_ref,
                    rng=rng, n_boot=n_boot)
    check_baseline(base, cfg)
    T_v = base['cor']['T']

    # ---- 1/2. dialogue profile, dip end, rise onset, gap to voice onset ----
    authors = sorted(curves)
    L = cfg.grand_len
    D = author_profiles(by_author, authors, dlg, L)
    have = ~np.all(np.isnan(D), axis=1)
    print(f'\nDialogue profile: {int(have.sum())} of {len(authors)} authors have a book with detectable dialogue.',
          flush=True)
    n_open, ref_lo, ref_hi = 5, 20, 40
    dip = np.nanmean(D[:, :n_open], axis=1) - np.nanmean(D[:, ref_lo:ref_hi], axis=1)
    dip_p = signflip_p(dip, rng)
    n_neg = int(np.sum(dip[~np.isnan(dip)] < 0))
    e_grid = fade_grid(cfg.conv_pos)
    fdes = fade_design(cfg.conv_pos, e_grid)
    prof = -np.nanmean(D[have][:, :cfg.conv_pos], axis=0)
    E_d = fit_fade(prof[None, :], e_grid, fdes)[0]
    idx = rng.integers(0, int(have.sum()), (n_boot, int(have.sum())))
    Eb = fit_fade(-np.nanmean(D[have][idx][:, :, :cfg.conv_pos], axis=1), e_grid, fdes)
    E_lo, E_hi = np.nanpercentile(Eb, [2.5, 97.5])
    rise = fit_grand(D[have], cfg, 'hinge', rng, n_boot)
    T_d = rise['T']

    # paired bootstrap: dialogue onset vs voice onset, same authors in each draw
    V = np.full((len(authors), L), np.nan)
    for a, row in zip(base['authors'], base['G_cor']):
        V[authors.index(a)] = row
    t_grid = np.arange(0, cfg.grand_max_t + 1, dtype=float)
    hdes = shape_design(L, t_grid, 'hinge')
    DD = D.copy()
    idx = rng.integers(0, len(authors), (n_boot, len(authors)))
    Tv_b, _ = fit_shape(np.nanmean(V[idx], axis=1), t_grid, hdes)
    Td_b, _ = fit_shape(np.nanmean(DD[idx], axis=1), t_grid, hdes)
    gap = (Tv_b - Td_b) * cfg.window
    g_lo, g_hi = np.nanpercentile(gap, [2.5, 97.5]) if np.isfinite(gap).any() else (np.nan, np.nan)
    gap0 = (T_v - T_d) * cfg.window
    E_reg = None
    if os.path.exists(GAP_TEST_CSV):
        g = pd.read_csv(GAP_TEST_CSV)
        E_reg = float(g.loc[g['quantity'] == 'register end (combined)', 'estimate_words'].iloc[0])

    # profile table + band for the plot
    boots = np.array([np.nanmean(D[have][rng.integers(0, int(have.sum()), int(have.sum()))], axis=0)
                      for _ in range(1000)])
    plo, phi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    pd.DataFrame({'position': np.arange(L), 'start_word': np.arange(L) * cfg.window,
                  'mean_share': np.nanmean(D, axis=0), 'ci_lo': plo, 'ci_hi': phi}).to_csv(
        os.path.join(OUT, 'dialogue_profile.csv'), index=False)
    plot_profile(D, plo, phi, base['G_cor'], T_d, T_v, E_reg, E_d, os.path.join(OUT, 'plots', 'dialogue_profile.png'))

    # ---- 3. which features move with dialogue ----
    corr = feature_dialogue_corr(by_author, dlg, feats)
    corr.to_csv(os.path.join(OUT, 'dialogue_feature_corr.csv'), index=False)

    # ---- 4. take dialogue out of the curves ----
    print('\nTaking dialogue out of the curves (residualising each curve on the author\'s dialogue share)...',
          flush=True)
    inc, curves_adj, curves_sub = [], {}, {}
    for a in curves:
        n = len(curves[a][FEATURE_NAMES[0]])
        d = dialogue_curve(a, n, by_author, dlg)
        if d is None:
            continue
        inc.append(a)
        curves_sub[a] = curves[a]
        curves_adj[a] = {f: residualise(curves[a][f], d) for f in FEATURE_NAMES}
    excl = sorted(set(curves) - set(inc))
    print(f'  {len(inc)} authors included' + (f'; left out (no detectable dialogue in their test books): '
                                              f'{", ".join(excl)}' if excl else ''), flush=True)
    df_adj = test_curves(curves_adj, cfg, n_perm, label='dialogue_adjusted')
    acc_adj = acc_dict(df_adj)
    acc_base_sub = {a: fl for a, fl in acc_full.items() if a in set(inc)}
    both = {a: [f for f in fl if f in acc_adj.get(a, [])] for a, fl in acc_base_sub.items()}
    kw = dict(mode='corpus_loo', resid_all=resid_all, resid_ref=resid_ref, rng=rng, n_boot=n_boot)
    e1 = estimate(curves_sub, habit, acc_base_sub, cfg, **kw)
    e2 = estimate(curves_adj, habit, acc_base_sub, cfg, **kw)
    e3 = estimate(curves_adj, habit, acc_adj, cfg, **kw)
    est_rows = [('same authors, original curves, original accumulators', e1, n_pairs(acc_base_sub)),
                ('dialogue removed from the curves, original accumulators', e2, n_pairs(acc_base_sub)),
                ('dialogue removed, accumulators re-selected on the adjusted curves', e3, n_pairs(acc_adj))]
    feat_rows = []
    for f in FEATURE_NAMES:
        b = sum(f in fl for fl in acc_base_sub.values())
        ad = sum(f in fl for fl in acc_adj.values())
        bo = sum(f in fl for fl in both.values())
        feat_rows.append({'feature': f, 'original': b, 'after_dialogue_removed': ad, 'both': bo})
    feat_tab = pd.DataFrame(feat_rows)

    # ---- report ----
    def ci(o):
        return f'{o[1]:,.0f}-{o[2]:,.0f}' if np.isfinite(o[1]) else '-'
    w = cfg.window
    L_ = ['# Dialogue-share check (verification plan A4)', '',
          f'Run {time.strftime("%Y-%m-%d")}. Dialogue = words inside quotation marks, per 300-word window. '
          f'Quote styles: {bt["style"].value_counts().to_dict()}; mean share {bt.loc[bt["style"] != "none", "mean_share"].mean():.0%} '
          'of words in books with detectable dialogue. Books with no detectable dialogue are left out of every '
          'statistic below' + (': ' + ', '.join(bt.loc[bt['style'] == 'none', 'book']) if n_none else ' (none found)') + '.', '']
    if len(bad):
        L_ += [f'**Warning:** window counts differ from the cached features for {len(bad)} books: '
               + ', '.join(bad['book']) + '.', '']
    L_ += ['## 1. Does dialogue dip in the opening?', '',
           f'First 1,500 words vs. words 6,000-12,000, authors as units ({int(have.sum())} authors): mean difference '
           f'{np.nanmean(dip):+.3f} (a share of 0.10 = 10 percentage points), lower in the opening for {n_neg} of '
           f'{int(np.sum(~np.isnan(dip)))} authors, sign-flip p = {dip_p:.4f}.',
           f'The dip ends (fade-out fit) at **{E_d * w:,.0f} words** (95% {E_lo * w:,.0f}-{E_hi * w:,.0f})'
           + (f'; the combined opening register ended at {E_reg:,.0f} (step 3b).' if E_reg else '.'),
           'If the two end points agree, "descriptive opening with little dialogue" is a good description of '
           'the register; if dialogue recovers much earlier or later, it is not the whole story.', '',
           '## 2. Does dialogue start to rise where voice does?', '',
           f'Dialogue share starts to rise (flat-then-rising line, authors resampled) at **{T_d * w:,.0f} words**'
           + (f' (95% {rise["lo"] * w:,.0f}-{rise["hi"] * w:,.0f})' if np.isfinite(rise['lo']) else '')
           + f'. Voice onset (step 3): {T_v * w:,.0f} words.',
           f'Paired gap (voice onset minus dialogue onset, same authors in each draw): {gap0:+,.0f} words, '
           f'95% interval {g_lo:+,.0f} to {g_hi:+,.0f}.',
           ('The interval includes 0: voice and dialogue start rising at indistinguishable places, so dialogue '
            'cannot be ruled out as what the onset is tracking.' if g_lo <= 0 <= g_hi else
            ('Voice starts rising AFTER dialogue does.' if g_lo > 0 else 'Voice starts rising BEFORE dialogue does.')),
           '(A "rise" fit is a crude description of a profile that dips and recovers; use the dip end point in '
           'section 1 alongside it.)', '',
           '## 3. Which features move with dialogue?', '',
           'Detrended within-book correlation with dialogue share (positive = more of the feature when there is '
           'more dialogue), averaged over books.', '']
    L_ += md_table(corr.head(12), [('feature', 'feature'), ('mean_r', 'mean r'), ('books_positive', 'books r>0'),
                                   ('books', 'books')], {'mean_r': '+.2f'})
    L_ += ['', 'Full table: dialogue_feature_corr.csv. Features above about |r| = 0.3 are mostly scene structure '
           '(who is talking), not voice.', '',
           '## 4. Do the accumulators and the onset survive taking dialogue out?', '',
           f'{len(inc)} authors with detectable dialogue in their test books'
           + (f' (left out: {", ".join(excl)})' if excl else '') + '. Accumulators among them: '
           f'{n_pairs(acc_base_sub)} originally; {n_pairs(acc_adj)} after dialogue is removed from the curves; '
           f'{n_pairs(both)} in both.', '',
           '| analysis | onset (words) | 95% interval | authors | accumulator pairs |', '|---|---:|---:|---:|---:|']
    for name, e, npair in est_rows:
        o = onset_words(e['cor'], cfg)
        L_.append(f'| {name} | {o[0]:,.0f} | {ci(o)} | {e["cor"]["n_authors"]} | {npair} |')
    L_ += ['', 'By feature (the three flagged as scene structure first):', '']
    ft = pd.concat([feat_tab[feat_tab['feature'].isin(DIALOGUE_FEATURES)],
                    feat_tab[~feat_tab['feature'].isin(DIALOGUE_FEATURES)].sort_values('original', ascending=False)])
    L_ += md_table(ft, [('feature', 'feature'), ('original', 'accumulators (original)'),
                        ('after_dialogue_removed', 'after dialogue removed'), ('both', 'in both')],
                   {'original': '.0f', 'after_dialogue_removed': '.0f', 'both': '.0f'})
    L_ += ['', 'How to read it: if the onset and most accumulators survive, dialogue is not what the onset is '
           'tracking. If many accumulators vanish, check section 3: the vanishing ones should be the '
           'dialogue-correlated features. Vanishing is ambiguous (see the top of this script): dialogue may be a '
           'confound or part of how voice shows up.', '']
    with open(os.path.join(OUT, 'dialogue_report.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L_))
    print('\n'.join(L_[:14]))
    print(f'\nOutputs written to {OUT}/ (report: dialogue_report.md) -- total {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
