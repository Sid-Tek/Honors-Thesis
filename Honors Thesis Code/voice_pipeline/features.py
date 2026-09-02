"""
Stage A (part 2): the feature battery.

No nltk/spacy dependency (neither installs in this sandbox's offline pip
mirror) -- syllable counts, POS-ish categories (subordinators/coordinators),
and function-word rates are all done with small hand-curated word lists and
a vowel-group heuristic. This is a deliberate simplification: swap in a real
POS tagger / CMU pronouncing dictionary later for production-quality
syntactic and rhythmic features (see README "Known limitations").
"""
import re
from statistics import mean, pstdev
from typing import List, Dict
from windowing import Sentence

FUNCTION_WORDS = ['the', 'and', 'of', 'to', 'in', 'that', 'it', 'was', 'but', 'as']

SUBORDINATORS = {
    'because', 'although', 'since', 'while', 'if', 'when', 'though',
    'unless', 'whereas', 'whenever', 'after', 'before', 'until', 'once',
}
COORDINATORS = {'and', 'but', 'or', 'nor', 'for', 'yet', 'so'}

VOWEL_GROUP_RE = re.compile(r'[aeiouy]+')


def count_syllables(word: str) -> int:
    word = word.lower()
    groups = VOWEL_GROUP_RE.findall(word)
    n = len(groups)
    if word.endswith('e') and not word.endswith('le') and n > 1:
        n -= 1
    return max(n, 1)


def extract_features(window: List[Sentence]) -> Dict[str, float]:
    words = [w for s in window for w in s.words]
    n_words = len(words)
    if n_words == 0:
        return {}
    sentence_lengths = [s.n_words for s in window]
    n_sent = len(window)

    from collections import Counter
    counts = Counter(words)

    feats = {}
    feats['sent_len_mean'] = mean(sentence_lengths)
    feats['sent_len_std'] = pstdev(sentence_lengths) if len(sentence_lengths) > 1 else 0.0
    feats['ttr'] = len(counts) / n_words
    feats['hapax_rate'] = sum(1 for c in counts.values() if c == 1) / n_words
    feats['avg_word_len'] = mean(len(w) for w in words)

    total_commas = sum(s.commas for s in window)
    total_semis = sum(s.semicolons for s in window)
    total_dashes = sum(s.dashes for s in window)
    feats['comma_rate'] = total_commas / n_words * 100
    feats['semicolon_rate'] = total_semis / n_words * 100
    feats['dash_rate'] = total_dashes / n_words * 100

    feats['question_rate'] = sum(1 for s in window if s.terminal == '?') / n_sent
    feats['exclaim_rate'] = sum(1 for s in window if s.terminal == '!') / n_sent

    sub_count = sum(1 for w in words if w in SUBORDINATORS)
    coord_count = sum(1 for w in words if w in COORDINATORS)
    feats['subordination_rate'] = sub_count / n_words * 100
    feats['coordination_rate'] = coord_count / n_words * 100

    syll = [count_syllables(w) for w in words]
    feats['syll_per_word'] = mean(syll)
    feats['pct_monosyllabic'] = sum(1 for s_ in syll if s_ == 1) / n_words

    for fw in FUNCTION_WORDS:
        feats[f'fw_{fw}'] = counts.get(fw, 0) / n_words * 100

    return feats


FEATURE_NAMES = (
    ['sent_len_mean', 'sent_len_std', 'ttr', 'hapax_rate', 'avg_word_len',
     'comma_rate', 'semicolon_rate', 'dash_rate', 'question_rate', 'exclaim_rate',
     'subordination_rate', 'coordination_rate', 'syll_per_word', 'pct_monosyllabic']
    + [f'fw_{fw}' for fw in FUNCTION_WORDS]
)


class RunningAggregator:
    """Incremental feature aggregator for CUMULATIVE windows on long texts.

    Calling extract_features() fresh on "first N words", "first 2N words", ...
    is O(n^2) overall (each snapshot re-scans everything before it) -- fine
    for the ~6,000-word synthetic demo books, much too slow for a 200,000+
    word real novel with hundreds of cumulative windows. This keeps running
    sums/counters and produces each snapshot in O(1) amortized, so the whole
    cumulative sequence for one book is O(total words).
    """

    def __init__(self):
        from collections import Counter
        self.n_words = 0
        self.word_counts = Counter()
        self.singleton_count = 0
        self.sent_len_sum = 0.0
        self.sent_len_sumsq = 0.0
        self.n_sent = 0
        self.commas = 0
        self.semis = 0
        self.dashes = 0
        self.question_sent = 0
        self.exclaim_sent = 0
        self.sub_count = 0
        self.coord_count = 0
        self.syll_sum = 0
        self.mono_count = 0
        self.wordlen_sum = 0

    def add_sentence(self, s: Sentence):
        wc = self.word_counts
        for w in s.words:
            c = wc[w] + 1
            wc[w] = c
            if c == 1:
                self.singleton_count += 1
            elif c == 2:
                self.singleton_count -= 1
            self.wordlen_sum += len(w)
            syl = count_syllables(w)
            self.syll_sum += syl
            if syl == 1:
                self.mono_count += 1
            if w in SUBORDINATORS:
                self.sub_count += 1
            if w in COORDINATORS:
                self.coord_count += 1
        self.n_words += s.n_words
        self.sent_len_sum += s.n_words
        self.sent_len_sumsq += s.n_words ** 2
        self.n_sent += 1
        self.commas += s.commas
        self.semis += s.semicolons
        self.dashes += s.dashes
        if s.terminal == '?':
            self.question_sent += 1
        elif s.terminal == '!':
            self.exclaim_sent += 1

    def add_window(self, window: List[Sentence]):
        for s in window:
            self.add_sentence(s)

    def snapshot(self) -> Dict[str, float]:
        if self.n_words == 0 or self.n_sent == 0:
            return {}
        mean_len = self.sent_len_sum / self.n_sent
        var_len = max(self.sent_len_sumsq / self.n_sent - mean_len ** 2, 0.0)
        feats = {
            'sent_len_mean': mean_len,
            'sent_len_std': var_len ** 0.5,
            'ttr': len(self.word_counts) / self.n_words,
            'hapax_rate': self.singleton_count / self.n_words,
            'avg_word_len': self.wordlen_sum / self.n_words,
            'comma_rate': self.commas / self.n_words * 100,
            'semicolon_rate': self.semis / self.n_words * 100,
            'dash_rate': self.dashes / self.n_words * 100,
            'question_rate': self.question_sent / self.n_sent,
            'exclaim_rate': self.exclaim_sent / self.n_sent,
            'subordination_rate': self.sub_count / self.n_words * 100,
            'coordination_rate': self.coord_count / self.n_words * 100,
            'syll_per_word': self.syll_sum / self.n_words,
            'pct_monosyllabic': self.mono_count / self.n_words,
        }
        for fw in FUNCTION_WORDS:
            feats[f'fw_{fw}'] = self.word_counts.get(fw, 0) / self.n_words * 100
        return feats


def extract_features_cumulative(local_window_list: List[List[Sentence]]) -> List[Dict[str, float]]:
    """Snapshot features after each successive local window is folded in --
    equivalent to calling extract_features() on cumulative_windows() output,
    but O(total words) instead of O(n^2)."""
    agg = RunningAggregator()
    snapshots = []
    for window in local_window_list:
        agg.add_window(window)
        snapshots.append(agg.snapshot())
    return snapshots
