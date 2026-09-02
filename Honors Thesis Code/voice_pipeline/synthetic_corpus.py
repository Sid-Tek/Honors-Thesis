"""
Synthetic two-author demo corpus with KNOWN GROUND TRUTH, used only to prove
the pipeline correctly recovers a burst feature and an accumulator feature
that were deliberately injected -- NOT a substitute for real novels.

Ground truth injected into Author A's TEST book only:
  - comma_rate:        BURST   -- high comma probability in the first ~15%
                        of the book, then drops to a lower constant baseline.
  - subordination_rate: ACCUMULATOR -- subordinator probability ramps up
                        linearly from low to high across the whole book.
  - every other feature is generated with a CONSTANT probability throughout
    (these are the "null" features -- the pipeline should mostly NOT flag
    them as burst or accumulator).

Author A's REFERENCE book and all of Author B's text are fully stationary
(no injected position effects) -- they exist only to give the scorer
something to compare against.
"""
import random

CONTENT_WORDS = (
    "house road river garden window story letter voice silence morning "
    "shadow city street corner table chair paper glass metal stone wood "
    "child woman man dog bird tree hill valley mountain sea ocean sky "
    "cloud rain snow wind fire light dark color sound smell taste touch "
    "thought memory dream fear hope anger joy grief love hate work rest "
    "journey home distance time moment hour day night year season change"
).split()

TERMINALS = ['.'] * 92 + ['?'] * 5 + ['!'] * 3


def _emit_word(rng, author_fw_bias):
    if rng.random() < author_fw_bias:
        from features import FUNCTION_WORDS
        return rng.choice(FUNCTION_WORDS)
    return rng.choice(CONTENT_WORDS)


def _make_sentence(rng, mean_len, comma_p, subordinator_p, semicolon_p, dash_p, author_fw_bias):
    length = max(3, round(rng.gauss(mean_len, mean_len * 0.35)))
    words = []
    if rng.random() < subordinator_p:
        words.append(rng.choice(
            ['because', 'although', 'since', 'while', 'when', 'though', 'unless']))
    for _ in range(length):
        words.append(_emit_word(rng, author_fw_bias))

    pieces = []
    for i, w in enumerate(words):
        pieces.append(w)
        if i < len(words) - 1:
            if rng.random() < comma_p:
                pieces[-1] += ','
            elif rng.random() < semicolon_p:
                pieces[-1] += ';'
            elif rng.random() < dash_p:
                pieces[-1] += ' --'
    terminal = rng.choice(TERMINALS)
    sentence = ' '.join(pieces) + terminal
    return sentence[0].upper() + sentence[1:]


def generate_book(seed, n_words_target, mean_sentence_len, comma_p_fn, subordinator_p_fn,
                   semicolon_p=0.01, dash_p=0.01, author_fw_bias=0.30):
    """comma_p_fn / subordinator_p_fn: callables taking frac-through-book
    (0..1) and returning a probability. Pass `lambda frac: constant` for a
    stationary (non-injected) feature."""
    rng = random.Random(seed)
    sentences = []
    total_words = 0
    while total_words < n_words_target:
        frac = total_words / n_words_target
        s = _make_sentence(
            rng, mean_sentence_len,
            comma_p=comma_p_fn(frac),
            subordinator_p=subordinator_p_fn(frac),
            semicolon_p=semicolon_p, dash_p=dash_p,
            author_fw_bias=author_fw_bias,
        )
        sentences.append(s)
        total_words += len(s.split())
    return ' '.join(sentences)


def build_demo_corpus(out_dir):
    import os
    os.makedirs(out_dir, exist_ok=True)

    # --- Author A: burst on comma_rate, accumulator on subordination_rate ---
    # Author A is the "heavy comma" author (0.30 baseline vs. B's 0.10) --
    # the burst pushes A's OWN habit even louder (0.55) for the opening
    # ~15%, then settles to A's normal 0.30. This is what makes it a
    # genuine self-signature burst rather than just "weird outlier text".
    def A_comma_burst(frac):
        return 0.75 if frac < 0.15 else 0.30

    def A_subordinator_accum(frac):
        return 0.05 + 0.30 * frac

    # Three independent replicate books for Author A, same injected pattern,
    # different random seeds -- this is what lets Stage C pool across books
    # (see run_pipeline.py) rather than relying on a single noisy realization.
    a_test_books = [
        generate_book(
            seed=s, n_words_target=6000, mean_sentence_len=9,
            comma_p_fn=A_comma_burst, subordinator_p_fn=A_subordinator_accum,
            author_fw_bias=0.32,
        )
        for s in (1, 11, 21)
    ]
    a_test = a_test_books[0]
    a_reference = generate_book(
        seed=2, n_words_target=6000, mean_sentence_len=9,
        comma_p_fn=lambda f: 0.30, subordinator_p_fn=lambda f: 0.20,
        author_fw_bias=0.32,
    )

    # --- Author B: fully stationary control, different baseline voice ---
    b_test = generate_book(
        seed=3, n_words_target=6000, mean_sentence_len=16,
        comma_p_fn=lambda f: 0.10, subordinator_p_fn=lambda f: 0.10,
        author_fw_bias=0.22,
    )
    b_reference = generate_book(
        seed=4, n_words_target=6000, mean_sentence_len=16,
        comma_p_fn=lambda f: 0.10, subordinator_p_fn=lambda f: 0.10,
        author_fw_bias=0.22,
    )

    paths = {
        'A_reference': os.path.join(out_dir, 'authorA_reference.txt'),
        'B_test': os.path.join(out_dir, 'authorB_test.txt'),
        'B_reference': os.path.join(out_dir, 'authorB_reference.txt'),
    }
    for key, text in zip(paths.keys(), [a_reference, b_test, b_reference]):
        with open(paths[key], 'w', encoding='utf-8') as f:
            f.write(text)

    paths['A_test_books'] = []
    for i, text in enumerate(a_test_books):
        p = os.path.join(out_dir, f'authorA_test_book{i+1}.txt')
        with open(p, 'w', encoding='utf-8') as f:
            f.write(text)
        paths['A_test_books'].append(p)
    paths['A_test'] = paths['A_test_books'][0]  # kept for backward compatibility
    return paths


if __name__ == '__main__':
    p = build_demo_corpus('demo_corpus')
    for k, v in p.items():
        print(k, '->', v)
