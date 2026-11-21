"""LAB 1 - BPE from scratch."""

import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

WORDS = [
    "low", "low", "low", "low", "low",
    "lower", "lower",
    "newest", "newest", "newest",
    "newest", "newest", "newest",
    "widest", "widest", "widest"
]

NUM_MERGES = 8

# Convert words into characters.
vocab = Counter(
    tuple(w) + ("</w>",)
    for w in WORDS
)


def count_pairs(vocab):
    pairs = Counter()

    for symbols, freq in vocab.items():
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += freq

    return pairs


def merge_pair(vocab, pair):
    new_vocab = Counter()

    for symbols, freq in vocab.items():
        out = []
        i = 0

        while i < len(symbols):

            if (
                i < len(symbols) - 1
                and (symbols[i], symbols[i + 1]) == pair
            ):
                out.append(symbols[i] + symbols[i + 1])
                i += 2

            else:
                out.append(symbols[i])
                i += 1

        new_vocab[tuple(out)] += freq

    return new_vocab


def tokenize(word, merges):

    symbols = list(word) + ["</w>"]

    for pair in merges:
        i = 0

        while i < len(symbols) - 1:

            if (symbols[i], symbols[i + 1]) == pair:
                symbols[i:i + 2] = [
                    symbols[i] + symbols[i + 1]
                ]

            else:
                i += 1

    return symbols


print("Training words:", WORDS, "\n")
print("Start:", dict(vocab), "\n")

merges = []

for step in range(1, NUM_MERGES + 1):

    pairs = count_pairs(vocab)

    if not pairs:
        break

    best = max(
        pairs,
        key=lambda p: (pairs[p], p)
    )

    merges.append(best)

    vocab = merge_pair(vocab, best)

    print(
        f"Merge {step}: {best[0]!r} + {best[1]!r} "
        f"(seen {pairs[best]} times)"
    )

    print(
        "   ",
        [" ".join(s) for s in vocab]
    )


print("\nLearned merge rules:", merges)

print("\nTokenizing NEW words:")

for w in ["lowest", "newer", "wider"]:

    print(f"{w:8s} -> {tokenize(w, merges)}")