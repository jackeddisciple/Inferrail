"""Synthetic text whose length in tokens is known in advance.

Every word is three or four lowercase letters, so each word, with the space
before it, is one token for the simulator's tokenizer and about one for BPE
tokenizers. Text drawn from different seeds shares no prefix beyond chance.
"""

import random
import string

_CONSONANTS = "bcdfghjklmnprstvwz"
_VOWELS = "aeiou"


def _vocabulary() -> list[str]:
    words = set()
    for a in _CONSONANTS:
        for b in _VOWELS:
            for c in _CONSONANTS:
                words.add(a + b + c)
                for d in _VOWELS:
                    words.add(a + b + c + d)
    return sorted(words)


VOCABULARY = _vocabulary()
assert all(set(w) <= set(string.ascii_lowercase) and 3 <= len(w) <= 4 for w in VOCABULARY)


def words(rng: random.Random, count: int) -> str:
    """`count` space separated words, about `count` tokens."""
    return " ".join(rng.choice(VOCABULARY) for _ in range(count))


def tokens(text: str) -> int:
    """The number of words, which is the token estimate the generators aim at."""
    return len(text.split())
