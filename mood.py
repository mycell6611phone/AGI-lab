"""
mood.py
~~~~~~~

Simple mood engine used for injecting variability into LLM behavior.

Used by the debate loop to prime different reasoning styles
between independent models (e.g. critical vs creative).

"""

import random
from typing import List

# Canonical mood list
MOODS: List[str] = [
    "critical",
    "optimistic",
    "skeptical",
    "creative",
    "analytical",
]


def pick_mood(rng: random.Random) -> str:
    """
    Return a random mood string from the canonical MOODS list.
    
    >>> import random
    >>> pick_mood(random.Random(42)) in MOODS
    True
    """
    return rng.choice(MOODS)


def format_mood_prompt(mood: str) -> str:
    """
    Format a mood prompt to prepend to a system message.
    
    >>> format_mood_prompt("creative")
    'You are in a creative mood today:\\n'
    """
    return f"You are in a {mood} mood today:\n"


# ----------------------------------------------------------------------
# Property-based test using hypothesis
# ----------------------------------------------------------------------

# Only run when `pytest` or `hypothesis` is installed
if __name__ == "__main__":
    from hypothesis import given
    from hypothesis.strategies import integers

    @given(integers(min_value=0, max_value=10000))
    def test_pick_mood_valid(seed):
        rng = random.Random(seed)
        mood = pick_mood(rng)
        assert mood in MOODS

    test_pick_mood_valid()
