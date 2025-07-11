"""
debate.py
~~~~~~~~~

Debate engine for dual-LLM comparison loop.

Takes user input, routes through two models with independent moods,
then uses cross-critique and (optionally) a GPT-4o guardian to resolve
disagreements or tie-break.

"""

import random
import asyncio
from typing import Callable, Optional

# Type alias for LLM call signature
LLMFunc = Callable[[str, list[dict], bool, str], dict]
MoodPicker = Callable[[random.Random], str]


async def debate_once(
    user_input: str,
    rng: random.Random,
    llama_a: LLMFunc,
    llama_b: LLMFunc,
    pick_mood: MoodPicker,
) -> dict:
    """
    Run a single debate cycle between two independent LLM models.

    Parameters
    ----------
    user_input : str
        The original user query or task input.
    rng : random.Random
        Random seed generator (for deterministic testing and mood selection).
    llama_a : Callable
        Async LLM wrapper for model A.
    llama_b : Callable
        Async LLM wrapper for model B.
    pick_mood : Callable
        Function to sample a mood string.

    Returns
    -------
    dict
        Structured debate outcome:

        {
            "input": user_input,
            "mood_a": ...,
            "resp_a": ...,
            "mood_b": ...,
            "resp_b": ...,
            "winner": ...,
            "reasoning": ...,
            "guardian_verdict": ... (optional)
        }

    Steps
    -----
    1. Sample a mood for each model.
    2. Build and inject mood prompt as system message.
    3. Call each model with user input + mood prompt.
    4. Build structured critique input: compare both answers.
    5. Send critique input to both models and analyze consistency.
    6. If models disagree beyond threshold, escalate to guardian_validator.
    7. Determine winning answer and return results.
    """

    # Experimental minimal debate loop.  Instead of sophisticated critique we
    # compare answer length as a cheap proxy for "better" reasoning.  This keeps
    # the function executable without real models while providing deterministic
    # behaviour for testing.

    mood_a = pick_mood(rng)
    mood_b = pick_mood(rng)

    prompt = {"role": "user", "content": user_input}
    resp_a, resp_b = await asyncio.gather(
        llama_a(messages=[prompt], mood_prompt=f"You feel {mood_a} today."),
        llama_b(messages=[prompt], mood_prompt=f"You feel {mood_b} today."),
    )

    text_a = resp_a.get("choices", [{"message": {"content": str(resp_a)}}])[0]["message"]["content"]
    text_b = resp_b.get("choices", [{"message": {"content": str(resp_b)}}])[0]["message"]["content"]

    # Cheap "critique": longer answer wins unless the lengths are almost equal.
    diff = abs(len(text_a) - len(text_b))
    if diff < 10:
        winner = rng.choice(["A", "B"])  # undecided -> random
        reasoning = "similar length; random tie break"
    else:
        winner = "A" if len(text_a) > len(text_b) else "B"
        reasoning = "longer answer preferred"

    return {
        "input": user_input,
        "mood_a": mood_a,
        "resp_a": text_a,
        "mood_b": mood_b,
        "resp_b": text_b,
        "winner": winner,
        "reasoning": reasoning,
    }

