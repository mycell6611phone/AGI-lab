"""
validator.py
~~~~~~~~~~~~

Anchors local debates using GPT-4o as a high-confidence external validator.
Used for final arbitration when local models disagree or hallucinate.

"""

import os
import time
import logging
from typing import Dict
try:
    import openai  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    openai = None
from config import settings

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------
def validate(payload: dict, api_key_env: str = "OPENAI_API_KEY") -> dict:
    """
    Send a summarised payload to GPT-4o for final verdict.

    Parameters
    ----------
    payload : dict
        Includes debate input, responses, and critique context.
    api_key_env : str
        Environment variable name holding the OpenAI API key.

    Returns
    -------
    dict
        {
            "verdict": "accept" | "reject" | "unsure",
            "rationale": "...",
            "confidence": float
        }
    """

    if openai is None:
        logger.warning("openai package not available; returning unsure verdict")
        return {"verdict": "unsure", "rationale": "openai unavailable", "confidence": 0.0}

    api_key = os.getenv(api_key_env)
    if not api_key:
        logger.warning("No API key found for validator; returning unsure verdict")
        return {"verdict": "unsure", "rationale": "missing API key", "confidence": 0.0}
    openai.api_key = api_key

    messages = _build_messages_from_payload(payload)

    retries = 3
    delay = 2

    for attempt in range(retries):
        try:
            logger.debug("Calling GPT-4o for verdict...")
            response = openai.ChatCompletion.create(
                model=settings.guardian_model,
                messages=messages,
                max_tokens=256,
                temperature=0,
            )

            content = response["choices"][0]["message"]["content"]
            verdict = "unsure"
            rationale = ""
            conf = 0.0
            for line in content.splitlines():
                if line.upper().startswith("VERDICT:"):
                    verdict = line.split(":", 1)[1].strip().lower()
                elif line.upper().startswith("RATIONALE:"):
                    rationale = line.split(":", 1)[1].strip()
                elif line.upper().startswith("CONFIDENCE:"):
                    try:
                        conf = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
            return {"verdict": verdict, "rationale": rationale, "confidence": conf}

        except Exception as e:
            logger.warning(f"Validator attempt {attempt+1} failed: {e}")
            time.sleep(delay)
            delay *= 2

    logger.error("All validator attempts failed. Returning fallback verdict.")
    return {
        "verdict": "unsure",
        "rationale": "Validator unreachable after retries.",
        "confidence": 0.0,
    }


# -------------------------------------------------------------------
# Helper: Payload Compressor (Stub)
# -------------------------------------------------------------------
def _build_messages_from_payload(payload: dict) -> list:
    """
    Convert debate context to OpenAI-style message list.
    Compress and truncate as needed.
    """
    text = str(payload)
    if len(text) > settings.max_tokens_guardian:
        text = text[: settings.max_tokens_guardian]
    return [
        {"role": "system", "content": "You are a neutral validator of AI responses."},
        {"role": "user", "content": f"Evaluate this:\n{text}"},
    ]


# -------------------------------------------------------------------
# Cost Estimator (TODO)
# -------------------------------------------------------------------
def estimate_cost(token_count: int) -> float:
    """
    Placeholder for cost tracking (OpenAI token usage).
    """
    # Simple linear model matching gpt-4o pricing (approximate).  Roughly
    # $0.01 per thousand tokens.
    return 0.01 * (token_count / 1000)
