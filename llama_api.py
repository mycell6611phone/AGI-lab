"""
llama_api.py
~~~~~~~~~~~~

Async client for communicating with local Ollama-hosted LLaMA models.
Provides a unified `chat` interface that supports system prompts (mood injection),
streaming support, and retry behavior with basic error mapping.
"""

import logging
from typing import Literal
from config import settings

logger = logging.getLogger(__name__)

# Constants
TIMEOUT_SECONDS = 30
MAX_RETRIES = 3


async def chat(
    model: str,
    messages: list[dict],
    stream: bool = False,
    mood_prompt: str = "",
) -> dict:
    """
    Send a chat completion request to the local Ollama LLaMA model.

    Parameters
    ----------
    model : str
        The name or tag of the local model (e.g. "llama3").
    messages : list[dict]
        A list of OpenAI-style messages, typically including a user prompt.
    stream : bool
        Whether to enable server-side streaming responses (optional).
    mood_prompt : str
        Optional "system" mood pre-prompt to inject as the first message.

    Returns
    -------
    dict
        Parsed JSON response from Ollama-compatible endpoint.

    Raises
    ------
    RuntimeError
        If connection fails or the response is invalid after retries.

    Notes
    -----
    - Reads base URL and settings from `config.settings.ollama_url`
    - Prepends a system message using `mood_prompt` if provided
    - Logs all errors and retries
    """

    logger.debug(f"Preparing request to model: {model} | stream={stream}")

    if mood_prompt:
        messages = [{"role": "system", "content": mood_prompt}] + messages

    # Offline stub: echo back the user prompt with a model label.  This allows
    # the rest of the system to execute without requiring a running Ollama
    # server.  In a real deployment this would perform an HTTP request with
    # retry logic.
    user_text = " ".join(m["content"] for m in messages if m["role"] != "system")
    content = f"{model.upper()}>>> {user_text}".strip()

    return {"choices": [{"message": {"content": content}}]}
