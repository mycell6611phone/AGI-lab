"""
config.py
~~~~~~~~~

Centralised runtime configuration loader.

Usage
=====

Simply import **`settings`** anywhere in your codebase:

    >>> from config import settings
    >>> settings.model_a_name  # doctest: +ELLIPSIS
    'llama3'

Environment–precedence order
----------------------------
1. **Hard-coded defaults** (sane fall-backs)
2. Keys inside a local ``.env`` file (loaded via *python-dotenv*)
3. Real OS environment variables (highest priority)

Fields
------

* ``ollama_url``            – REST endpoint for local Ollama server  
* ``model_a_name`` / ``model_b_name`` – LLaMA-3 model aliases  
* ``guardian_model``        – OpenAI validator model (default *gpt-4o*)  
* ``faiss_path``            – Disk location of FAISS index file  
* ``sqlite_path``           – Path to SQLite metadata DB  
* ``log_dir``               – Directory for all log files  
* ``max_tokens_guardian``   – Token cap sent to GPT-4o  
* ``drift_window_size``     – Sliding window size for drift metrics  

Doctest examples
----------------

Ensure the loader respects precedence:

    >>> import os, tempfile
    >>> os.environ['OLLAMA_URL'] = 'http://override:1234'
    >>> from config import Settings
    >>> Settings.load().ollama_url
    'http://override:1234'

Default fall-back when key is absent:

    >>> del os.environ['OLLAMA_URL']
    >>> Settings.load().ollama_url
    'http://localhost:11434/api/chat'
"""

from __future__ import annotations

import os
from pathlib import Path
try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    def load_dotenv(*args, **kwargs):
        return False
try:
    from pydantic.dataclasses import dataclass
except Exception:  # pragma: no cover - optional dependency
    from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    ollama_url: str = "http://localhost:11434/api/chat"
    model_a_name: str = "llama3"
    model_b_name: str = "llama3"
    guardian_model: str = "gpt-4o"
    faiss_path: Path = Path("./data/faiss.index")
    sqlite_path: Path = Path("./data/memory.db")
    log_dir: Path = Path("./logs")
    max_tokens_guardian: int = 2048
    drift_window_size: int = 100
    
    # ---- NEW: Web search API keys ----
    google_api_key: str = ""
    google_cse_id: str = ""

    # ------------------------------------------------------------------ #
    # Factory loader – merge defaults → .env → OS env                     #
    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls) -> "Settings":
        """Return a populated, immutable `Settings` instance."""
        # Load variables from .env (only fills *missing* keys unless override=True)
        load_dotenv(override=False)

        def _get(key: str, default: str) -> str:
            return os.getenv(key, default)

        return cls(
            ollama_url=_get("OLLAMA_URL", cls.ollama_url),
            model_a_name=_get("MODEL_A_NAME", cls.model_a_name),
            model_b_name=_get("MODEL_B_NAME", cls.model_b_name),
            guardian_model=_get("GUARDIAN_MODEL", cls.guardian_model),
            faiss_path=Path(_get("FAISS_PATH", str(cls.faiss_path))).expanduser().resolve(),
            sqlite_path=Path(_get("SQLITE_PATH", str(cls.sqlite_path))).expanduser().resolve(),
            log_dir=Path(_get("LOG_DIR", str(cls.log_dir))).expanduser().resolve(),
            max_tokens_guardian=int(_get("MAX_TOKENS_GUARDIAN", str(cls.max_tokens_guardian))),
            drift_window_size=int(_get("DRIFT_WINDOW_SIZE", str(cls.drift_window_size))),
            google_api_key=_get("GOOGLE_API_KEY", ""),
            google_cse_id=_get("GOOGLE_CSE_ID", ""),
        )
       


# Singleton-style convenience handle ------------------------------------------------
settings: Settings = Settings.load()
