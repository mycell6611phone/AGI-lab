"""
memoryloop.py
~~~~~~~~~~~~~

Filters debate output into structured memory chunks with tags.
Uses a local LLM for tagging and a guardian validator (e.g., GPT-4o)
to approve or reject each memory candidate.

"""

import enum
import re
import logging
import asyncio
from difflib import SequenceMatcher
from typing import Callable, List, Dict

from config import settings
from memory_debate import CandidateMemory, ModelInterface, DebateConsensusEngine  # MEMORY-DEBATE-IMPORT

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Memory Tag Enum
# -------------------------------------------------------------------
class Tag(enum.Enum):
    KNOWN = "known"
    NEW = "new"
    DISCARD = "discard"
    FALSE = "false"


# -------------------------------------------------------------------
# Memory Filtering Entry Point
# -------------------------------------------------------------------
def filter_memory(
    debate_json: dict,
    guardian_fn: Callable[[str, List[dict]], dict],
) -> dict:
    """
    Parse debate output, extract memory candidates, assign tags,
    and validate with a guardian model.

    Parameters
    ----------
    debate_json : dict
        Output from `debate_once`, including model responses and reasoning.
    guardian_fn : Callable
        Function to call GPT-4o or similar validator with prompt/messages.

    Returns
    -------
    dict
        {
            "validated": [ {"text": ..., "tag": Tag.X} ],
            "rejected": [ {"text": ..., "reason": ...} ],
            "confidence": float  # overall agreement score
        }
    """

    logger.debug("Starting memory filter on debate output")

    text = debate_json.get("reasoning") or ""
    candidates = [c for c in split_into_chunks(text) if c]
    tagged = tag_with_local_model(candidates)

    # MEMORY-DEBATE-IMPORT: run a consensus debate on each candidate
    model_a = ModelInterface("MemoryA", settings.model_a_name)
    model_b = ModelInterface("MemoryB", settings.model_b_name)
    engine = DebateConsensusEngine(model_a, model_b, max_rounds=2)

    validated = []
    rejected = []
    confidences = []

    for idx, item in enumerate(tagged):
        cand = CandidateMemory(str(idx), item["text"])
        status = asyncio.run(engine.debate_candidate(cand))
        item["debate_log"] = cand.debate_log
        if status != "ACCEPTED":
            rejected.append({"text": item["text"], "reason": status})
            continue
        verdict = guardian_fn(item["text"], cand.debate_log)
        if verdict.get("verdict") == "reject":
            rejected.append({"text": item["text"], "reason": verdict.get("rationale", "")})
        else:
            validated.append({"text": item["text"], "tag": item["tag"], "confidence": verdict.get("confidence", 0.0)})
            confidences.append(verdict.get("confidence", 0.0))

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        "validated": validated,
        "rejected": rejected,
        "confidence": avg_conf,
    }


# -------------------------------------------------------------------
# Helper: Placeholder Chunk Splitter
# -------------------------------------------------------------------
def split_into_chunks(text: str) -> List[str]:
    """
    Naive sentence splitter for extracting memory candidates.
    """
    return re.split(r"[.?!]\s+", text.strip())


# -------------------------------------------------------------------
# Helper: Local Model Tagging Stub
# -------------------------------------------------------------------
def tag_with_local_model(chunks: List[str]) -> List[Dict[str, str]]:
    """
    Experimental offline tagger using string similarity.
    Previous chunks marked as known if closely matching history.
    """
    results = []
    for c in chunks:
        sim = max((SequenceMatcher(None, c, s).ratio() for s in _SEEN), default=0)
        tag = Tag.KNOWN.value if sim > 0.8 else Tag.NEW.value
        results.append({"text": c, "tag": tag})
        _SEEN.add(c)
    return results

_SEEN: set[str] = set()
