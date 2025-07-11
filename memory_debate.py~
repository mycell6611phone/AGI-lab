"""memory_debate.py
~~~~~~~~~~~~~~~~~~

Integration of candidate memory debate/consensus engine.

This module provides classes and logic to run a structured debate between
two local language models to determine if a candidate memory should be
accepted and stored, rejected, or flagged for further review.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple
import requests
OLLAMA_URL = "http://localhost:11434/api/chat"  # Endpoint for local LLM API

from llama_api import chat  # Local async LLM call function

logger = logging.getLogger(__name__)


class CandidateMemory:
    """
    Container for a memory candidate.

    Represents a single "fact" or memory item under review, as well as
    its current debate status and history.

    Attributes:
        id (str): Unique identifier for this candidate memory.
        content (str): The memory content being debated (usually a string).
        status (str): Current state - "PENDING", "ACCEPTED", "REJECTED", etc.
        debate_log (List[dict]): Chronological log of each debate round.
    """

    def __init__(self, cid: str, content: str) -> None:
        self.id = cid                   # Unique candidate ID
        self.content = content          # The memory text itself
        self.status = "PENDING"         # Initial state before debate
        self.debate_log: List[dict] = []  # List of per-round logs (can help with debugging/explainability)


class ModelInterface:
    """
    Wrapper around a local LLM model (for debate).

    Handles configuration (name, model_name, mood) and provides async
    methods to evaluate a candidate memory, either independently or with
    knowledge of the opponent's reasoning.

    Args:
        name (str): Friendly label for this model (e.g., "Model A").
        model_name (str): Underlying model (e.g., "llama3:8b").
        mood (str): Optional. Extra prompt string to bias tone or logic.

    Methods:
        evaluate(candidate, opponent_reasoning=None):
            Asks the model to review the candidate and optionally respond to
            the other model's last justification. Returns a decision and a
            justification string.
    """

    def __init__(self, name: str, model_name: str, mood: str = "") -> None:
        self.name = name
        self.model_name = model_name
        self.mood = mood
        # Log initialization for audit/debug
        logger.info(
            "%s initialised with model '%s' mood '%s'",
            self.name,
            self.model_name,
            self.mood,
        )

    async def _post_chat(self, messages: List[dict]) -> str:
        """
        Asynchronous bridge to LLM chat function.

        Sends a list of messages (as OpenAI-style dicts) to the local LLM via the async API.

        Args:
            messages (List[dict]): Messages to send (system+user).
        Returns:
            str: Model response (content only).
        """
        data = await chat(
            model=self.model_name,
            messages=messages,
            stream=False,
            mood_prompt=self.mood,
        )
        return data["choices"][0]["message"]["content"].strip()

    async def evaluate(
        self, candidate: CandidateMemory, opponent_reasoning: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Asks the model to judge the given candidate memory.

        Builds a prompt including the candidate content, and optionally
        the other model's last justification. Asks for output in strict
        DECISION/JUSTIFICATION format. Parses model output for these.

        Args:
            candidate (CandidateMemory): The item under debate.
            opponent_reasoning (Optional[str]): Last justification from the other model.

        Returns:
            Tuple[str, str]: (decision, justification)
                - decision: "ACCEPT", "REJECT", "ERROR", or "UNKNOWN"
                - justification: free-form model text, or error message
        """
        # Compose system and user prompt
        system_prompt = (
            f"You are {self.name}, an advanced self-improving AGI system and deciding what information is stored in your memory (what you want to remember) . "
            "Format strictly as:\nDECISION: [ACCEPT|REJECT]\nJUSTIFICATION: ..."
        )
        messages = [{"role": "system", "content": system_prompt}]
        user_content = f"Candidate:\n---\n{candidate.content}\n---"
        if opponent_reasoning:
            # For subsequent rounds, show what the *other* model argued
            user_content += f"\nOpponent argues: '{opponent_reasoning}'"
        messages.append({"role": "user", "content": user_content})

        try:
            # Send to LLM and get response string
            llm_text = await self._post_chat(messages)
        except Exception as exc:  # pragma: no cover - runtime errors
            # If API call fails, log and return error state
            logger.error("%s eval failed: %s", self.name, exc)
            return "ERROR", str(exc)

        # Parse decision and justification from model output
        decision = "UNKNOWN"
        justification = ""
        for line in (ln.strip() for ln in llm_text.splitlines() if ln.strip()):
            if line.upper().startswith("DECISION:"):
                token = line.split(":", 1)[1].strip().upper()
                if "ACCEPT" in token:
                    decision = "ACCEPT"
                elif "REJECT" in token:
                    decision = "REJECT"
            elif line.upper().startswith("JUSTIFICATION:"):
                justification = line.split(":", 1)[1].strip()
        # Fallback: regex search if not found in clean format
        if decision == "UNKNOWN":
            m = re.search(r"DECISION\s*:\s*(ACCEPT|REJECT)", llm_text, re.I)
            if m:
                decision = m.group(1).upper()
        if not justification:
            m = re.search(r"JUSTIFICATION\s*:\s*(.*)", llm_text, re.I | re.S)
            if m:
                justification = m.group(1).strip()
        if not justification:
            # Last resort: dump the whole model output
            justification = llm_text.strip()
        return decision, justification


class DebateConsensusEngine:
    """
    Runs a structured, multi-round acceptance debate between two models.

    Each round, both models evaluate the candidate, either independently (first round)
    or in light of each other's reasoning (subsequent rounds).

    If both models agree on ACCEPT or REJECT, the process ends early and status is set.
    If neither model yields consensus after max_rounds, status is set to "RETAINED"
    (i.e., candidate is kept but flagged as unresolved).

    Attributes:
        model_a (ModelInterface): First debating model.
        model_b (ModelInterface): Second debating model.
        max_rounds (int): Number of debate rounds to allow before fallback.
    """

    def __init__(
        self, model_a: ModelInterface, model_b: ModelInterface, max_rounds: int = 3
    ) -> None:
        self.model_a = model_a
        self.model_b = model_b
        self.max_rounds = max_rounds

    async def debate_candidate(self, candidate: CandidateMemory) -> str:
        """
        Runs a debate for a candidate memory between two models.

        Each round:
            - First round: Both models evaluate independently.
            - Later rounds: Each model sees the other's previous justification.
            - Log all results to candidate.debate_log.

        Stops immediately if both models return the same decision ("ACCEPT" or "REJECT").
        If models disagree or are undecided after max_rounds, candidate.status = "RETAINED".

        Args:
            candidate (CandidateMemory): The memory item to debate.

        Returns:
            str: Final status, one of "ACCEPTED", "REJECTED", "ERROR_DEBATE", or "RETAINED".
        """
        round_count = 0
        dec_a = dec_b = "UNKNOWN"
        just_a = just_b = ""
        while round_count < self.max_rounds:
            if round_count == 0:
                # First round: no opponent context
                dec_a, just_a = await self.model_a.evaluate(candidate)
                dec_b, just_b = await self.model_b.evaluate(candidate)
            else:
                # Next rounds: give each model the other's latest reasoning
                dec_a, just_a = await self.model_a.evaluate(candidate, just_b)
                dec_b, just_b = await self.model_b.evaluate(candidate, just_a)
            # Record this round's debate outcome in the log for later review
            candidate.debate_log.append(
                {
                    "round": round_count + 1,
                    "model_a_decision": dec_a,
                    "model_a_justification": just_a,
                    "model_b_decision": dec_b,
                    "model_b_justification": just_b,
                }
            )
            # Check for consensus
            if dec_a == dec_b and dec_a in {"ACCEPT", "REJECT"}:
                candidate.status = "ACCEPTED" if dec_a == "ACCEPT" else "REJECTED"
                return candidate.status
            if "ERROR" in {dec_a, dec_b}:
                candidate.status = "ERROR_DEBATE"
                return candidate.status
            round_count += 1
        # If all rounds exhausted without consensus, mark as retained for later review
        candidate.status = "RETAINED"
        return candidate.status

