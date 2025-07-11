"""
trainer.py
~~~~~~~~~~

LoRA-based fine-tuning stub. When drift is detected and memory
confidence is high, this module schedules and prepares a tuning job
via `ollama create`.

Note: This is only a structural outline — actual fine-tuning logic,
tokenization, and model training are delegated to Ollama + external tools.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from metrics import log_micro_update

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------
def schedule_training(
    memories: List[Dict[str, str]],
    target_model: str,
) -> Optional[Path]:
    """
    If drift is confirmed and memory quality is sufficient, prepare
    a training dataset and schedule a LoRA fine-tune job.

    Parameters
    ----------
    memories : list of dict
        Each item: {"text": ..., "tag": ..., "confidence": ...}
    target_model : str
        The model alias (e.g., "llama3") to fine-tune.

    Returns
    -------
    Path | None
        Path to updated weights or None if training not triggered.
    """
    print(f"[Trainer] schedule_training called for model: {target_model} with {len(memories)} memories")
    logger.info(f"Evaluating {len(memories)} memory items for training schedule.")

    # Extremely naive criteria: require at least 5 NEW memories with confidence
    # above 0.8 before triggering a training run.  This keeps experimentation
    # cheap and easy to reason about.
    selected = [
        m for m in memories if m.get("tag") == "new" and m.get("confidence", 0) > 0.8
    ]
    if len(selected) < 5:
        print("[Trainer] Not enough high quality memories for training")
        logger.info("Not enough high quality memories for training")
        return None

    dataset_path = Path("data/lora_train.jsonl")
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    _save_jsonl(selected, dataset_path)

    modelfile = dataset_path.with_suffix(".Modelfile")
    _write_modelfile(target_model, dataset_path, modelfile)
    print(f"[Trainer] Training would be scheduled with dataset at {dataset_path}")
    logger.info("Training would be scheduled with dataset at %s", dataset_path)
    return modelfile


# -------------------------------------------------------------------
# Helper: Save JSONL (Placeholder)
# -------------------------------------------------------------------
def _save_jsonl(memories: List[Dict[str, str]], path: Path) -> None:
    """
    Serialize memories into Ollama fine-tune format.
    """
    print(f"[Trainer] Saving {len(memories)} memories to {path}")
    with open(path, "w", encoding="utf-8") as f:
        for m in memories:
            f.write(f"{{\"prompt\": {m['text']!r}, \"response\": {m['text']!r}}}\n")


# -------------------------------------------------------------------
# Helper: Build Modelfile (Placeholder)
# -------------------------------------------------------------------
def _write_modelfile(base_model: str, dataset_path: Path, output_path: Path) -> None:
    """
    Write Ollama-compatible Modelfile with LoRA config.
    """
    print(f"[Trainer] Writing Modelfile for base {base_model} at {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"FROM {base_model}\n")
        f.write(f"TRAIN {dataset_path}\n")


# -------------------------------------------------------------------
# MICRO-FINETUNE-ADD: Minimalistic micro-tuning step
# -------------------------------------------------------------------
_micro_state: Dict[str, float] = {}


def micro_finetune_step(
    debate_json: Dict[str, str],
    validated: List[Dict[str, str]],
    model_a: str,
    model_b: str,
) -> None:
    """Apply a tiny parameter-efficient nudge based on debate outcome."""  # MICRO-FINETUNE-ADD
    print(f"[Trainer] micro_finetune_step called for models {model_a} and {model_b}")
    diff = len(debate_json.get("resp_a", "")) - len(debate_json.get("resp_b", ""))
    if diff == 0:
        return
    loser = model_a if diff < 0 else model_b
    delta = abs(diff) / 1000.0
    _micro_state[loser] = _micro_state.get(loser, 0.0) + delta
    print(f"[Trainer] MICRO-FINETUNE-ADD nudge {loser} += {delta:.4f} based on debate")
    log_micro_update(loser, delta)
    logger.info("MICRO-FINETUNE-ADD nudge %s += %.4f based on debate", loser, delta)

    # Experimental trigger: once accumulated delta exceeds 1.0, schedule
    # a tiny LoRA training pass using the latest validated memories.
    if _micro_state[loser] > 1.0 and validated:
        print("[Trainer] MICRO-FINETUNE-ADD threshold reached; scheduling training")
        logger.info("MICRO-FINETUNE-ADD threshold reached; scheduling training")
        schedule_training(validated, target_model=loser)
        _micro_state[loser] = 0.0

