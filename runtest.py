import asyncio
import logging
import random

from config import settings
from llama_api import chat as llama_chat
from debate import debate_once
from mood import pick_mood
from memoryloop import filter_memory
from memory import VectorStore, MetaStore, store_memories, recall
from drift import DriftWindow, check_drift
from trainer import schedule_training, micro_finetune_step
from validator import validate as guardian_validate

logger = logging.getLogger("cycle_test")


async def run_cycle(idx: int, text: str, rng: random.Random, vector_store: VectorStore, meta_store: MetaStore, drift_window: DriftWindow) -> None:
    logger.info(f"Cycle {idx} input: {text}")
    result = await debate_once(
        user_input=text,
        rng=rng,
        llama_a=lambda **kw: llama_chat(model=settings.model_a_name, **kw),
        llama_b=lambda **kw: llama_chat(model=settings.model_b_name, **kw),
        pick_mood=pick_mood,
    )
    logger.info(f"Debate result: {result}")

    # run memory filtering in a background thread because it uses asyncio.run internally
    mem_out = await asyncio.to_thread(filter_memory, result, guardian_validate)
    logger.info(f"Memory filter: {mem_out}")
    store_memories(mem_out["validated"], vector_store, meta_store)
    micro_finetune_step(result, mem_out["validated"], settings.model_a_name, settings.model_b_name)

    metrics = {
        "agreement_rate": 1.0 if result.get("winner") in ("A", "B") else 0.0,
        "guardian_override_rate": 1.0 if result.get("guardian_verdict") == "override" else 0.0,
        "avg_response_length_delta": abs(len(result.get("resp_a", "")) - len(result.get("resp_b", ""))),
    }
    drift_window.update(metrics)
    summary = drift_window.compute_summary()
    logger.info(f"Drift summary: {summary}")
    if check_drift(summary):
        logger.info("Drift detected, scheduling training")
        schedule_training(mem_out["validated"], target_model=settings.model_a_name)

    recalls = recall(text, 3, vector_store, meta_store)
    logger.info(f"Recall results: {recalls}")


async def main() -> None:
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = settings.log_dir / "cycle_test.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),
        ],
    )
    rng = random.Random(0)
    vector_store = VectorStore(settings.faiss_path)
    meta_store = MetaStore(settings.sqlite_path)
    drift_window = DriftWindow(size=10)

    for i in range(10):
        await run_cycle(i, f"Test cycle {i}", rng, vector_store, meta_store, drift_window)

    vector_store.persist()
    meta_store.close()
    logger.info("Completed all cycles")


if __name__ == "__main__":
    asyncio.run(main())

