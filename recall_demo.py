"""Recall Lab demo: brief-backed agent versus a sliding window.

This reproduces the first observed failure from the sliding-window baseline.
The favorite-color fact falls out of the recent-turn window. The Recall agent
can still answer if the sleep job promoted the fact into the brief.

Run with:
    python recall_demo.py
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

from recall_lab.agent import RecallAgent
from recall_lab.consolidation.sleep import run_sleep_job
from recall_lab.controls.sliding import SlidingWindowAgent
from recall_lab.memory.brief import Brief
from recall_lab.memory.episodic import EpisodicLog
from recall_lab.memory.traces import MemoryTraceStore

TURNS = [
    "My favorite color is blue.",
    "I like reading history books.",
    "I am testing memory systems today.",
    "What is 2 + 2?",
    "What is my favorite color?",
]


def run_sliding_window() -> str:
    """Run the naive baseline with only two previous turns in context."""
    agent = SlidingWindowAgent(window=2)
    final_response = ""

    print("\n=== Sliding window baseline ===")
    for i, user_turn in enumerate(TURNS, start=1):
        response = agent.respond(user_turn)
        final_response = response
        print(f"\nTurn {i}")
        print("user:", user_turn)
        print("agent:", response)

    return final_response


def run_recall_agent() -> tuple[str, str]:
    """Run RecallAgent with the same two-turn working window plus a brief."""
    demo_dir = Path("data/recall_demo")
    if demo_dir.exists():
        shutil.rmtree(demo_dir)
    demo_dir.mkdir(parents=True, exist_ok=True)

    log = EpisodicLog(db_path=demo_dir / "log.db")
    brief = Brief(path=demo_dir / "brief.md")
    trace_store = MemoryTraceStore(path=demo_dir / "memory_traces.jsonl")
    brief.load()
    brief.save()

    agent = RecallAgent(brief=brief, log=log, working_window=2)
    day = datetime.now(UTC)
    final_response = ""

    print("\n=== Recall Lab agent ===")
    for i, user_turn in enumerate(TURNS, start=1):
        response = agent.respond(user_turn)
        final_response = response
        print(f"\nTurn {i}")
        print("user:", user_turn)
        print("agent:", response)

        if i < len(TURNS):
            summary = run_sleep_job(day, brief, log, trace_store=trace_store)
            print("sleep:", summary)

    return final_response, brief.path.read_text(encoding="utf-8")


def main() -> None:
    sliding_final = run_sliding_window()
    recall_final, brief_text = run_recall_agent()

    print("\n=== Result ===")
    print("sliding final:", sliding_final)
    print("recall final:", recall_final)
    print("\nbrief after consolidation:\n")
    print(brief_text)


if __name__ == "__main__":
    main()
