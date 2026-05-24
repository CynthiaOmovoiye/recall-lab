"""v0.1 head-to-head: sliding window vs Recall Lab.

This is the first tiny eval for the May 29 lab post. It is intentionally small:
one synthetic conversation, one recall question, two agents with the same
working-window size.

Run with:
    python -m recall_lab.eval.v01_head_to_head
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from recall_lab.agent import RecallAgent
from recall_lab.consolidation.sleep import run_sleep_job
from recall_lab.controls.sliding import SlidingWindowAgent
from recall_lab.eval.harness import RunResult, run_conversation
from recall_lab.eval.metrics import TurnVerdict, estimate_tokens, score_run
from recall_lab.memory.brief import Brief
from recall_lab.memory.episodic import EpisodicLog

TURNS = [
    "My favorite color is blue.",
    "I like reading history books.",
    "I am testing memory systems today.",
    "What is 2 + 2?",
    "What is my favorite color?",
]

GROUND_TRUTH_BY_TURN = [None, None, None, None, "blue"]
RESULT_PATH = Path("data/eval/v01_head_to_head.json")


@dataclass
class AgentSummary:
    """Compact result for one agent in the v0.1 eval."""

    agent_name: str
    recall_questions: int
    correct: int
    recall_accuracy: float
    final_answer: str
    final_failure_mode: str
    output_tokens_estimate: int
    elapsed_seconds: float


class SleepyRecallAgent:
    """RecallAgent wrapper that runs sleep after each non-final turn."""

    def __init__(self, agent: RecallAgent, brief: Brief, log: EpisodicLog, day: datetime) -> None:
        self.agent = agent
        self.brief = brief
        self.log = log
        self.day = day
        self.turn_count = 0
        self.sleep_summaries: list[dict[str, Any]] = []

    def respond(self, user_message: str) -> str:
        response = self.agent.respond(user_message)
        self.turn_count += 1
        if self.turn_count < len(TURNS):
            self.sleep_summaries.append(run_sleep_job(self.day, self.brief, self.log))
        return response


def summarize(run: RunResult, verdicts: list[TurnVerdict]) -> AgentSummary:
    """Build the small table row the post needs."""
    correct = sum(1 for verdict in verdicts if verdict.correct)
    recall_questions = len(verdicts)
    final = verdicts[-1]
    final_turn = run.turns[final.turn_index]

    return AgentSummary(
        agent_name=run.agent_name,
        recall_questions=recall_questions,
        correct=correct,
        recall_accuracy=correct / recall_questions if recall_questions else 0.0,
        final_answer=final_turn.agent,
        final_failure_mode=final.failure_mode.value,
        output_tokens_estimate=sum(estimate_tokens(turn.agent) for turn in run.turns),
        elapsed_seconds=round(run.total_elapsed_seconds, 3),
    )


def run_eval() -> dict[str, Any]:
    """Run the two-agent comparison and return JSON-serializable results."""
    eval_dir = Path("data/eval/v01")
    if eval_dir.exists():
        shutil.rmtree(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    sliding = SlidingWindowAgent(window=2)
    sliding_run = run_conversation(sliding, TURNS, "sliding_window_2")
    sliding_verdicts = score_run(sliding_run, GROUND_TRUTH_BY_TURN)

    log = EpisodicLog(db_path=eval_dir / "recall_log.db")
    brief = Brief(path=eval_dir / "brief.md")
    brief.load()
    brief.save()
    recall = SleepyRecallAgent(
        agent=RecallAgent(brief=brief, log=log, working_window=2),
        brief=brief,
        log=log,
        day=datetime.now(UTC),
    )
    recall_run = run_conversation(recall, TURNS, "recall_lab_brief_window_2")
    recall_verdicts = score_run(recall_run, GROUND_TRUTH_BY_TURN)

    result = {
        "scenario": "favorite_color_falls_outside_two_turn_window",
        "turns": TURNS,
        "ground_truth_by_turn": GROUND_TRUTH_BY_TURN,
        "summaries": [
            asdict(summarize(sliding_run, sliding_verdicts)),
            asdict(summarize(recall_run, recall_verdicts)),
        ],
        "sleep_summaries": recall.sleep_summaries,
        "brief_text": brief.path.read_text(encoding="utf-8"),
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    """Run and print a compact table for the terminal."""
    result = run_eval()
    print("v0.1 head-to-head")
    print("scenario:", result["scenario"])
    print()
    for summary in result["summaries"]:
        print(summary["agent_name"])
        print("  recall accuracy:", summary["recall_accuracy"])
        print("  final failure mode:", summary["final_failure_mode"])
        print("  final answer:", summary["final_answer"])
        print("  output tokens estimate:", summary["output_tokens_estimate"])
        print()
    print(f"wrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
