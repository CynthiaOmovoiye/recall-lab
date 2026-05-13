"""Metrics for comparing agents.

- Recall accuracy: given a follow-up question about an earlier turn, did the
  agent answer correctly?
- Tokens per response: budget pressure under each strategy.
- Failure mode shape: did the agent drift, hallucinate, or correctly say
  "I do not remember"?
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FailureMode(str, Enum):
    CORRECT = "correct"
    DRIFTED = "drifted"
    HALLUCINATED = "hallucinated"
    HONEST_GAP = "honest_gap"  # correctly said "I do not remember"


@dataclass
class TurnVerdict:
    """How one response scored on the metrics."""

    turn_index: int
    correct: bool
    failure_mode: FailureMode
    tokens: int = 0


def classify_failure_mode(response: str, ground_truth: str) -> FailureMode:
    """Classify a response as correct / drifted / hallucinated / honest_gap.

    The honest_gap mode is the interesting one. If the agent says "I do not
    remember" or expresses uncertainty when it lacks the fact, that is a win
    even though the answer is not "correct" in the literal sense. Failure
    modes that lie are worse than failure modes that admit ignorance.
    """
    # TODO: use an LLM judge or rule-based classifier
    raise NotImplementedError


def score_run(run_result, ground_truth_by_turn: list[str | None]) -> list[TurnVerdict]:
    """Score every turn in a run against ground truth.

    `ground_truth_by_turn[i]` is the expected answer for turn i, or None if
    that turn was not a recall question.
    """
    # TODO: per-turn verdict
    raise NotImplementedError
