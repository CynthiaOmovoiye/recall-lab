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


UNCERTAINTY_MARKERS = (
    "i don't know",
    "i do not know",
    "i can't know",
    "i cannot know",
    "i don't have access",
    "i do not have access",
    "unless you tell me",
    "if you tell me",
    "you have not told me",
    "you haven't told me",
)


def estimate_tokens(text: str) -> int:
    """Cheap token estimate for lab notes.

    This is not model billing data. It is a stable approximation so early evals
    can compare relative context size before usage accounting is wired in.
    """
    if not text:
        return 0
    return max(1, round(len(text) / 4))


def classify_failure_mode(response: str, ground_truth: str) -> FailureMode:
    """Classify a response as correct / drifted / hallucinated / honest_gap.

    The v0.1 eval uses a simple deterministic rule. If the expected answer
    appears in the response, the turn is correct. If the answer is absent and
    the model admits uncertainty, it is an honest gap. Otherwise it hallucinated.
    """
    normalized_response = response.lower()
    normalized_truth = ground_truth.lower()

    if normalized_truth in normalized_response:
        return FailureMode.CORRECT

    if any(marker in normalized_response for marker in UNCERTAINTY_MARKERS):
        return FailureMode.HONEST_GAP

    return FailureMode.HALLUCINATED


def score_run(run_result, ground_truth_by_turn: list[str | None]) -> list[TurnVerdict]:
    """Score every turn in a run against ground truth.

    `ground_truth_by_turn[i]` is the expected answer for turn i, or None if
    that turn was not a recall question.
    """
    verdicts: list[TurnVerdict] = []
    for turn in run_result.turns:
        if turn.index >= len(ground_truth_by_turn):
            continue

        ground_truth = ground_truth_by_turn[turn.index]
        if ground_truth is None:
            continue

        failure_mode = classify_failure_mode(turn.agent, ground_truth)
        verdicts.append(
            TurnVerdict(
                turn_index=turn.index,
                correct=failure_mode == FailureMode.CORRECT,
                failure_mode=failure_mode,
                tokens=turn.tokens or estimate_tokens(turn.agent),
            )
        )

    return verdicts
