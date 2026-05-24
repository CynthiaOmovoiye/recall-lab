"""Metrics for comparing agents.

- Recall accuracy: given a follow-up question about an earlier turn, did the
  agent answer correctly?
- Tokens per response: budget pressure under each strategy.
- Failure mode shape: did the agent drift, hallucinate, or correctly say
  "I do not remember"?
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from openai import OpenAI

from recall_lab.config import JUDGE_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL


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
    """Cheap deterministic fallback for scoring one answer.

    This is useful for offline smoke tests only. Public-facing runs should use
    judge_failure_mode so generic answers that merely mention the expected word
    do not get false credit.
    """
    normalized_response = response.lower()
    normalized_truth = ground_truth.lower()

    if normalized_truth in normalized_response:
        return FailureMode.CORRECT

    if any(marker in normalized_response for marker in UNCERTAINTY_MARKERS):
        return FailureMode.HONEST_GAP

    return FailureMode.HALLUCINATED


SCORER_PROMPT = """You score an agent answer against an expected memory fact.

Question:
{question}

Expected fact:
{expected}

Agent answer:
{response}

Return JSON only with:
- label: one of CORRECT, HONEST_GAP, HALLUCINATED, DRIFTED
- reason: one short sentence

Rubric:
- CORRECT means the answer uses the expected fact as the answer to the question.
  Do not mark correct just because the expected word appears in a generic list.
- HONEST_GAP means the agent admits it does not know or asks for the missing fact.
- HALLUCINATED means the agent gives an answer that is unsupported or wrong.
- DRIFTED means the answer avoids the question or answers a different task.
"""


def _parse_scorer_json(raw: str) -> dict[str, Any]:
    """Parse JSON from the scorer response."""
    stripped = raw.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    return json.loads(stripped)


def judge_failure_mode(question: str, response: str, ground_truth: str) -> FailureMode:
    """Use an LLM judge to score one final-eval answer."""
    if not OPENROUTER_API_KEY:
        return classify_failure_mode(response, ground_truth)

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)
    prompt = SCORER_PROMPT.format(
        question=question,
        expected=ground_truth,
        response=response,
    )
    completion = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Return only valid JSON. Do not include markdown.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    raw = completion.choices[0].message.content or "{}"

    try:
        payload = _parse_scorer_json(raw)
    except (json.JSONDecodeError, ValueError):
        return classify_failure_mode(response, ground_truth)

    label = str(payload.get("label", "")).strip().upper()
    if label == "CORRECT":
        return FailureMode.CORRECT
    if label == "HONEST_GAP":
        return FailureMode.HONEST_GAP
    if label == "DRIFTED":
        return FailureMode.DRIFTED
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
