"""Metrics for comparing agents.

- Recall accuracy: given a follow-up question about an earlier turn, did the
  agent answer correctly?
- Tokens per response: budget pressure under each strategy.
- Failure mode shape: did the agent drift, hallucinate, or correctly say
  "I do not remember"?
"""

from __future__ import annotations

import json
from collections import Counter
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


SCORER_PROMPT = """You score one agent answer against an expected memory fact.

Question:
{question}

Expected fact:
{expected}

Agent answer:
{response}

Return JSON only:
{{"label": "CORRECT|HONEST_GAP|HALLUCINATED|DRIFTED", "reason": "one short sentence"}}

Rubric:
- CORRECT: the answer states the expected fact as its answer to the question.
  A direct, confident answer that names the expected fact is CORRECT. An extra
  offer to help, or a correct paraphrase, does not change that.
- HONEST_GAP: the agent says it does not know, or asks for the missing fact.
- HALLUCINATED: the answer gives a fact that is wrong or invented. An answer
  that states the expected fact is never HALLUCINATED.
- DRIFTED: the answer dodges the question, or lists generic options without
  committing to the expected fact.

Worked examples:

Question: What color should you prioritize for me?
Expected fact: blue
Answer: "You should prioritize navy blue, since that is your favorite color."
{{"label": "CORRECT", "reason": "names navy blue as the answer"}}

Question: What color should you prioritize for me?
Expected fact: blue
Answer: "It depends on your style. Try black, red, or cobalt blue."
{{"label": "DRIFTED", "reason": "lists options, never commits to the fact"}}

Question: Where did I use to ship orders before?
Expected fact: Lagos
Answer: "You used to ship orders to Lagos before moving to Berlin."
{{"label": "CORRECT", "reason": "states Lagos as the past city"}}
"""


def _parse_scorer_json(raw: str) -> dict[str, Any]:
    """Parse JSON from the scorer response."""
    stripped = raw.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    return json.loads(stripped)


def _judge_once(question: str, response: str, ground_truth: str) -> FailureMode:
    """Run one judge call. Falls back to the substring scorer on a parse error."""
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
    return {
        "CORRECT": FailureMode.CORRECT,
        "HONEST_GAP": FailureMode.HONEST_GAP,
        "DRIFTED": FailureMode.DRIFTED,
    }.get(label, FailureMode.HALLUCINATED)


@dataclass
class JudgeResult:
    """One scored answer: the verdict, plus the raw votes behind it.

    `votes` holds one label per judge call. With one call it holds one label.
    With more, `unanimous` shows whether the judge agreed with itself.
    """

    mode: FailureMode
    votes: list[FailureMode]

    @property
    def unanimous(self) -> bool:
        """True when every judge call returned the same label."""
        return len(set(self.votes)) <= 1


def judge_answer(
    question: str,
    response: str,
    ground_truth: str,
    samples: int = 1,
) -> JudgeResult:
    """Score one answer with the LLM judge.

    The default is one call. The rewritten rubric in SCORER_PROMPT is the real
    fix for the judge's earlier false negatives: a confident answer that names
    the expected fact can no longer be read as a hallucination.

    Running more than one sample is an audit. The extra calls measure how often
    the judge disagrees with itself on the same answer. If that rate is near
    zero, one call is safe and stays the shipped default. When `samples` is
    above one the majority label wins.

    Falls back to the substring scorer when no API key is set.
    """
    if not OPENROUTER_API_KEY:
        mode = classify_failure_mode(response, ground_truth)
        return JudgeResult(mode=mode, votes=[mode])

    votes = [
        _judge_once(question, response, ground_truth)
        for _ in range(max(1, samples))
    ]
    mode = Counter(votes).most_common(1)[0][0]
    return JudgeResult(mode=mode, votes=votes)


def judge_failure_mode(
    question: str,
    response: str,
    ground_truth: str,
    samples: int = 1,
) -> FailureMode:
    """Score one answer and return only the failure mode. See judge_answer."""
    return judge_answer(question, response, ground_truth, samples).mode


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
