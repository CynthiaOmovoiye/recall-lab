"""LLM-based salience judge.

Given an exchange, returns a salience score in [0, 1] plus a one-line reason.
Inputs to the score: recency, goal-relevance, correction signal, user-stated
importance.

The judge scores the user turn only. The agent turn is never passed to it.
Earlier the judge saw both turns and was asked to ignore agent-stated facts. It
did not always comply, and a fact the agent merely recalled would get filed as
fresh user memory. Hiding the agent turn makes the rule hold every time.

The judge is intentionally simple. The goal of Recall Lab is not to build a
clever judge but to test whether selective forgetting at any reasonable
threshold beats storing everything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from recall_lab.config import JUDGE_MODEL, MAX_OUTPUT_TOKENS, OPENROUTER_API_KEY
from recall_lab.llm import chat_client, complete
from recall_lab.memory.episodic import Exchange


@dataclass
class SalienceVerdict:
    """The judge's output for one exchange."""

    score: float  # 0.0 to 1.0
    reason: str
    suggested_brief_section: str | None = None
    suggested_statement: str | None = None  # compressed semantic form


VALID_SECTIONS = {
    "stable_facts",
    "active_intents",
    "open_commitments",
    "corrections",
    "never_repeat",
}

JUDGE_PROMPT = """You score one user turn for salience.

You see only what the user said. Score how important it is to remember
long-term, and return JSON with these fields:
- score: float in [0,1]. Higher = more important to remember long-term.
- reason: one short sentence.
- suggested_brief_section: one of stable_facts | active_intents | open_commitments
  | corrections | never_repeat | null.
- suggested_statement: a single compressed semantic sentence to file under that
  section. Use null if score < 0.5.

The statement must come from what the user said in this turn. Do not file a fact
the user did not state.

Score high when the user turn:
- States a stable fact about the user
- Names an active intent or goal
- Records a commitment
- Corrects an earlier wrong statement
- Marks something the agent must never do or repeat

Score low for chit-chat, restatements, and ephemeral context.

User turn:
{user}
"""


def _extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from plain or fenced model output."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(stripped[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("Judge response must be a JSON object.")
    return parsed


def _normalise_verdict(payload: dict[str, Any]) -> SalienceVerdict:
    """Convert model JSON into a safe SalienceVerdict."""
    score = float(payload.get("score", 0.0))
    score = max(0.0, min(1.0, score))

    reason = str(payload.get("reason") or "No reason supplied.").strip()

    section = payload.get("suggested_brief_section")
    if section not in VALID_SECTIONS:
        section = None

    statement = payload.get("suggested_statement")
    if statement is not None:
        statement = str(statement).strip() or None

    if score < 0.5:
        section = None
        statement = None

    return SalienceVerdict(
        score=score,
        reason=reason,
        suggested_brief_section=section,
        suggested_statement=statement,
    )


def score_exchange(exchange: Exchange) -> SalienceVerdict:
    """Score one exchange for salience.

    Calls the judge model via OpenRouter. Returns structured output.
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is required to run the salience judge.")

    client = chat_client()

    prompt = JUDGE_PROMPT.format(user=exchange.user)
    completion = complete(
        client,
        model=JUDGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Return only valid JSON. Do not include markdown or commentary.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=MAX_OUTPUT_TOKENS,
    )

    content = completion.choices[0].message.content or "{}"
    return _normalise_verdict(_extract_json(content))
