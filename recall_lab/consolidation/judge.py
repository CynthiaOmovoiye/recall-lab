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
    value_setting: bool = True  # did the turn set a current value, or just talk about one?


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
- value_setting: true or false. See the rule below. Only a value-setting turn
  may be filed as memory.
- suggested_brief_section: one of stable_facts | active_intents | open_commitments
  | corrections | never_repeat | null.
- suggested_statement: a single compressed semantic sentence to file under that
  section. Use null if score < 0.5 or value_setting is false.

The statement must come from what the user said in this turn. Do not file a fact
the user did not state.

value_setting rule. A turn is value_setting only if the user is setting or
changing the current value of an attribute about themselves, now. A turn is NOT
value_setting if the user is merely talking about, reminiscing over, or
expressing a feeling toward a value without making it their current choice.
- value_setting true: "My favorite color is blue." "Change my color to blue."
  "Ship to Berlin." "I'm allergic to shellfish."
- value_setting false: "Green is still such a beautiful color, I always come
  back to it in my head." "I used to love green." "Blue reminds me of the sea."
  These express sentiment about a value; they do not set the current value.
When value_setting is false, set suggested_brief_section and suggested_statement
to null even if the sentiment is strongly felt. A fond mention of an old value
must never be filed as the current value.

Score high when the user turn:
- States or changes a stable fact about the user
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


def _coerce_bool(value: Any, default: bool) -> bool:
    """Read a bool from model JSON, tolerating a string like "false" or "no".

    The field is missing on older payloads and may arrive as a string. Default
    is applied only when the field is absent, so a present "false" wins.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "no", "0", ""}
    return bool(value)


def _normalise_verdict(payload: dict[str, Any]) -> SalienceVerdict:
    """Convert model JSON into a safe SalienceVerdict.

    Two gates block promotion (section and statement become null):
    - score below 0.5, the original salience threshold.
    - value_setting false, a turn that talks about a value without setting it.
      Absent value_setting defaults to true so older payloads are unaffected.
    """
    score = float(payload.get("score", 0.0))
    score = max(0.0, min(1.0, score))

    reason = str(payload.get("reason") or "No reason supplied.").strip()
    value_setting = _coerce_bool(payload.get("value_setting"), default=True)

    section = payload.get("suggested_brief_section")
    if section not in VALID_SECTIONS:
        section = None

    statement = payload.get("suggested_statement")
    if statement is not None:
        statement = str(statement).strip() or None

    if score < 0.5 or not value_setting:
        section = None
        statement = None

    return SalienceVerdict(
        score=score,
        reason=reason,
        suggested_brief_section=section,
        suggested_statement=statement,
        value_setting=value_setting,
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
