"""Deterministic latest-value resolver baseline.

The sharpest objection to validity-state consolidation is the simplest one: if
the current answer is always "the most recently stated value", why classify
contradictions and track validity at all? Just take the latest. This control
makes that objection a measured baseline instead of a rhetorical one.

How it works, and what is deliberately dumb about it:

1. Extract. On each user turn, an LLM pulls out value-setting statements as
   (attribute, value) pairs: "ship to Berlin" -> (shipping city, Berlin). This
   is the one LLM step, and it only reads the user turn, never the agent turn,
   matching the system's user-only salience boundary.
2. Store. Each pair is appended with its timestamp (the scenario date, set per
   day via set_clock) and turn index. Nothing is ever overwritten or superseded.
3. Resolve. At answer time, for each attribute the winner is chosen by Python
   `max(timestamp, turn)`. Pure code, no model deciding what is current. The
   resolved table is handed to the model only to phrase the answer.

What it cannot do, by construction, is the interesting part:

- It has no notion of a confirmation versus a change. "Yes, still Berlin"
  becomes another Berlin row, harmless, but "actually keep the old one" has no
  representation: there is no contradiction classifier to read intent.
- It answers past-tense questions only if the attribute is modelled as history.
  A plain latest-value table collapses Lagos and Berlin into one current row, so
  "what was the first city" has nowhere to read from. The resolver keeps the
  full history per attribute so it can answer first/previous, but it orders them
  by timestamp alone, never by a validity decision.

So the test it sets up for Chapter 3: on the relocation chain, does max-by-time
match Recall Lab? If yes, validity state is over-engineered for this scenario
and the honest move is to say so. If it breaks, where, an implicit correction,
a confirmation misread as a change, an attribute the extractor splits wrong,
that break is the argument for why a validity decision is not the same as a
timestamp sort.

The extractor is injectable so the resolve logic is tested offline with a
deterministic fake and no API key.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from recall_lab.config import (
    AGENT_MODEL,
    JUDGE_MODEL,
    MAX_OUTPUT_TOKENS,
    OPENROUTER_API_KEY,
)
from recall_lab.eval.metrics import estimate_tokens
from recall_lab.llm import chat_client, complete

EXTRACT_PROMPT = (
    "Extract durable value-setting facts the user states about themselves from "
    "this one user message. A value-setting fact assigns a value to an "
    "attribute: a shipping city, a favorite color, an allergy, a name. Ignore "
    "questions, chit-chat, and anything that sets no value.\n\n"
    "Return JSON only: a list of objects with keys \"attribute\" and \"value\". "
    "Use a short, stable attribute name (for example \"shipping city\", "
    '"favorite color", "daughter food restriction"). If the message sets no '
    "value, return [].\n\n"
    "User message: {message}"
)

ANSWER_PREAMBLE = (
    "You are a helpful assistant. Below is a resolved table of what the user has "
    "told you, with the current value of each attribute and its earlier values "
    "in time order. The current value is the most recently stated one. Use this "
    "table to answer. For a question about the past (the first or previous "
    "value), read the earlier values. If the table has no attribute matching the "
    "question, say you do not know rather than guessing."
)


@dataclass
class _Statement:
    """One extracted value-setting statement with its recency signal."""

    attribute: str
    value: str
    turn: int
    added_at: float | None = None


class DeterministicResolverAgent:
    """Latest-value-wins resolver: extract pairs, pick winner by max timestamp.

    Implements the `.respond(str) -> str` agent protocol used by the harness.
    The currency decision is `max(timestamp, turn)` in Python, never a model
    judgement, which is exactly the point: it isolates "take the latest" from
    validity reasoning.
    """

    def __init__(self, extractor: Callable[[str], list[dict]] | None = None) -> None:
        self.statements: list[_Statement] = []
        self.last_input_tokens = 0
        self._turn = 0
        self._clock: float | None = None
        self._injected_extractor = extractor

    def set_clock(self, timestamp: float | None) -> None:
        """Set the timestamp stamped onto statements extracted next, per day."""
        self._clock = timestamp

    def _extract(self, message: str) -> list[dict]:
        """Pull (attribute, value) pairs from one user message."""
        if self._injected_extractor is not None:
            return self._injected_extractor(message)
        client = chat_client()
        completion = complete(
            client,
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": EXTRACT_PROMPT.format(message=message)}],
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()
        return _parse_pairs(raw)

    def _resolve(self) -> dict[str, list[_Statement]]:
        """Group statements by attribute, each list ordered oldest-first.

        Ordering is by (added_at, turn) alone. The last item is the current
        value by pure recency; earlier items are history. No validity decision
        is made anywhere in here.
        """
        by_attr: dict[str, list[_Statement]] = {}
        for st in self.statements:
            by_attr.setdefault(st.attribute, []).append(st)
        for attr in by_attr:
            by_attr[attr].sort(key=lambda s: (s.added_at if s.added_at is not None else 0.0, s.turn))
        return by_attr

    def _render_table(self) -> str:
        """Render the resolved table: current value plus prior values per attribute."""
        resolved = self._resolve()
        if not resolved:
            return "(nothing recorded yet)"
        lines = []
        for attr, history in resolved.items():
            current = history[-1].value
            line = f"- {attr}: current = {current}"
            if len(history) > 1:
                priors = ", ".join(s.value for s in history[:-1])
                line += f"; earlier (oldest first): {priors}"
            lines.append(line)
        return "\n".join(lines)

    def respond(self, user_message: str) -> str:
        """Extract from the turn, store, resolve by max timestamp, answer."""
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is required to run DeterministicResolverAgent.")

        self._turn += 1
        for pair in self._extract(user_message):
            attribute = str(pair.get("attribute", "")).strip()
            value = str(pair.get("value", "")).strip()
            if attribute and value:
                self.statements.append(
                    _Statement(
                        attribute=attribute,
                        value=value,
                        turn=self._turn,
                        added_at=self._clock,
                    )
                )

        prompt = "\n".join(
            [
                ANSWER_PREAMBLE,
                "",
                "## Resolved memory",
                self._render_table(),
                "",
                "## Current user message",
                user_message,
            ]
        ).strip()
        self.last_input_tokens = estimate_tokens(prompt)

        client = chat_client()
        completion = complete(
            client,
            model=AGENT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        return completion.choices[0].message.content or ""


def _parse_pairs(raw: str) -> list[dict]:
    """Parse the extractor's JSON list, tolerant of a ```json fence.

    Returns [] on anything unparseable rather than raising, so a malformed
    extraction degrades to "nothing set this turn" instead of failing the run.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [p for p in parsed if isinstance(p, dict)]
