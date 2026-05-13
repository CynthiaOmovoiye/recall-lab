"""LLM-based salience judge.

Given an exchange, returns a salience score in [0, 1] plus a one-line reason.
Inputs to the score: recency, goal-relevance, correction signal, user-stated
importance.

The judge is intentionally simple. The goal of Recall Lab is not to build a
clever judge but to test whether selective forgetting at any reasonable
threshold beats storing everything.
"""

from __future__ import annotations

from dataclasses import dataclass

from recall_lab.memory.episodic import Exchange


@dataclass
class SalienceVerdict:
    """The judge's output for one exchange."""

    score: float  # 0.0 to 1.0
    reason: str
    suggested_brief_section: str | None = None
    suggested_statement: str | None = None  # compressed semantic form


JUDGE_PROMPT = """You score conversation exchanges for salience.

For the exchange below, return JSON with these fields:
- score: float in [0,1]. Higher = more important to remember long-term.
- reason: one short sentence.
- suggested_brief_section: one of stable_facts | active_intents | open_commitments | corrections | never_repeat | null.
- suggested_statement: a single compressed semantic sentence to file under that section. null if score < 0.5.

Score high when the exchange:
- States a stable fact about the user
- Names an active intent or goal
- Records a commitment
- Corrects an earlier wrong statement
- Marks something the agent must never do or repeat

Score low for chit-chat, restatements, and ephemeral context.

Exchange:
USER: {user}
AGENT: {agent}
"""


def score_exchange(exchange: Exchange) -> SalienceVerdict:
    """Score one exchange for salience.

    Calls the judge model via OpenRouter. Returns structured output.
    """
    # TODO: build prompt from JUDGE_PROMPT, call client, parse JSON into SalienceVerdict
    raise NotImplementedError
