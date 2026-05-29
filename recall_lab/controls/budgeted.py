"""Budget-bounded sliding-window baseline.

The plain sliding window keeps a fixed number of recent turns. That makes the
mechanism visible but invites a fair objection: maybe Recall Lab only wins
because the baseline was starved of context, not because validity-aware memory
is better. This agent removes that objection.

Instead of "last N turns", it keeps as many of the most recent turns as fit
inside a target input-token budget. Point the budget at what Recall Lab's brief
plus working memory actually consumes, and the comparison becomes apples to
apples: both agents pay the same input-token price, and the only difference is
what each chooses to put in that budget. Raw recency versus consolidated,
validity-aware memory.

Giving recency *more* budget does not obviously help: more raw history means
more stale facts competing with current truth. That is exactly the pressure
this control applies.
"""

from __future__ import annotations

from recall_lab.config import (
    AGENT_MODEL,
    MAX_OUTPUT_TOKENS,
    OPENROUTER_API_KEY,
)
from recall_lab.eval.metrics import estimate_tokens
from recall_lab.llm import chat_client, complete


class BudgetedSlidingWindowAgent:
    """Sliding window bounded by an input-token budget instead of a turn count.

    Implements the `.respond(str) -> str` agent protocol used by the harness.
    """

    def __init__(self, input_token_budget: int) -> None:
        if input_token_budget <= 0:
            raise ValueError("input_token_budget must be positive.")
        self.input_token_budget = input_token_budget
        self.history: list[dict] = []
        self.last_input_tokens = 0
        self.last_included_turns = 0

    def _select_context(self, user_message: str) -> list[dict]:
        """Take the most recent turns that fit under the token budget.

        The new user message is always included and counts against the budget.
        Older turns are added newest-first until the next one would overflow.
        """
        budget_left = self.input_token_budget - estimate_tokens(user_message)
        selected: list[dict] = []
        for exchange in reversed(self.history):
            cost = estimate_tokens(exchange["user"]) + estimate_tokens(exchange["agent"])
            if cost > budget_left:
                break
            selected.append(exchange)
            budget_left -= cost
        selected.reverse()
        return selected

    def respond(self, user_message: str) -> str:
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is required to run BudgetedSlidingWindowAgent.")

        context = self._select_context(user_message)
        self.last_included_turns = len(context)

        messages = []
        for exchange in context:
            messages.append({"role": "user", "content": exchange["user"]})
            messages.append({"role": "assistant", "content": exchange["agent"]})
        messages.append({"role": "user", "content": user_message})

        self.last_input_tokens = sum(estimate_tokens(m["content"]) for m in messages)

        client = chat_client()
        completion = complete(
            client,
            model=AGENT_MODEL,
            messages=messages,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        response = completion.choices[0].message.content or ""

        self.history.append({"user": user_message, "agent": response})
        return response
