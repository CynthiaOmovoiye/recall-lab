"""Sliding-window baseline.

Keeps the last N turns in context. No brief, no judge, no consolidation.
This is the naive industry default and the bar Recall Lab has to beat.
"""

from __future__ import annotations

from recall_lab.config import (
    AGENT_MODEL,
    MAX_OUTPUT_TOKENS,
    SLIDING_WINDOW_TURNS,
)
from recall_lab.eval.metrics import estimate_tokens
from recall_lab.llm import chat_client, complete


class SlidingWindowAgent:
    """The simplest possible long-conversation agent."""

    def __init__(self, window: int = SLIDING_WINDOW_TURNS) -> None:
        self.window = window
        self.history: list[dict] = []
        self.last_input_tokens = 0

    def respond(self, user_message: str) -> str:
        """Use the last `window` turns plus the new message as context."""
        context = self.history[-self.window :]

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
