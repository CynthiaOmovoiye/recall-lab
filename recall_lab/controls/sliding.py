"""Sliding-window baseline.

Keeps the last N turns in context. No brief, no judge, no consolidation.
This is the naive industry default and the bar Recall Lab has to beat.
"""

from __future__ import annotations

from recall_lab.config import SLIDING_WINDOW_TURNS


class SlidingWindowAgent:
    """The simplest possible long-conversation agent."""

    def __init__(self, window: int = SLIDING_WINDOW_TURNS) -> None:
        self.window = window
        self.history: list[dict] = []

    def respond(self, user_message: str) -> str:
        """Use the last `window` turns plus the new message as context."""
        context = self.history[-self.window :]
        # TODO: call the model with `context` plus `user_message`
        response: str = ""
        raise NotImplementedError(
            "Wire up the OpenRouter client and return its response."
        )
        self.history.append({"user": user_message, "agent": response})
        return response
