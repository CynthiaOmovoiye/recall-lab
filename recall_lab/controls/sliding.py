"""Sliding-window baseline.

Keeps the last N turns in context. No brief, no judge, no consolidation.
This is the naive industry default and the bar Recall Lab has to beat.
"""

from __future__ import annotations

from openai import OpenAI

from recall_lab.config import AGENT_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL, SLIDING_WINDOW_TURNS


class SlidingWindowAgent:
    """The simplest possible long-conversation agent."""

    def __init__(self, window: int = SLIDING_WINDOW_TURNS) -> None:
        self.window = window
        self.history: list[dict] = []

    def respond(self, user_message: str) -> str:
        """Use the last `window` turns plus the new message as context."""
        context = self.history[-self.window :]

        messages = []
        for exchange in context:
            messages.append({"role": "user", "content": exchange["user"]})
            messages.append({"role": "assistant", "content": exchange["agent"]})

        messages.append({"role": "user", "content": user_message})

        client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )

        completion = client.chat.completions.create(
            model=AGENT_MODEL,
            messages=messages,
        )

        response = completion.choices[0].message.content or ""

        self.history.append({"user": user_message, "agent": response})
        return response
