"""The experimental Recall Lab agent.

Uses three memory layers: working memory, episodic log, consolidated brief.
On every turn, the brief is read first. Episodic log is appended to but not
re-injected into context after the day it happened.

This is the agent the controls (sliding window, vector retrieval, long context)
are compared against.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from openai import OpenAI

from recall_lab.config import (
    AGENT_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    WORKING_MAX_TURNS,
)
from recall_lab.memory.brief import Brief
from recall_lab.memory.episodic import Exchange, EpisodicLog
from recall_lab.memory.working import WorkingMemory


class RecallAgent:
    """The experimental agent the lab is built around."""

    def __init__(
        self,
        brief: Brief,
        log: EpisodicLog,
        working_window: int = WORKING_MAX_TURNS,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.brief = brief
        self.log = log
        self.working_window = working_window
        self.clock = clock or (lambda: datetime.now(UTC))
        self.recent_turns: list[dict] = []

    def respond(self, user_message: str) -> str:
        """Produce one response.

        1. Load the brief from disk.
        2. Compose working memory (brief + recent turns + user message).
        3. Call the model.
        4. Append the exchange to the episodic log.
        5. Update the recent turns buffer.
        """
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is required to run RecallAgent.")

        self.brief.load()
        working = WorkingMemory(
            user_message=user_message,
            brief_text=self.brief.render(),
            recent_turns=list(self.recent_turns),
            max_turns=self.working_window,
        )
        prompt = working.render()

        client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )
        completion = client.chat.completions.create(
            model=AGENT_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        response = completion.choices[0].message.content or ""

        self.log.append(
            Exchange(
                user=user_message,
                agent=response,
                timestamp=self.clock(),
            )
        )
        self.recent_turns.append({"user": user_message, "agent": response})
        self.recent_turns = self.recent_turns[-self.working_window :]
        return response
