"""The experimental Recall Lab agent.

Uses three memory layers: working memory, episodic log, consolidated brief.
On every turn, the brief is read first. Episodic log is appended to but not
re-injected into context after the day it happened.

This is the agent the controls (sliding window, vector retrieval, long context)
are compared against.
"""

from __future__ import annotations

from datetime import datetime

from recall_lab.memory.brief import Brief
from recall_lab.memory.episodic import Exchange, EpisodicLog
from recall_lab.memory.working import WorkingMemory


class RecallAgent:
    """The experimental agent the lab is built around."""

    def __init__(self, brief: Brief, log: EpisodicLog) -> None:
        self.brief = brief
        self.log = log
        self.recent_turns: list[dict] = []

    def respond(self, user_message: str) -> str:
        """Produce one response.

        1. Load the brief from disk.
        2. Compose working memory (brief + recent turns + user message).
        3. Call the model.
        4. Append the exchange to the episodic log.
        5. Update the recent turns buffer.
        """
        self.brief.load()
        working = WorkingMemory(
            user_message=user_message,
            brief_text=self.brief.render(),
            recent_turns=list(self.recent_turns),
        )
        prompt = working.render()
        # TODO: call the model with `prompt`. Return its text response.
        response: str = ""
        raise NotImplementedError(
            "Wire up the OpenRouter client and return its response."
        )
        # Below is left intentionally as the eventual completed flow:
        self.log.append(
            Exchange(
                user=user_message,
                agent=response,
                timestamp=datetime.utcnow(),
            )
        )
        self.recent_turns.append({"user": user_message, "agent": response})
        return response
