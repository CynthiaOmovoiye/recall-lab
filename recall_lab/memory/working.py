"""Working memory: the current turn's active context.

Just the immediate input plus the consolidated brief plus a small number of
recent turns. No long history is injected here. Long history lives in the
episodic log on disk and never returns to context after the day it happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WorkingMemory:
    """The minimal context the agent uses for the current response."""

    user_message: str = ""
    brief_text: str = ""
    recent_turns: list[dict] = field(default_factory=list)

    def render(self) -> str:
        """Render working memory as a single text block for the model.

        Composition order:
        1. Brief (read first, sets the agent's identity and stable facts)
        2. Recent turns (last WORKING_MAX_TURNS)
        3. The new user message

        Returns a single string ready to pass as the user content in a chat call.
        """
        # TODO: implement composition order: brief, recent_turns, user_message
        raise NotImplementedError
