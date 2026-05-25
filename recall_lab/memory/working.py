"""Working memory: the current turn's active context.

Just the immediate input plus the consolidated brief plus a small number of
recent turns. No long history is injected here. Long history lives in the
episodic log on disk and never returns to context after the day it happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from recall_lab.config import WORKING_MAX_TURNS


@dataclass
class WorkingMemory:
    """The minimal context the agent uses for the current response."""

    user_message: str = ""
    brief_text: str = ""
    recent_turns: list[dict] = field(default_factory=list)
    max_turns: int = WORKING_MAX_TURNS

    def render(self) -> str:
        """Render working memory as a single text block for the model.

        Composition order:
        1. Brief (read first, sets the agent's identity and stable facts)
        2. Recent turns (last WORKING_MAX_TURNS)
        3. The new user message

        Returns a single string ready to pass as the user content in a chat call.
        """
        lines = [
            "You are the Recall Lab agent.",
            "Use the memory brief as durable memory.",
            "Use recent turns as short-term context.",
            "For personal facts, use memory. If a personal fact is missing, say you do not know.",
            (
                "The 'Past, no longer current' section is history. Use it only when the "
                "user asks about the past. Never treat it as a current fact."
            ),
            "For general knowledge or simple reasoning, answer normally.",
            "",
            "## Memory brief",
            self.brief_text.strip() or "(empty)",
            "",
            "## Recent turns",
        ]

        recent = self.recent_turns[-self.max_turns :]
        if recent:
            for turn in recent:
                lines.append(f"User: {turn.get('user', '')}")
                lines.append(f"Agent: {turn.get('agent', '')}")
        else:
            lines.append("(none)")

        lines.extend(
            [
                "",
                "## Current user message",
                self.user_message,
            ]
        )
        return "\n".join(lines).strip()
