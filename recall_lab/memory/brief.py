"""Consolidated brief: a living markdown document.

Read first every turn. Sections: stable facts, active intents, open commitments,
corrections, things to never repeat. Updated by the sleep job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from recall_lab.config import BRIEF_PATH


@dataclass
class Brief:
    """The living brief, read first on every turn."""

    path: Path = BRIEF_PATH
    sections: dict[str, list[str]] = field(default_factory=dict)

    def load(self) -> None:
        """Read the brief from disk into structured sections.

        Parses the markdown by ## headings. Empty sections marked "(none yet)"
        load as empty lists.
        """
        # TODO: parse markdown by ## headings
        raise NotImplementedError

    def render(self) -> str:
        """Return the brief as a single string for injection into context."""
        # TODO: serialise sections back to markdown
        raise NotImplementedError

    def add_entry(self, section: str, text: str) -> None:
        """Append a new semantic statement to a section.

        If the section does not exist, it is created. Duplicates are deduplicated
        on exact text match.
        """
        # TODO: implement
        raise NotImplementedError

    def decay(self, half_life_days: float) -> None:
        """Apply Ebbinghaus-style decay to brief entries.

        For Forgetting Curves Lab (secondary experiment). Off by default in
        Recall Lab proper.
        """
        # TODO: implement (Forgetting Curves Lab)
        raise NotImplementedError

    def save(self) -> None:
        """Write the rendered brief back to disk."""
        self.path.write_text(self.render(), encoding="utf-8")
