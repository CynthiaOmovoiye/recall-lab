"""Consolidated brief: a living markdown document.

Read first every turn. Sections: stable facts, active intents, open commitments,
corrections, things to never repeat, and past facts that have been superseded.
Updated by the sleep job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from recall_lab.config import BRIEF_PATH


DEFAULT_SECTIONS = [
    "Stable facts about the user",
    "Active intents",
    "Open commitments",
    "Corrections",
    "Things to never repeat",
    "Past, no longer current",
]

SECTION_ALIASES = {
    "stable_facts": "Stable facts about the user",
    "active_intents": "Active intents",
    "open_commitments": "Open commitments",
    "corrections": "Corrections",
    "never_repeat": "Things to never repeat",
    "past": "Past, no longer current",
}


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
        self.sections = {section: [] for section in DEFAULT_SECTIONS}

        if not self.path.exists():
            return

        current_section: str | None = None
        for raw_line in self.path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()

            if line.startswith("## "):
                current_section = line.removeprefix("## ").strip()
                self.sections.setdefault(current_section, [])
                continue

            if current_section is None or not line or line == "(none yet)":
                continue

            if line.startswith("- "):
                line = line[2:].strip()

            if line and line not in self.sections[current_section]:
                self.sections[current_section].append(line)

    def render(self) -> str:
        """Return the brief as a single string for injection into context."""
        if not self.sections:
            self.sections = {section: [] for section in DEFAULT_SECTIONS}

        lines = [
            "# Brief",
            "",
            (
                "The living memory brief for Recall Lab. The agent reads this first on every turn. "
                "The sleep job updates it nightly."
            ),
            "",
        ]

        ordered = [*DEFAULT_SECTIONS]
        ordered.extend(section for section in self.sections if section not in DEFAULT_SECTIONS)

        for section in ordered:
            entries = self.sections.get(section, [])
            lines.append(f"## {section}")
            lines.append("")
            if entries:
                lines.extend(f"- {entry}" for entry in entries)
            else:
                lines.append("(none yet)")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def add_entry(self, section: str, text: str) -> None:
        """Append a new semantic statement to a section.

        If the section does not exist, it is created. Duplicates are deduplicated
        on exact text match.
        """
        section_name = SECTION_ALIASES.get(section, section)
        clean_text = text.strip()
        if not clean_text:
            return

        self.sections.setdefault(section_name, [])
        if clean_text not in self.sections[section_name]:
            self.sections[section_name].append(clean_text)

    def decay(self, half_life_days: float) -> None:
        """Apply Ebbinghaus-style decay to brief entries.

        For Forgetting Curves Lab (secondary experiment). Off by default in
        Recall Lab proper.
        """
        raise NotImplementedError("Brief decay is reserved for the Forgetting Curves Lab.")

    def save(self) -> None:
        """Write the rendered brief back to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self.render(), encoding="utf-8")
