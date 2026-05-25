"""Tests for the consolidated brief: render, add_entry, and the disk round trip."""

from __future__ import annotations

from recall_lab.memory.brief import DEFAULT_SECTIONS, Brief


def test_render_includes_every_default_section() -> None:
    rendered = Brief().render()
    for section in DEFAULT_SECTIONS:
        assert f"## {section}" in rendered


def test_add_entry_maps_a_raw_key_to_its_human_section() -> None:
    brief = Brief()
    brief.add_entry("stable_facts", "User's favorite color is blue.")
    assert "User's favorite color is blue." in brief.sections["Stable facts about the user"]


def test_add_entry_deduplicates_exact_text() -> None:
    brief = Brief()
    brief.add_entry("corrections", "Ship to Berlin.")
    brief.add_entry("corrections", "Ship to Berlin.")
    assert brief.sections["Corrections"].count("Ship to Berlin.") == 1


def test_load_round_trips_through_disk(tmp_path) -> None:
    original = Brief(path=tmp_path / "brief.md")
    original.add_entry("stable_facts", "User ships to Berlin.")
    original.add_entry("past", "Previously: User shipped to Lagos.")
    original.save()

    reloaded = Brief(path=tmp_path / "brief.md")
    reloaded.load()
    assert "User ships to Berlin." in reloaded.sections["Stable facts about the user"]
    assert "Previously: User shipped to Lagos." in reloaded.sections["Past, no longer current"]
