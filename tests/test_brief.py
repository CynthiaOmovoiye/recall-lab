"""Tests for the consolidated brief: render, add_entry, and the disk round trip."""

from __future__ import annotations

from datetime import datetime, timedelta

from recall_lab.consolidation.activation import MemoryTrace
from recall_lab.consolidation.contradiction import MemoryStatus
from recall_lab.memory.brief import DEFAULT_SECTIONS, Brief
from recall_lab.memory.traces import MemoryTraceStore


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


def test_render_brief_labels_past_lineage_oldest_to_newest(tmp_path) -> None:
    now = datetime(2026, 5, 28, 8, 0, 0)
    traces = [
        MemoryTrace(
            turn_id=1,
            compression="User ships to Lagos.",
            base_salience=0.8,
            created_at=now - timedelta(days=4),
            status=MemoryStatus.SUPERSEDED.value,
        ),
        MemoryTrace(
            turn_id=2,
            compression="User ships to Berlin.",
            base_salience=0.8,
            created_at=now - timedelta(days=3),
            status=MemoryStatus.SUPERSEDED.value,
            supersedes=1,
        ),
        MemoryTrace(
            turn_id=3,
            compression="User ships to Nairobi.",
            base_salience=0.8,
            created_at=now - timedelta(days=2),
        ),
    ]
    store = MemoryTraceStore(path=tmp_path / "traces.jsonl")
    brief = Brief(path=tmp_path / "brief.md")

    store.save(traces)
    store.render_brief(brief, now=now)

    brief.load()
    past = brief.sections["Past, no longer current"]
    assert past == [
        "Earliest past: User ships to Lagos.",
        "Most recent past before current: User ships to Berlin.",
    ]
    assert "User ships to Nairobi." in brief.sections["Stable facts about the user"]
