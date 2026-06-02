"""Tests for the raw episodic read-time-judge control.

The model call needs an API key, so `respond` is not exercised. What matters
here is the raw-history plumbing: every statement is kept verbatim, nothing is
compressed or superseded, and the full log renders oldest-first. The key guard
is checked too.
"""

from __future__ import annotations

import pytest

from recall_lab.controls.episodic import EpisodicJudgeAgent


def test_empty_history_renders_placeholder() -> None:
    agent = EpisodicJudgeAgent()
    assert agent._render_history() == "(no history yet)"
    assert agent.last_input_tokens == 0


def test_history_is_kept_verbatim_oldest_first() -> None:
    agent = EpisodicJudgeAgent()
    agent.history = [
        {"user": "Ship to Lagos.", "agent": "Okay, Lagos."},
        {"user": "Actually, ship to Berlin.", "agent": "Updated to Berlin."},
    ]
    rendered = agent._render_history()
    # Both statements survive raw; the correction does not erase the original.
    assert "Lagos" in rendered
    assert "Berlin" in rendered
    # Oldest first: Lagos appears before Berlin.
    assert rendered.index("Lagos") < rendered.index("Berlin")


def test_nothing_is_compressed_or_dropped() -> None:
    agent = EpisodicJudgeAgent()
    for i in range(10):
        agent.history.append({"user": f"statement {i}", "agent": f"reply {i}"})
    rendered = agent._render_history()
    # Every one of the 10 statements is present; raw means raw.
    for i in range(10):
        assert f"statement {i}" in rendered
        assert f"reply {i}" in rendered


def test_respond_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("recall_lab.controls.episodic.OPENROUTER_API_KEY", "")
    agent = EpisodicJudgeAgent()
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        agent.respond("hello")
