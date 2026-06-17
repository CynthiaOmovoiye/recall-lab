"""Tests for the deterministic latest-value resolver control.

The model answer call needs an API key, so `respond` is not exercised. What
matters here is the resolve logic, which is pure Python: extracted pairs are
grouped by attribute, ordered by (timestamp, turn), and the current value is the
most recent one. A deterministic extractor is injected so no network or key is
needed. The JSON parser and the key guard are checked too.
"""

from __future__ import annotations

import pytest

from recall_lab.controls.deterministic import (
    DeterministicResolverAgent,
    _parse_pairs,
    _Statement,
)


def _store(agent: DeterministicResolverAgent, attribute: str, value: str, ts: float) -> None:
    """Push one extracted statement at a given timestamp, as respond would."""
    agent._turn += 1
    agent.statements.append(
        _Statement(attribute=attribute, value=value, turn=agent._turn, added_at=ts)
    )


def test_latest_timestamp_is_the_current_value() -> None:
    agent = DeterministicResolverAgent()
    _store(agent, "shipping city", "Lagos", ts=1.0)
    _store(agent, "shipping city", "Berlin", ts=2.0)
    _store(agent, "shipping city", "Nairobi", ts=3.0)
    resolved = agent._resolve()["shipping city"]
    # Oldest-first ordering; the last entry is the current value.
    assert [s.value for s in resolved] == ["Lagos", "Berlin", "Nairobi"]
    assert resolved[-1].value == "Nairobi"


def test_history_is_preserved_for_past_questions() -> None:
    agent = DeterministicResolverAgent()
    _store(agent, "shipping city", "Lagos", ts=1.0)
    _store(agent, "shipping city", "Berlin", ts=2.0)
    table = agent._render_table()
    # Current value plus the earlier one both appear, so first/previous are answerable.
    assert "current = Berlin" in table
    assert "Lagos" in table


def test_separate_attributes_do_not_collide() -> None:
    agent = DeterministicResolverAgent()
    _store(agent, "shipping city", "Berlin", ts=2.0)
    _store(agent, "favorite color", "green", ts=1.0)
    resolved = agent._resolve()
    assert resolved["shipping city"][-1].value == "Berlin"
    assert resolved["favorite color"][-1].value == "green"


def test_turn_breaks_ties_when_timestamps_equal() -> None:
    # Same-day statements share a timestamp; insertion order must still resolve.
    agent = DeterministicResolverAgent()
    _store(agent, "shipping city", "Berlin", ts=5.0)
    _store(agent, "shipping city", "Nairobi", ts=5.0)
    assert agent._resolve()["shipping city"][-1].value == "Nairobi"


def test_empty_table_renders_placeholder() -> None:
    agent = DeterministicResolverAgent()
    assert agent._render_table() == "(nothing recorded yet)"


def test_injected_extractor_drives_storage() -> None:
    # An injected extractor lets the whole respond path run offline up to the
    # model call; here we check extraction feeds the table.
    pairs = {"My favorite color is green.": [{"attribute": "favorite color", "value": "green"}]}
    agent = DeterministicResolverAgent(extractor=lambda m: pairs.get(m, []))
    for pair in agent._extract("My favorite color is green."):
        agent.statements.append(
            _Statement(attribute=pair["attribute"], value=pair["value"], turn=1)
        )
    assert "current = green" in agent._render_table()


def test_parse_pairs_tolerates_json_fence() -> None:
    raw = '```json\n[{"attribute": "shipping city", "value": "Berlin"}]\n```'
    assert _parse_pairs(raw) == [{"attribute": "shipping city", "value": "Berlin"}]


def test_parse_pairs_returns_empty_on_garbage() -> None:
    assert _parse_pairs("not json at all") == []
    assert _parse_pairs('{"not": "a list"}') == []


def test_respond_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("recall_lab.controls.deterministic.OPENROUTER_API_KEY", "")
    agent = DeterministicResolverAgent(extractor=lambda m: [])
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        agent.respond("hello")
