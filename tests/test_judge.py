"""Tests for the salience judge's promotion gate.

The model call needs an API key, so `score_exchange` is not exercised. What
matters here is `_normalise_verdict`: it must refuse to promote a turn that
sets no current value, even when the model scores it highly. This is the
primary fix for the v16 blue stale-re-assertion bug, where a fond reminiscence
about an old colour ("green is still such a beautiful color, I always come back
to it") was scored 0.55 and promoted as a preference, then superseded the
current blue.
"""

from __future__ import annotations

from recall_lab.consolidation.judge import _normalise_verdict


def test_value_setting_fact_is_promotable() -> None:
    verdict = _normalise_verdict(
        {
            "score": 0.9,
            "reason": "User set their shipping city.",
            "value_setting": True,
            "suggested_brief_section": "stable_facts",
            "suggested_statement": "User ships to Berlin.",
        }
    )
    assert verdict.value_setting is True
    assert verdict.suggested_brief_section == "stable_facts"
    assert verdict.suggested_statement == "User ships to Berlin."


def test_reminiscence_is_not_promoted_even_when_scored_high() -> None:
    # The v16 failure shape: high score, but the turn sets no current value.
    verdict = _normalise_verdict(
        {
            "score": 0.55,
            "reason": "User expresses a stable aesthetic preference for green.",
            "value_setting": False,
            "suggested_brief_section": "stable_facts",
            "suggested_statement": "User's favorite color is green.",
        }
    )
    # Score may stay informational, but nothing is filed: no section, no statement.
    assert verdict.value_setting is False
    assert verdict.suggested_brief_section is None
    assert verdict.suggested_statement is None


def test_missing_value_setting_defaults_to_promotable() -> None:
    # Back-compat: a payload without the field behaves as before (promotable).
    verdict = _normalise_verdict(
        {
            "score": 0.8,
            "reason": "User stated a durable fact.",
            "suggested_brief_section": "stable_facts",
            "suggested_statement": "User's daughter is allergic to shellfish.",
        }
    )
    assert verdict.value_setting is True
    assert verdict.suggested_statement == "User's daughter is allergic to shellfish."


def test_value_setting_false_as_string_is_respected() -> None:
    # Models sometimes emit a string instead of a JSON bool; "false" must count.
    verdict = _normalise_verdict(
        {
            "score": 0.7,
            "reason": "Reminiscence.",
            "value_setting": "false",
            "suggested_brief_section": "stable_facts",
            "suggested_statement": "User likes green.",
        }
    )
    assert verdict.value_setting is False
    assert verdict.suggested_statement is None


def test_low_score_still_blocks_promotion() -> None:
    # The existing score gate is unchanged.
    verdict = _normalise_verdict(
        {
            "score": 0.2,
            "reason": "Chit-chat.",
            "value_setting": True,
            "suggested_brief_section": "stable_facts",
            "suggested_statement": "Thanks.",
        }
    )
    assert verdict.suggested_brief_section is None
    assert verdict.suggested_statement is None
