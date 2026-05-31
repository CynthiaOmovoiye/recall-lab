"""Tests for OpenRouter provider routing.

The pin must hold for the agent model and must not break the Anthropic judge.
These tests drive `provider_routing` directly with patched config values, so
they need no API key and make no network call.
"""

from __future__ import annotations

import pytest

from recall_lab import llm


@pytest.fixture
def pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Apply the shipped defaults: pin OpenAI, no fallback, ignore Azure."""
    monkeypatch.setattr(llm, "OPENROUTER_PROVIDER_ORDER", "OpenAI")
    monkeypatch.setattr(llm, "OPENROUTER_ALLOW_FALLBACKS", False)
    monkeypatch.setattr(llm, "OPENROUTER_IGNORE_PROVIDERS", "Azure")


def test_agent_model_is_pinned_to_openai_with_no_fallback(pinned: None) -> None:
    routing = llm.provider_routing("openai/gpt-4o-mini")
    assert routing == {
        "order": ["OpenAI"],
        "allow_fallbacks": False,
        "ignore": ["Azure"],
    }


def test_anthropic_judge_is_not_pinned_to_openai(pinned: None) -> None:
    # The OpenAI order must NOT be applied to an Anthropic model, or the call
    # would error with fallbacks off. Azure stays excluded; no order, no fallback key.
    routing = llm.provider_routing("anthropic/claude-sonnet-4.6")
    assert routing == {"ignore": ["Azure"]}
    assert "order" not in routing


def test_unknown_model_still_gets_the_pin(pinned: None) -> None:
    # When the model is unknown, apply the pin rather than silently dropping it.
    routing = llm.provider_routing(None)
    assert routing["order"] == ["OpenAI"]
    assert routing["allow_fallbacks"] is False


def test_azure_excluded_for_every_model(pinned: None) -> None:
    for model in ["openai/gpt-4o-mini", "anthropic/claude-sonnet-4.6", "meta/llama"]:
        assert llm.provider_routing(model)["ignore"] == ["Azure"]


def test_empty_config_means_no_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "OPENROUTER_PROVIDER_ORDER", "")
    monkeypatch.setattr(llm, "OPENROUTER_IGNORE_PROVIDERS", "")
    assert llm.provider_routing("openai/gpt-4o-mini") is None


def test_order_match_is_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "OPENROUTER_PROVIDER_ORDER", "openai")
    monkeypatch.setattr(llm, "OPENROUTER_ALLOW_FALLBACKS", False)
    monkeypatch.setattr(llm, "OPENROUTER_IGNORE_PROVIDERS", "Azure")
    routing = llm.provider_routing("OpenAI/GPT-4O-Mini")
    assert routing["order"] == ["openai"]
