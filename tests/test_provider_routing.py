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
    """Apply the shipped defaults: pin OpenAI (agent) and Anthropic (judge)."""
    monkeypatch.setattr(llm, "OPENROUTER_PROVIDER_ORDER", "OpenAI")
    monkeypatch.setattr(llm, "OPENROUTER_JUDGE_PROVIDER_ORDER", "Anthropic")
    monkeypatch.setattr(llm, "OPENROUTER_ALLOW_FALLBACKS", False)
    monkeypatch.setattr(llm, "OPENROUTER_IGNORE_PROVIDERS", "Azure")


def test_agent_model_is_pinned_to_openai_with_no_fallback(pinned: None) -> None:
    routing = llm.provider_routing("openai/gpt-4o-mini")
    assert routing == {
        "order": ["OpenAI"],
        "allow_fallbacks": False,
        "ignore": ["Azure"],
    }


def test_anthropic_judge_is_pinned_to_anthropic(pinned: None) -> None:
    # The judge model pins to Anthropic, not the OpenAI agent order, so it lands
    # on one provider every run instead of whichever Anthropic node load balances.
    routing = llm.provider_routing("anthropic/claude-sonnet-4.6")
    assert routing == {
        "order": ["Anthropic"],
        "allow_fallbacks": False,
        "ignore": ["Azure"],
    }


def test_unpinned_vendor_keeps_own_routing(pinned: None) -> None:
    # A model matching neither pin gets no order, just Azure excluded.
    routing = llm.provider_routing("meta/llama-3.1-8b")
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
    monkeypatch.setattr(llm, "OPENROUTER_JUDGE_PROVIDER_ORDER", "")
    monkeypatch.setattr(llm, "OPENROUTER_IGNORE_PROVIDERS", "")
    assert llm.provider_routing("openai/gpt-4o-mini") is None


def test_order_match_is_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "OPENROUTER_PROVIDER_ORDER", "openai")
    monkeypatch.setattr(llm, "OPENROUTER_JUDGE_PROVIDER_ORDER", "")
    monkeypatch.setattr(llm, "OPENROUTER_ALLOW_FALLBACKS", False)
    monkeypatch.setattr(llm, "OPENROUTER_IGNORE_PROVIDERS", "Azure")
    routing = llm.provider_routing("OpenAI/GPT-4O-Mini")
    assert routing["order"] == ["openai"]
