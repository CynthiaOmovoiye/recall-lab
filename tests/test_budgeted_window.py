"""Tests for the budget-bounded sliding window's context selection.

The model call needs an API key, so `respond` is not exercised. What matters
here is `_select_context`: it must keep the most recent turns that fit under
the input-token budget, always include the current message's cost, and never
overflow.
"""

from __future__ import annotations

import pytest

from recall_lab.controls.budgeted import BudgetedSlidingWindowAgent
from recall_lab.eval.metrics import estimate_tokens


def _budget_of(messages: list[dict]) -> int:
    return sum(estimate_tokens(m["content"]) for m in messages)


def test_budget_must_be_positive() -> None:
    with pytest.raises(ValueError):
        BudgetedSlidingWindowAgent(input_token_budget=0)


def test_empty_history_selects_nothing() -> None:
    agent = BudgetedSlidingWindowAgent(input_token_budget=100)
    assert agent._select_context("hello there") == []


def test_recent_turns_are_preferred_and_kept_in_order() -> None:
    agent = BudgetedSlidingWindowAgent(input_token_budget=1000)
    agent.history = [
        {"user": "oldest user turn", "agent": "oldest agent reply"},
        {"user": "middle user turn", "agent": "middle agent reply"},
        {"user": "newest user turn", "agent": "newest agent reply"},
    ]
    selected = agent._select_context("what now?")
    # All three fit under a generous budget, returned oldest-first.
    assert [turn["user"] for turn in selected] == [
        "oldest user turn",
        "middle user turn",
        "newest user turn",
    ]


def test_tight_budget_drops_oldest_first() -> None:
    agent = BudgetedSlidingWindowAgent(input_token_budget=1000)
    long_turn = {"user": "x" * 2000, "agent": "y" * 2000}
    recent_turn = {"user": "ship to Nairobi", "agent": "okay, Nairobi"}
    agent.history = [long_turn, recent_turn]

    selected = agent._select_context("where to?")
    # The 1000-char-ish old turn cannot fit; the recent short turn can.
    assert selected == [recent_turn]


def test_selection_never_exceeds_budget() -> None:
    budget = 50
    agent = BudgetedSlidingWindowAgent(input_token_budget=budget)
    agent.history = [
        {"user": f"turn {i} user content here", "agent": f"turn {i} agent reply here"}
        for i in range(20)
    ]
    user_message = "current question"
    selected = agent._select_context(user_message)

    messages = []
    for turn in selected:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["agent"]})
    messages.append({"role": "user", "content": user_message})

    assert _budget_of(messages) <= budget
