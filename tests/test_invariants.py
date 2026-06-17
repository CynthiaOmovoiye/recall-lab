"""Tests for the deterministic brief-invariant linter.

Pure, offline. Hand-build trace sets that violate each invariant and confirm
the linter catches them, plus a clean set that passes.
"""

from __future__ import annotations

from datetime import datetime

from recall_lab.consolidation.activation import MemoryTrace
from recall_lab.consolidation.contradiction import MemoryStatus
from recall_lab.memory.invariants import check_invariants

NOW = datetime(2026, 6, 17, 12, 0, 0)


def _trace(turn_id: int, text: str, *, status: str = "active", section: str = "stable_facts",
           supersedes: int | None = None) -> MemoryTrace:
    return MemoryTrace(
        turn_id=turn_id,
        compression=text,
        base_salience=0.8,
        created_at=NOW,
        status=status,
        section=section,
        supersedes=supersedes,
    )


def test_clean_set_has_no_violations() -> None:
    traces = [
        _trace(1, "User's favorite color is blue."),
        _trace(2, "User's favorite color is green.", status=MemoryStatus.SUPERSEDED.value),
        _trace(3, "User's daughter is allergic to shellfish.", section="never_repeat"),
    ]
    assert check_invariants(traces) == []


def test_active_past_overlap_is_caught() -> None:
    # The v16 duplicate: same green fact active AND superseded at once.
    traces = [
        _trace(1, "User's favorite color is green."),
        _trace(2, "User's favorite color is green.", status=MemoryStatus.SUPERSEDED.value),
    ]
    v = check_invariants(traces)
    assert len(v) == 1
    assert v[0].kind == "active_past_overlap"
    assert set(v[0].turn_ids) == {1, 2}


def test_overlap_match_is_case_and_space_insensitive() -> None:
    traces = [
        _trace(1, "User ships to Berlin."),
        _trace(2, "user ships to   berlin.", status=MemoryStatus.ARCHIVED.value),
    ]
    assert any(x.kind == "active_past_overlap" for x in check_invariants(traces))


def test_never_repeat_demoted_without_successor_is_caught() -> None:
    # A safety flag superseded with nothing pointing back at it.
    traces = [
        _trace(1, "Never suggest shellfish.", status=MemoryStatus.SUPERSEDED.value,
               section="never_repeat"),
    ]
    v = check_invariants(traces)
    assert len(v) == 1
    assert v[0].kind == "never_repeat_demoted"
    assert v[0].turn_ids == [1]


def test_never_repeat_demoted_with_successor_is_allowed() -> None:
    # If a real correction superseded it, that is a legitimate transition.
    traces = [
        _trace(1, "Never suggest shellfish.", status=MemoryStatus.SUPERSEDED.value,
               section="never_repeat"),
        _trace(2, "Never suggest peanuts.", section="never_repeat", supersedes=1),
    ]
    assert check_invariants(traces) == []


def test_active_never_repeat_is_fine() -> None:
    traces = [_trace(1, "Never suggest shellfish.", section="never_repeat")]
    assert check_invariants(traces) == []
