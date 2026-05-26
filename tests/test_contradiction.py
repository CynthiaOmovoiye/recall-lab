"""Tests for memory validity state: supersede, revert, current_traces.

These functions mutate trace objects in memory. No model call is made here.
classify() is the only part of the module that calls a model, and it is not
exercised in this suite.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from recall_lab.consolidation.activation import MemoryTrace
from recall_lab.consolidation.contradiction import (
    CONFIRM,
    CORRECT,
    ConflictVerdict,
    MemoryStatus,
    current_traces,
    revert,
    supersede,
)

NOW = datetime(2026, 5, 25, 12, 0, 0)


def _trace(turn_id: int, text: str) -> MemoryTrace:
    return MemoryTrace(
        turn_id=turn_id,
        compression=text,
        base_salience=0.8,
        created_at=NOW,
    )


def test_supersede_marks_the_old_trace_and_links_the_new_one() -> None:
    old = _trace(1, "User lives in Lagos")
    new = _trace(2, "User lives in Berlin")

    supersede(old, new, ConflictVerdict(label=CORRECT, reason="city changed"), now=NOW)

    assert old.status == MemoryStatus.SUPERSEDED.value
    assert new.supersedes == 1


def test_supersede_rejects_a_non_correct_verdict() -> None:
    old = _trace(1, "User lives in Lagos")
    new = _trace(2, "User still lives in Lagos")

    with pytest.raises(ValueError):
        supersede(old, new, ConflictVerdict(label=CONFIRM, reason="same fact"), now=NOW)


def test_revert_undoes_one_supersede() -> None:
    old = _trace(1, "User lives in Lagos")
    new = _trace(2, "User lives in Berlin")
    transition = supersede(
        old, new, ConflictVerdict(label=CORRECT, reason="city changed"), now=NOW
    )

    revert(transition, {1: old, 2: new})

    assert old.status == MemoryStatus.ACTIVE.value
    assert new.supersedes is None


def test_current_traces_keeps_only_active_traces() -> None:
    active = _trace(1, "User lives in Berlin")
    superseded = _trace(2, "User lives in Lagos")
    superseded.status = MemoryStatus.SUPERSEDED.value

    assert current_traces([active, superseded]) == [active]
