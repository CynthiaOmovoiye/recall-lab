"""Tests for the ACT-R activation function.

Activation is a pure function of a trace and a clock. No I/O, no network.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from recall_lab.consolidation.activation import MemoryTrace, activation, should_keep

NOW = datetime(2026, 5, 25, 12, 0, 0)


def _trace(
    days_ago: float,
    base_salience: float = 0.5,
    refs: list[float] | None = None,
) -> MemoryTrace:
    """Build a trace anchored to NOW for reproducible checks."""
    return MemoryTrace(
        turn_id=1,
        compression="a fact",
        base_salience=base_salience,
        created_at=NOW - timedelta(days=days_ago),
        references=[NOW - timedelta(days=d) for d in (refs or [])],
    )


def test_recent_trace_outranks_an_old_one() -> None:
    recent = _trace(days_ago=1)
    old = _trace(days_ago=60)
    assert activation(recent, NOW) > activation(old, NOW)


def test_re_references_raise_activation() -> None:
    quiet = _trace(days_ago=10)
    referenced = _trace(days_ago=10, refs=[5, 2, 1])
    assert activation(referenced, NOW) > activation(quiet, NOW)


def test_higher_base_salience_raises_activation() -> None:
    low = _trace(days_ago=5, base_salience=0.2)
    high = _trace(days_ago=5, base_salience=0.9)
    assert activation(high, NOW) > activation(low, NOW)


def test_should_keep_drops_a_stale_low_salience_trace() -> None:
    fresh = _trace(days_ago=1, base_salience=0.9)
    stale = _trace(days_ago=120, base_salience=0.1)
    assert should_keep(fresh, NOW) is True
    assert should_keep(stale, NOW) is False
