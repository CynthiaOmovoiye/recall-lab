"""Interference check: does decay alone resolve contradictions?

Pits an old fact against a contradicting new fact through the activation
function. Runs two scenarios:

1. Clean slate: neither memory has been re-referenced.
2. Re-referenced: the old fact has been re-referenced 3 times in the last week.

The expected result is that pure decay favors the newer fact in scenario 1,
but repeated references can keep the older fact alive in scenario 2.

Run with:
    python -m recall_lab.consolidation.interference_check
"""

from __future__ import annotations

from datetime import datetime, timedelta

from recall_lab.consolidation.activation import MemoryTrace, activation


NOW = datetime(2026, 5, 20, 12, 0, 0)


def make_trace(
    turn_id: int,
    compression: str,
    base_salience: float,
    days_ago: int,
    refs_days_ago: list[int] | None = None,
) -> MemoryTrace:
    """Build a MemoryTrace anchored to NOW for reproducible checks."""
    created = NOW - timedelta(days=days_ago)
    refs = [NOW - timedelta(days=d) for d in (refs_days_ago or [])]

    return MemoryTrace(
        turn_id=turn_id,
        compression=compression,
        base_salience=base_salience,
        created_at=created,
        references=refs,
    )


def run_scenario(label: str, old: MemoryTrace, new: MemoryTrace) -> None:
    """Print activation scores for an old fact and a newer contradiction."""
    old_activation = activation(old, NOW)
    new_activation = activation(new, NOW)
    winner = "new" if new_activation > old_activation else "old"

    print(f"--- {label} ---")
    print(f"old: {old.compression!r}")
    print(
        f"  created {(NOW - old.created_at).days}d ago, "
        f"{len(old.references)} re-references"
    )
    print(f"  activation = {round(old_activation, 3)}")

    print(f"new: {new.compression!r}")
    print(
        f"  created {(NOW - new.created_at).days}d ago, "
        f"{len(new.references)} re-references"
    )
    print(f"  activation = {round(new_activation, 3)}")

    print(f"winner: {winner}")
    print()

def main() -> None:
    """Run the two interference scenarios."""
    old_clean = make_trace(
        turn_id=1,
        compression="User lives in Lagos",
        base_salience=0.8,
        days_ago=40,
    )
    new_clean = make_trace(
        turn_id=2,
        compression="User moved to Berlin in April 2026",
        base_salience=0.8,
        days_ago=5,
    )
    run_scenario("Scenario 1: clean slate", old_clean, new_clean)

    old_kept_alive = make_trace(
        turn_id=3,
        compression="User lives in Lagos",
        base_salience=0.8,
        days_ago=40,
        refs_days_ago=[6, 4, 1],
    )
    new_quiet = make_trace(
        turn_id=4,
        compression="User moved to Berlin in April 2026",
        base_salience=0.8,
        days_ago=5,
    )
    run_scenario("Scenario 2: old fact re-referenced", old_kept_alive, new_quiet)


if __name__ == "__main__":
    main()