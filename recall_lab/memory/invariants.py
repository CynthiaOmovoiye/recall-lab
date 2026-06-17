"""Deterministic brief-invariant checks.

A cheap, model-free guard over the memory trace set. The LLM judge and
classifier can be wrong in ways no single test catches; these invariants catch
two structural failures mechanically, so a regression shows up without a live
campaign.

The trace analysis of the v16 campaign motivated this. A green fact appeared in
the active set and the past set at the same time (the sleep job promoted a new
trace without retiring the old duplicate), and the concern was raised that a
Never-Repeat safety item could be demoted with no correction to justify it.
Both are structural, both are checkable from the trace list alone.

Run over the live trace store, or in a test with hand-built traces. Pure
function: it reads traces and returns violations, it mutates nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

from recall_lab.consolidation.activation import MemoryTrace
from recall_lab.consolidation.contradiction import MemoryStatus

NEVER_REPEAT_SECTION = "never_repeat"


@dataclass
class Violation:
    """One invariant breach, with enough detail to locate it."""

    kind: str  # "active_past_overlap" | "never_repeat_demoted"
    detail: str
    turn_ids: list[int]


def _norm(text: str) -> str:
    """Compare compressions case- and whitespace-insensitively."""
    return " ".join(text.lower().split())


def check_invariants(traces: list[MemoryTrace]) -> list[Violation]:
    """Return every invariant violation in a trace set. Empty list means clean.

    Invariant 1, active and past are disjoint. The same fact must not be both
    current truth and history at once. A compression that appears in an ACTIVE
    trace must not also appear in a SUPERSEDED or ARCHIVED trace.

    Invariant 2, a Never-Repeat item is not demoted without a correction. A
    safety flag (never_repeat section) that has left the active set must have a
    real successor: a trace that supersedes it. A never_repeat trace that is
    superseded or archived with nothing pointing back at it was dropped with no
    correction to justify it, which is the safety-flag erosion case.
    """
    violations: list[Violation] = []

    active = [t for t in traces if t.status == MemoryStatus.ACTIVE.value]
    not_active = [t for t in traces if t.status != MemoryStatus.ACTIVE.value]
    active_by_text = {_norm(t.compression): t for t in active}

    # Invariant 1: no compression is simultaneously active and not-active.
    for past in not_active:
        hit = active_by_text.get(_norm(past.compression))
        if hit is not None:
            violations.append(
                Violation(
                    kind="active_past_overlap",
                    detail=(
                        f"compression {past.compression!r} is both active "
                        f"(turn {hit.turn_id}) and {past.status} (turn {past.turn_id})"
                    ),
                    turn_ids=sorted({hit.turn_id, past.turn_id}),
                )
            )

    # Invariant 2: a demoted never_repeat item must have a successor superseding it.
    superseding_targets = {
        t.supersedes for t in traces if t.supersedes is not None
    }
    for t in traces:
        if (
            t.section == NEVER_REPEAT_SECTION
            and t.status != MemoryStatus.ACTIVE.value
            and t.turn_id not in superseding_targets
        ):
            violations.append(
                Violation(
                    kind="never_repeat_demoted",
                    detail=(
                        f"never_repeat item {t.compression!r} (turn {t.turn_id}) is "
                        f"{t.status} but nothing supersedes it; a safety flag was "
                        f"dropped with no correction"
                    ),
                    turn_ids=[t.turn_id],
                )
            )

    return violations
