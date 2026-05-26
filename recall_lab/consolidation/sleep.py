"""The sleep job: nightly consolidation.

Steps:
1. Pull the day's episodic exchanges.
2. Score each with the judge.
3. Convert high-salience exchanges into memory traces.
4. Check whether each new trace confirms, corrects, or is unrelated to memory.
5. Render active traces into the consolidated brief.
"""

from __future__ import annotations

from datetime import datetime

from recall_lab.config import CONTRADICTION_COMPARE_LIMIT, SALIENCE_THRESHOLD
from recall_lab.consolidation.activation import MemoryTrace, activation
from recall_lab.consolidation.contradiction import (
    CONFIRM,
    CORRECT,
    append_transition,
    classify,
    supersede,
)
from recall_lab.consolidation.judge import SalienceVerdict, score_exchange
from recall_lab.memory.brief import Brief
from recall_lab.memory.episodic import EpisodicLog, Exchange
from recall_lab.memory.traces import MemoryTraceStore


def _trace_from_verdict(exchange: Exchange, verdict: SalienceVerdict) -> MemoryTrace:
    """Create a machine-readable memory trace from a salience verdict."""
    if exchange.id is None:
        raise ValueError("Promoted exchanges need an id before they become memory traces.")
    if not verdict.suggested_statement or not verdict.suggested_brief_section:
        raise ValueError("Promoted exchanges need a statement and brief section.")

    return MemoryTrace(
        turn_id=exchange.id,
        compression=verdict.suggested_statement,
        base_salience=verdict.score,
        created_at=exchange.timestamp,
        section=verdict.suggested_brief_section,
    )


def _integrate_trace(
    new_trace: MemoryTrace,
    traces: list[MemoryTrace],
    now: datetime,
) -> str:
    """Merge one new trace into memory using contradiction and activation.

    Returns a compact action label for reporting:
    - added: no active memory conflicted with the new trace
    - confirmed: an active memory matched, so it got a reference bump
    - corrected: an active memory was superseded by the new trace
    """
    active = [trace for trace in traces if trace.status == "active"]
    active = sorted(active, key=lambda trace: activation(trace, now), reverse=True)
    active = active[:CONTRADICTION_COMPARE_LIMIT]

    for old_trace in active:
        verdict = classify(old_trace.compression, new_trace.compression)

        if verdict.label == CONFIRM:
            old_trace.references.append(new_trace.created_at)
            return "confirmed"

        if verdict.label == CORRECT:
            transition = supersede(old_trace, new_trace, verdict, now=now)
            append_transition(transition)
            traces.append(new_trace)
            return "corrected"

    traces.append(new_trace)
    return "added"


def run_sleep_job(
    day: datetime,
    brief: Brief,
    log: EpisodicLog,
    trace_store: MemoryTraceStore | None = None,
) -> dict:
    """Run one consolidation pass over a single day's exchanges.

    Returns a summary dict suitable for appending to research-log.md.
    """
    trace_store = trace_store or MemoryTraceStore()
    traces = trace_store.load()
    exchanges = log.fetch_day(day)

    scored = 0
    promoted = 0
    added = 0
    confirmed = 0
    corrected = 0
    skipped = 0

    for ex in exchanges:
        if ex.promoted:
            continue

        scored += 1
        verdict = score_exchange(ex)
        if not (
            verdict.score >= SALIENCE_THRESHOLD
            and verdict.suggested_statement
            and verdict.suggested_brief_section
        ):
            skipped += 1
            continue

        new_trace = _trace_from_verdict(ex, verdict)
        action = _integrate_trace(new_trace, traces, now=day)

        if action == "added":
            added += 1
        elif action == "confirmed":
            confirmed += 1
        elif action == "corrected":
            corrected += 1

        if ex.id is not None:
            log.mark_promoted(ex.id, verdict.score)
        promoted += 1

    trace_store.save(traces)
    trace_store.render_brief(brief, now=day)

    return {
        "day": day.isoformat(),
        "exchanges": len(exchanges),
        "scored": scored,
        "promoted": promoted,
        "added": added,
        "confirmed": confirmed,
        "corrected": corrected,
        "skipped": skipped,
        "compare_limit": CONTRADICTION_COMPARE_LIMIT,
        "active_traces": len(trace_store.active()),
        "total_traces": len(trace_store.load()),
        "threshold": SALIENCE_THRESHOLD,
    }
