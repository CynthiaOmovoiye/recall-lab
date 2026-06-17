"""Tests for sleep-job trace integration.

These tests patch the model-facing classifier, so they exercise the merge
policy without spending API calls.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from recall_lab.consolidation.activation import MemoryTrace
from recall_lab.consolidation.contradiction import (
    CONFIRM,
    CORRECT,
    ConflictVerdict,
    MemoryStatus,
)
from recall_lab.consolidation import sleep
from recall_lab.consolidation.sleep import BriefInvariantError, run_sleep_job
from recall_lab.memory.brief import Brief
from recall_lab.memory.episodic import EpisodicLog
from recall_lab.memory.traces import MemoryTraceStore


NOW = datetime(2026, 5, 25, 12, 0, 0)


def _seed_store_with_overlap(path) -> MemoryTraceStore:
    """A trace store violating invariant 1: same fact active and superseded."""
    store = MemoryTraceStore(path=path)
    store.save(
        [
            MemoryTrace(turn_id=1, compression="User's favorite color is green.",
                        base_salience=0.8, created_at=NOW),
            MemoryTrace(turn_id=2, compression="User's favorite color is green.",
                        base_salience=0.8, created_at=NOW,
                        status=MemoryStatus.SUPERSEDED.value),
        ]
    )
    return store


def _empty_components(tmp_path):
    """A brief, an empty episodic log, and the tmp paths for a no-exchange run."""
    brief = Brief(path=tmp_path / "brief.md")
    brief.load()
    log = EpisodicLog(db_path=tmp_path / "log.db")  # empty: no exchanges, no model calls
    return brief, log


def test_sleep_job_reports_invariant_violation_without_raising(tmp_path) -> None:
    store = _seed_store_with_overlap(tmp_path / "traces.jsonl")
    brief, log = _empty_components(tmp_path)
    summary = run_sleep_job(NOW, brief, log, trace_store=store)
    kinds = [v["kind"] for v in summary["invariant_violations"]]
    assert "active_past_overlap" in kinds


def test_sleep_job_strict_raises_on_violation(tmp_path) -> None:
    store = _seed_store_with_overlap(tmp_path / "traces.jsonl")
    brief, log = _empty_components(tmp_path)
    with pytest.raises(BriefInvariantError):
        run_sleep_job(NOW, brief, log, trace_store=store, strict=True)


def test_sleep_job_clean_store_reports_no_violations(tmp_path) -> None:
    store = MemoryTraceStore(path=tmp_path / "traces.jsonl")
    store.save(
        [MemoryTrace(turn_id=1, compression="User's favorite color is blue.",
                     base_salience=0.8, created_at=NOW)]
    )
    brief, log = _empty_components(tmp_path)
    summary = run_sleep_job(NOW, brief, log, trace_store=store)
    assert summary["invariant_violations"] == []


def _trace(turn_id: int, text: str, hours_ago: int = 0) -> MemoryTrace:
    return MemoryTrace(
        turn_id=turn_id,
        compression=text,
        base_salience=0.8,
        created_at=NOW - timedelta(hours=hours_ago),
    )


def test_integrate_trace_corrects_all_active_stale_traces(monkeypatch) -> None:
    lagos_a = _trace(1, "User ships to Lagos", hours_ago=48)
    lagos_b = _trace(2, "User wants orders sent to Lagos", hours_ago=24)
    nairobi = _trace(3, "User ships to Nairobi")
    traces = [lagos_a, lagos_b]

    def fake_classify(old: str, new: str) -> ConflictVerdict:
        return ConflictVerdict(label=CORRECT, reason=f"{old} changed to {new}")

    transitions = []
    monkeypatch.setattr(sleep, "classify", fake_classify)
    monkeypatch.setattr(sleep, "append_transition", transitions.append)

    action = sleep._integrate_trace(nairobi, traces, now=NOW)

    assert action == "corrected"
    assert lagos_a.status == MemoryStatus.SUPERSEDED.value
    assert lagos_b.status == MemoryStatus.SUPERSEDED.value
    assert nairobi in traces
    assert len(transitions) == 2


def test_integrate_trace_uses_stored_canonical_trace_when_confirming_and_correcting(
    monkeypatch,
) -> None:
    lagos = _trace(1, "User ships to Lagos", hours_ago=48)
    berlin = _trace(2, "User ships to Berlin", hours_ago=24)
    duplicate_berlin = _trace(3, "User ships to Berlin")
    traces = [lagos, berlin]

    def fake_classify(old: str, new: str) -> ConflictVerdict:
        if "Lagos" in old:
            return ConflictVerdict(label=CORRECT, reason="Berlin replaced Lagos")
        return ConflictVerdict(label=CONFIRM, reason="same Berlin fact")

    transitions = []
    monkeypatch.setattr(sleep, "classify", fake_classify)
    monkeypatch.setattr(sleep, "append_transition", transitions.append)

    action = sleep._integrate_trace(duplicate_berlin, traces, now=NOW)

    assert action == "corrected"
    assert lagos.status == MemoryStatus.SUPERSEDED.value
    assert berlin.status == MemoryStatus.ACTIVE.value
    assert berlin.references == [duplicate_berlin.created_at]
    assert duplicate_berlin not in traces
    assert transitions[0].new_turn_id == berlin.turn_id
