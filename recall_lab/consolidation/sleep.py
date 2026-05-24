"""The sleep job: nightly consolidation.

Steps:
1. Pull the day's episodic exchanges.
2. Score each with the judge.
3. Promote high-salience exchanges to the brief as compressed semantic statements.
4. Mark low-salience exchanges as not-promoted (they stay on disk).
5. Optionally apply decay to existing brief entries (Forgetting Curves Lab).
"""

from __future__ import annotations

from datetime import datetime

from recall_lab.config import SALIENCE_THRESHOLD
from recall_lab.consolidation.judge import score_exchange
from recall_lab.memory.brief import Brief
from recall_lab.memory.episodic import EpisodicLog


def run_sleep_job(day: datetime, brief: Brief, log: EpisodicLog) -> dict:
    """Run one consolidation pass over a single day's exchanges.

    Returns a summary dict suitable for appending to research-log.md.
    """
    exchanges = log.fetch_day(day)
    scored = 0
    promoted = 0
    for ex in exchanges:
        if ex.promoted:
            continue

        scored += 1
        verdict = score_exchange(ex)
        if (
            verdict.score >= SALIENCE_THRESHOLD
            and verdict.suggested_statement
            and verdict.suggested_brief_section
        ):
            brief.add_entry(
                section=verdict.suggested_brief_section,
                text=verdict.suggested_statement,
            )
            if ex.id is not None:
                log.mark_promoted(ex.id, verdict.score)
            promoted += 1
    brief.save()
    return {
        "day": day.isoformat(),
        "exchanges": len(exchanges),
        "scored": scored,
        "promoted": promoted,
        "threshold": SALIENCE_THRESHOLD,
    }
