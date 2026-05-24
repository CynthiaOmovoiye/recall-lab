"""ACT-R style activation function for memory traces.

A memory's activation is its current retrievability. Higher activation means
the memory is more likely to win retrieval against alternatives.

Activation is computed from:
- recency: time since creation
- frequency: time since each re-reference
- base salience: the 0-1 score from the judge
- emotional weight: 0 unless the user signal was strong
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MemoryTrace:
    """A single memory entry tracked by the activation function.

    `status` and `supersedes` are validity fields. They are written and read
    by consolidation/contradiction.py. The activation() function below ignores
    them on purpose. Rank and validity are separate decisions.
    """

    turn_id: int
    compression: str
    base_salience: float
    created_at: datetime
    references: list[datetime] = field(default_factory=list)
    emotional_weight: float = 0.0
    status: str = "active"  # active | superseded | archived
    supersedes: int | None = None  # turn_id of the version this one replaced
    section: str = "stable_facts"


def activation(trace: MemoryTrace, now: datetime, decay: float = 0.5) -> float:
    """Compute activation. Higher means more retrievable.

    Implements the ACT-R log-sum-of-power-decay form:
        A = ln(sum_i t_i^-d) + base_salience + emotional_weight

    where t_i is the age in hours of the creation event and each re-reference.
    """
    times = [trace.created_at] + trace.references
    age_terms = [
        max((now - t).total_seconds() / 3600.0, 1e-3) ** -decay
        for t in times
    ]
    base = math.log(sum(age_terms))
    return base + trace.base_salience + trace.emotional_weight


def should_keep(trace: MemoryTrace, now: datetime, threshold: float = -1.0) -> bool:
    """Return True if the trace is still retrievable at the given threshold."""
    return activation(trace, now) >= threshold