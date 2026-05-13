"""Episodic log: every exchange, raw, on disk.

SQLite-backed. Append-only. Never injected into context after the day it
happened. Read only by the sleep job for consolidation, and by the eval harness
for ground truth.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from recall_lab.config import EPISODIC_DB_PATH


@dataclass
class Exchange:
    """One turn of conversation: user said X, agent said Y, at time T."""

    user: str
    agent: str
    timestamp: datetime
    salience: float | None = None
    promoted: bool = False
    id: int | None = None


SCHEMA = """
CREATE TABLE IF NOT EXISTS exchanges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user TEXT NOT NULL,
    agent TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    salience REAL,
    promoted INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_exchanges_timestamp ON exchanges(timestamp);
"""


class EpisodicLog:
    """SQLite-backed log of every conversation turn."""

    def __init__(self, db_path: Path = EPISODIC_DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(SCHEMA)

    def append(self, exchange: Exchange) -> int:
        """Append a new exchange to the log. Returns the assigned row id."""
        # TODO: insert into exchanges table, return lastrowid
        raise NotImplementedError

    def fetch_day(self, day: datetime) -> list[Exchange]:
        """Return all exchanges for the given calendar day (UTC)."""
        # TODO: query by date range
        raise NotImplementedError

    def mark_promoted(self, exchange_id: int, salience: float) -> None:
        """Record that an exchange was promoted to the brief, with its score."""
        # TODO: update row by id
        raise NotImplementedError
