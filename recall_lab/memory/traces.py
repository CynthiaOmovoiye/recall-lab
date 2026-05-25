"""Machine-readable memory trace store.

The brief is for the agent to read. The trace store is for the sleep job to
reason over: salience, contradiction, validity state, references, and activation.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from recall_lab.config import TRACE_STORE_PATH
from recall_lab.consolidation.activation import MemoryTrace, activation
from recall_lab.consolidation.contradiction import MemoryStatus, current_traces
from recall_lab.memory.brief import Brief, SECTION_ALIASES


class MemoryTraceStore:
    """JSONL-backed store of promoted semantic memories."""

    def __init__(self, path: Path = TRACE_STORE_PATH) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[MemoryTrace]:
        """Read all traces from disk."""
        if not self.path.exists():
            return []

        traces: list[MemoryTrace] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            traces.append(self._from_dict(data))
        return traces

    def save(self, traces: list[MemoryTrace]) -> None:
        """Write traces to disk as JSONL."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(self._to_dict(trace), sort_keys=True) for trace in traces]
        self.path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def active(self) -> list[MemoryTrace]:
        """Return traces allowed to count as current truth."""
        return current_traces(self.load())

    def render_brief(self, brief: Brief, now: datetime) -> None:
        """Render the trace store into the human-readable brief.

        Active traces become current facts under their own sections. Superseded
        traces become history under the past section, each line marked so the
        agent does not read it as current truth. Archived traces are dropped.
        """
        all_traces = self.load()
        sections: dict[str, list[str]] = {}

        active = sorted(
            current_traces(all_traces),
            key=lambda trace: activation(trace, now),
            reverse=True,
        )
        for trace in active:
            section = SECTION_ALIASES.get(trace.section, trace.section)
            sections.setdefault(section, [])
            if trace.compression not in sections[section]:
                sections[section].append(trace.compression)

        superseded = sorted(
            (t for t in all_traces if t.status == MemoryStatus.SUPERSEDED.value),
            key=lambda trace: trace.created_at,
        )
        past_section = SECTION_ALIASES["past"]
        total_past = len(superseded)
        for index, trace in enumerate(superseded, start=1):
            if total_past == 1:
                label = "Past"
            elif index == 1:
                label = "Earliest past"
            elif index == total_past:
                label = "Most recent past before current"
            else:
                label = f"Past step {index}"
            line = f"{label}: {trace.compression}"
            sections.setdefault(past_section, [])
            if line not in sections[past_section]:
                sections[past_section].append(line)

        brief.sections = sections
        brief.save()

    def to_dicts(self) -> list[dict]:
        """Return all traces as JSON-serializable dictionaries."""
        return [self._to_dict(trace) for trace in self.load()]

    @staticmethod
    def _to_dict(trace: MemoryTrace) -> dict:
        data = asdict(trace)
        data["created_at"] = trace.created_at.isoformat()
        data["references"] = [ref.isoformat() for ref in trace.references]
        return data

    @staticmethod
    def _from_dict(data: dict) -> MemoryTrace:
        return MemoryTrace(
            turn_id=int(data["turn_id"]),
            compression=str(data["compression"]),
            base_salience=float(data["base_salience"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            references=[datetime.fromisoformat(ref) for ref in data.get("references", [])],
            emotional_weight=float(data.get("emotional_weight", 0.0)),
            status=str(data.get("status", MemoryStatus.ACTIVE.value)),
            supersedes=data.get("supersedes"),
            section=str(data.get("section", "stable_facts")),
        )
