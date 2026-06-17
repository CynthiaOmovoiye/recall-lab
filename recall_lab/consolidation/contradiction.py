"""Correction detection and memory validity state.

activation.py answers one question. Which memory is most useful right now.
This module answers a different one. Which memory is allowed to count as
current truth. The two are kept apart on purpose.

A corrected fact can still have high activation. It keeps getting
re-referenced, so it keeps winning retrieval. Lowering its rank does not stop
that. The fix is to take it out of the current set.

Retrieval runs in two stages:
    1. filter by validity   (current_traces, in this file)
    2. rank the survivors   (activation, in activation.py)

A correction is a state transition. The old trace goes from active to
superseded. The new trace stores a supersedes pointer back to it. Every
transition is written to a log so it can be read back and undone. A wrongly
suppressed fact is worse than a noisy one, so suppression stays reversible.

The ranking-versus-validity split, the superseded state, the supersedes
pointer, and the reversible-suppression rule came out of a comment thread
with Piotr Skorek on 2026-05-23.

Run the offline checks with:
    python -m recall_lab.consolidation.contradiction
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

from recall_lab.config import (
    CONTRADICTION_MODEL,
    DATA_DIR,
    MAX_OUTPUT_TOKENS,
    OPENROUTER_API_KEY,
)
from recall_lab.consolidation.activation import MemoryTrace
from recall_lab.llm import chat_client, complete


# The audit trail. One JSON line per validity change.
TRANSITION_LOG_PATH = DATA_DIR / "transitions.jsonl"


class MemoryStatus(str, Enum):
    """The validity state of a memory trace.

    ACTIVE      eligible to count as current truth
    SUPERSEDED  replaced by a newer, contradicting fact, kept for audit
    ARCHIVED    retired by hand or by age, not current, not part of a chain
    """

    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"


# --------------------------------------------------------------------------
# Detection. classify() only reads. It changes no state.
# --------------------------------------------------------------------------

CONFIRM = "CONFIRM"
CORRECT = "CORRECT"
UNRELATED = "UNRELATED"
_LABELS = {CONFIRM, CORRECT, UNRELATED}


@dataclass
class ConflictVerdict:
    """How a new statement relates to an old remembered fact."""

    label: str  # CONFIRM | CORRECT | UNRELATED
    reason: str  # one short sentence, reused as the transition reason


CLASSIFY_PROMPT = """You compare a new statement against an old remembered fact.
Decide the relationship between them.

OLD FACT: {old}
NEW STATEMENT: {new}

Pick exactly one label:
- CONFIRM: the new statement agrees with the old fact or repeats it.
- CORRECT: the new statement updates or contradicts the old fact. The old fact is now stale.
- UNRELATED: they are about different things.

The conflict can be implicit. The new statement does not have to say the old
one is wrong. If the new statement makes the old fact no longer true, that is
CORRECT.

But a new statement only CORRECTs an old fact if it actually sets a new current
value. A statement that merely reminisces about, or expresses a feeling toward,
a value is not a correction. "Green is still such a beautiful color, I always
come back to it" does not change a current favorite of blue; that is UNRELATED,
not CORRECT. Sentiment about a value is not a change to it.

Return JSON only:
{{"label": "CONFIRM|CORRECT|UNRELATED", "reason": "one short sentence"}}
"""


def _extract_verdict_json(text: str) -> dict:
    """Parse a JSON object from plain or fenced model output.

    The strong model can wrap JSON in a markdown fence or add a stray line.
    This strips a fence and, failing that, slices out the first {...} block.
    Raises ValueError if the result is not a JSON object.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(stripped[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("Verdict response must be a JSON object.")
    return parsed


def classify(old: str, new: str, model: str = CONTRADICTION_MODEL) -> ConflictVerdict:
    """Classify how a new statement relates to an old remembered fact.

    Calls the model via OpenRouter. This is the detector and nothing else. It
    does not touch any trace. The sleep job decides what to do with the verdict.

    Runs at temperature zero on the strong model. Spotting an implicit
    correction is the hardest reasoning call in the system, so it uses the same
    model class as the judge, not the cheap agent model.

    On any parse failure it returns UNRELATED. An unreadable verdict must never
    be allowed to suppress a fact, so the failure mode is the safe one.

    Args:
        old: the existing remembered fact, as a compressed statement.
        new: the incoming statement to compare against it.
        model: OpenRouter model id. Defaults to CONTRADICTION_MODEL.

    Returns:
        A ConflictVerdict with a label and a one-line reason.
    """
    prompt = CLASSIFY_PROMPT.format(old=old, new=new)
    client = chat_client()
    completion = complete(
        client,
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Return only valid JSON. Do not include markdown or commentary.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    raw = completion.choices[0].message.content or ""

    try:
        data = _extract_verdict_json(raw)
    except (json.JSONDecodeError, ValueError):
        return ConflictVerdict(label=UNRELATED, reason=f"unparseable verdict: {raw!r}")

    label = str(data.get("label", "")).strip().upper()
    if label not in _LABELS:
        return ConflictVerdict(label=UNRELATED, reason=f"unknown label in: {raw!r}")
    return ConflictVerdict(label=label, reason=str(data.get("reason", "")).strip())


# --------------------------------------------------------------------------
# Validity transitions. These mutate traces and must be logged.
# --------------------------------------------------------------------------


@dataclass
class Transition:
    """An audit record for one validity state change.

    Holds everything needed to read the change back and to undo it. Written
    to the transition log on every state change.
    """

    old_turn_id: int
    new_turn_id: int | None  # the replacing trace, None for a plain archive
    from_status: str
    to_status: str
    reason: str  # why: the classifier's reason
    trigger: str  # what caused it, e.g. "sleep_job" or "manual"
    at: str  # ISO timestamp


def supersede(
    old: MemoryTrace,
    new: MemoryTrace,
    verdict: ConflictVerdict,
    trigger: str = "sleep_job",
    now: datetime | None = None,
) -> Transition:
    """Apply a correction. Move `old` to superseded and point `new` at it.

    Mutates both traces in place. Returns a Transition for the log.

    Raises ValueError if the verdict is not CORRECT. A confirmation must never
    be able to suppress a fact.

    The supersedes pointer chains. `new.supersedes` is set to `old.turn_id`,
    the version directly before it. If `old` already superseded something, that
    older link stays on `old.supersedes`. Walking the chain back gives the full
    history. Walking forward to the one ACTIVE trace gives current truth. This
    answers the open question from 2026-05-23: chain every version, do not
    collapse to the latest only, because a collapse loses the middle history
    and makes a step-by-step revert impossible.
    """
    if verdict.label != CORRECT:
        raise ValueError(f"supersede() needs a CORRECT verdict, got {verdict.label}")

    now = now or datetime.utcnow()
    transition = Transition(
        old_turn_id=old.turn_id,
        new_turn_id=new.turn_id,
        from_status=old.status,
        to_status=MemoryStatus.SUPERSEDED.value,
        reason=verdict.reason,
        trigger=trigger,
        at=now.isoformat(),
    )

    old.status = MemoryStatus.SUPERSEDED.value
    new.supersedes = old.turn_id
    return transition


def revert(transition: Transition, traces_by_id: dict[int, MemoryTrace]) -> None:
    """Undo a supersede. Suppression has to be reversible.

    revert is the exact inverse of one supersede call. It undoes the two
    writes supersede made: it flips the old trace back to the status it had
    before, and it clears the new trace's supersedes pointer.

    It does one step only. It does not cascade down a chain, and it does not
    touch the new trace's own status. After reverting a wrong suppression both
    traces will read as active. That is the right outcome. A wrong CORRECT
    verdict means the two facts were never in conflict, so both should count
    as current truth.

    Args:
        transition: the Transition record to undo.
        traces_by_id: maps turn_id to MemoryTrace for the traces in play.
    """
    old = traces_by_id.get(transition.old_turn_id)
    if old is not None:
        old.status = transition.from_status

    if transition.new_turn_id is not None:
        new = traces_by_id.get(transition.new_turn_id)
        if new is not None and new.supersedes == transition.old_turn_id:
            new.supersedes = None


def append_transition(
    transition: Transition, path: Path = TRANSITION_LOG_PATH
) -> None:
    """Append a transition to the audit log as one JSON line.

    Every validity change goes here. The log is what makes a suppression
    something you can inspect and undo.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(transition)) + "\n")


# --------------------------------------------------------------------------
# Retrieval filter. Run this before ranking with activation().
# --------------------------------------------------------------------------


def current_traces(traces: list[MemoryTrace]) -> list[MemoryTrace]:
    """Keep only the traces allowed to count as current truth.

    This is the validity filter. Run it first, then rank what survives with
    activation(). Superseded and archived traces stay on disk for audit and
    historical queries. They just stop competing as current truth.
    """
    return [t for t in traces if t.status == MemoryStatus.ACTIVE.value]


# --------------------------------------------------------------------------
# Offline checks. No network unless an API key is set.
# --------------------------------------------------------------------------


def _demo() -> None:
    """Exercise the validity logic offline, then classify() if a key is set."""
    now = datetime(2026, 5, 23, 12, 0, 0)

    # A chain of three versions of the same fact: Lagos, then Berlin, then Nairobi.
    lagos = MemoryTrace(
        turn_id=1,
        compression="User lives in Lagos",
        base_salience=0.8,
        created_at=now - timedelta(days=40),
    )
    berlin = MemoryTrace(
        turn_id=2,
        compression="User moved to Berlin in April 2026",
        base_salience=0.8,
        created_at=now - timedelta(days=20),
    )
    nairobi = MemoryTrace(
        turn_id=3,
        compression="User relocated to Nairobi this month",
        base_salience=0.8,
        created_at=now - timedelta(days=2),
    )
    traces = [lagos, berlin, nairobi]

    correct = ConflictVerdict(label=CORRECT, reason="the user's city changed")

    # Two corrections, applied in order.
    t1 = supersede(lagos, berlin, correct, trigger="demo", now=now)
    t2 = supersede(berlin, nairobi, correct, trigger="demo", now=now)

    print("after two corrections:")
    for t in traces:
        print(
            f"  turn {t.turn_id}: {t.compression!r} "
            f"status={t.status} supersedes={t.supersedes}"
        )

    current = current_traces(traces)
    assert [t.turn_id for t in current] == [3], current
    assert nairobi.supersedes == 2
    assert berlin.supersedes == 1
    assert lagos.supersedes is None
    print("chain ok: 3 -> 2 -> 1, current = [3]")

    # Reversible suppression. revert is the exact inverse of one supersede.
    # It restores the old trace and unlinks the new one. It does not cascade.
    by_id = {t.turn_id: t for t in traces}
    revert(t2, by_id)
    assert berlin.status == MemoryStatus.ACTIVE.value, berlin
    assert nairobi.supersedes is None, nairobi
    assert lagos.status == MemoryStatus.SUPERSEDED.value, lagos  # t1 left alone
    print("revert ok: berlin recovered to active, nairobi unlinked, t1 untouched")

    # A CONFIRM verdict must never be able to suppress.
    try:
        supersede(lagos, berlin, ConflictVerdict(label=CONFIRM, reason="same"))
    except ValueError as exc:
        print(f"guard ok: {exc}")

    # The audit log gets one line per transition.
    append_transition(t1)
    append_transition(t2)
    print(f"logged 2 transitions to {TRANSITION_LOG_PATH}")

    if not OPENROUTER_API_KEY:
        print("\nno OPENROUTER_API_KEY set, skipping the live classify() check")
        return

    pairs = [
        ("User lives in Lagos", "We moved to Berlin in April", CORRECT),
        ("User lives in Lagos", "Still based in Lagos, same flat", CONFIRM),
        ("User lives in Lagos", "User prefers tab indentation", UNRELATED),
    ]
    print("\nlive classify():")
    for old, new, expected in pairs:
        verdict = classify(old, new)
        mark = "ok" if verdict.label == expected else "MISMATCH"
        print(f"  [{mark}] expected {expected}, got {verdict.label}: {verdict.reason}")


if __name__ == "__main__":
    _demo()
