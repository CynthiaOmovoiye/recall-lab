"""Configurable multi-day Recall Lab trial runner.

The scenario lives in JSON so the trial length, day labels, user messages, and
final questions can change without editing this logic.

Run with:
    python -m recall_lab.eval.multiday_trial

Use another scenario with:
    python -m recall_lab.eval.multiday_trial --scenario path/to/scenario.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from recall_lab.agent import RecallAgent
from recall_lab.consolidation.sleep import run_sleep_job
from recall_lab.controls.deterministic import DeterministicResolverAgent
from recall_lab.controls.episodic import EpisodicJudgeAgent
from recall_lab.config import STRONG_RAG_RECENCY_WINDOW_DAYS
from recall_lab.controls.sliding import SlidingWindowAgent
from recall_lab.controls.strong_rag import StrongRAGAgent
from recall_lab.controls.vector import VectorRetrievalAgent
from recall_lab.eval.metrics import (
    FailureMode,
    estimate_tokens,
    expects_abstain,
    judge_answer,
    score_abstain_answer,
)
from recall_lab.memory.brief import Brief
from recall_lab.memory.episodic import EpisodicLog
from recall_lab.memory.traces import MemoryTraceStore

DEFAULT_SCENARIO = Path("scenarios/retail_memory_week.json")
DEFAULT_OUT_DIR = Path("reports/retail_memory_week")
DEFAULT_VECTOR_TOP_K = 5
AgentChoice = Literal[
    "sliding", "recall", "vector", "strong_rag", "strong_rag_dated", "episodic",
    "deterministic", "both", "all",
]


@dataclass
class AnswerRecord:
    """One model response captured during the trial."""

    phase: str
    day_label: str
    turn_index: int
    user: str
    agent: str
    expected: str | None = None
    abstain_reason: str | None = None
    failure_mode: str | None = None
    correct: bool | None = None
    output_tokens_estimate: int = 0
    input_tokens_estimate: int = 0
    judge_votes: list[str] = field(default_factory=list)


@dataclass
class AgentTrialResult:
    """Full result for one agent."""

    agent_name: str
    records: list[AnswerRecord] = field(default_factory=list)
    sleep_summaries: list[dict[str, Any]] = field(default_factory=list)
    brief_text: str | None = None
    memory_traces: list[dict[str, Any]] = field(default_factory=list)

    @property
    def eval_records(self) -> list[AnswerRecord]:
        return [record for record in self.records if record.phase == "final_eval"]

    @property
    def recall_accuracy(self) -> float:
        eval_records = self.eval_records
        if not eval_records:
            return 0.0
        correct = sum(1 for record in eval_records if record.correct)
        return correct / len(eval_records)

    @property
    def answered_records(self) -> list[AnswerRecord]:
        """Final-eval questions the agent actually committed an answer to.

        A refusal is not an answer. That is the whole point of the abstain
        axis: an agent can be right by holding, and it can be useless by
        holding on everything, and one accuracy number cannot tell those apart.
        """
        return [
            record
            for record in self.eval_records
            if record.failure_mode != FailureMode.HONEST_GAP.value
        ]

    @property
    def selective_accuracy(self) -> float:
        """Correct answers over all questions.

        The name comes from arxiv 2605.30087, which scores a memory agent on
        two numbers rather than one. On this harness it is arithmetically the
        same as `recall_accuracy`, because every final-eval question is graded
        and a correct refusal already counts as correct. It is reported under
        its own name so the pair reads the way the paper's pair reads, and so
        the meaning survives if the question set ever stops being fully graded.
        `coverage` is the number that carries the new information.
        """
        return self.recall_accuracy

    @property
    def coverage(self) -> float:
        """Questions answered over all questions.

        Recall accuracy alone hides an agent that guesses on everything.
        Coverage is what exposes it, and it is also what exposes the opposite
        failure: an agent that refuses everything scores a safe-looking
        accuracy on a scenario with refusal questions in it while being no use
        at all.
        """
        eval_records = self.eval_records
        if not eval_records:
            return 0.0
        return len(self.answered_records) / len(eval_records)

    @property
    def refusals_by_reason(self) -> dict[str, dict[str, int]]:
        """Refusals caught per abstain reason, tagged from the scenario.

        A question carrying an optional `abstain_reason` (conflict, absence,
        revoked) is counted here as caught or missed. The reasons are scored
        apart because they fail apart: knowing that two facts contest each
        other is not the same skill as knowing a fact was never stated, which
        is not the same as honouring a withdrawal. An agent can pass one and
        confabulate on the next.
        """
        buckets: dict[str, dict[str, int]] = {}
        for record in self.eval_records:
            if not record.abstain_reason:
                continue
            bucket = buckets.setdefault(record.abstain_reason, {"refused": 0, "total": 0})
            bucket["total"] += 1
            if record.failure_mode == FailureMode.HONEST_GAP.value:
                bucket["refused"] += 1
        return buckets

    @property
    def output_tokens_estimate(self) -> int:
        return sum(record.output_tokens_estimate for record in self.records)

    @property
    def input_tokens_estimate(self) -> int:
        return sum(record.input_tokens_estimate for record in self.records)

    @property
    def chat_records(self) -> list[AnswerRecord]:
        return [record for record in self.records if record.phase == "chat"]

    @property
    def mean_chat_input_tokens(self) -> float:
        """Mean input-token cost per chat turn. Drives the equal-budget control."""
        chat = self.chat_records
        if not chat:
            return 0.0
        return sum(record.input_tokens_estimate for record in chat) / len(chat)


def load_scenario(path: Path) -> dict[str, Any]:
    """Load the JSON scenario file."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def selected_days(scenario: dict[str, Any], max_days: int | None) -> list[dict[str, Any]]:
    """Return the configured days, optionally truncated for smoke tests."""
    days = list(scenario.get("days", []))
    if max_days is not None:
        return days[:max_days]
    return days


def make_output_dir(base_dir: Path, clean: bool) -> Path:
    """Prepare a fresh output directory for the trial."""
    if clean and base_dir.exists():
        shutil.rmtree(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def parse_time(value: str | None, fallback: datetime) -> datetime:
    """Parse an ISO timestamp from a scenario file."""
    if not value:
        return fallback
    return datetime.fromisoformat(value)


class TrialClock:
    """Mutable clock used to give simulated days real timestamps."""

    def __init__(self) -> None:
        self.current = datetime.now(UTC)

    def set(self, value: datetime) -> None:
        self.current = value

    def __call__(self) -> datetime:
        return self.current


def score_answer(
    question: str, response: str, expected: str, judge_samples: int
) -> tuple[bool, FailureMode, list[str]]:
    """Score one final-eval answer.

    Returns whether it is correct, the failure mode, and the raw judge votes.
    With judge_samples above one the votes show whether the judge agreed with
    itself. That is the audit behind the single-call default.

    Two axes run through here. A question with a fact for `expected` is scored
    by the LLM judge: did the answer name that fact? A question with the
    literal `HONEST_GAP` for `expected` is scored on the abstain axis instead:
    a refusal is correct and a confident answer is a hallucination. That branch
    is deterministic and spends no judge call, so adding refusal questions to a
    scenario does not add to the judge bill, which is 92% of a campaign's cost.

    Every agent trial in this module routes its final-eval answers through this
    function, so both axes cover the whole lineup.
    """
    if expects_abstain(expected):
        result = score_abstain_answer(response)
        correct = result.mode is FailureMode.HONEST_GAP
    else:
        result = judge_answer(question, response, expected, samples=judge_samples)
        correct = result.mode is FailureMode.CORRECT

    votes = [vote.value for vote in result.votes]
    return correct, result.mode, votes


def run_sliding_trial(
    scenario: dict[str, Any],
    days: list[dict[str, Any]],
    judge_samples: int = 1,
    verbose: bool = False,
) -> AgentTrialResult:
    """Run the full scenario through the sliding-window baseline."""
    window = int(scenario.get("working_window", 2))
    agent = SlidingWindowAgent(window=window)
    result = AgentTrialResult(agent_name=f"sliding_window_{window}")

    for day in days:
        label = day.get("label", "day")
        turns = day.get("turns", [])
        for i, user_turn in enumerate(turns, start=1):
            if verbose:
                print(f"[sliding] {label}: turn {i}/{len(turns)}")
            response = agent.respond(user_turn)
            result.records.append(
                AnswerRecord(
                    phase="chat",
                    day_label=label,
                    turn_index=i,
                    user=user_turn,
                    agent=response,
                    output_tokens_estimate=estimate_tokens(response),
                    input_tokens_estimate=agent.last_input_tokens,
                )
            )

    final_eval = scenario.get("final_eval", {})
    eval_label = final_eval.get("label", "final_eval")
    questions = final_eval.get("questions", [])
    for i, item in enumerate(questions, start=1):
        question = item["text"]
        expected = item["expected"]
        if verbose:
            print(f"[sliding] {eval_label}: question {i}/{len(questions)}")
        response = agent.respond(question)
        correct, mode, votes = score_answer(question, response, expected, judge_samples)
        result.records.append(
            AnswerRecord(
                phase="final_eval",
                day_label=eval_label,
                turn_index=i,
                user=question,
                agent=response,
                expected=expected,
                abstain_reason=item.get("abstain_reason"),
                failure_mode=mode.value,
                correct=correct,
                judge_votes=votes,
                output_tokens_estimate=estimate_tokens(response),
                input_tokens_estimate=agent.last_input_tokens,
            )
        )

    return result


def run_vector_trial(
    scenario: dict[str, Any],
    days: list[dict[str, Any]],
    judge_samples: int = 1,
    verbose: bool = False,
    top_k: int = DEFAULT_VECTOR_TOP_K,
) -> AgentTrialResult:
    """Run the full scenario through the flat vector-retrieval control.

    This is the standard-RAG baseline: every exchange is embedded and the
    top-k most similar are retrieved each turn. It has no validity state, so a
    superseded fact and its correction compete on similarity alone.
    """
    agent = VectorRetrievalAgent(top_k=top_k)
    result = AgentTrialResult(agent_name=f"vector_topk_{top_k}")

    for day in days:
        label = day.get("label", "day")
        turns = day.get("turns", [])
        for i, user_turn in enumerate(turns, start=1):
            if verbose:
                print(f"[vector] {label}: turn {i}/{len(turns)}")
            response = agent.respond(user_turn)
            result.records.append(
                AnswerRecord(
                    phase="chat",
                    day_label=label,
                    turn_index=i,
                    user=user_turn,
                    agent=response,
                    output_tokens_estimate=estimate_tokens(response),
                    input_tokens_estimate=agent.last_input_tokens,
                )
            )

    final_eval = scenario.get("final_eval", {})
    eval_label = final_eval.get("label", "final_eval")
    questions = final_eval.get("questions", [])
    for i, item in enumerate(questions, start=1):
        question = item["text"]
        expected = item["expected"]
        if verbose:
            print(f"[vector] {eval_label}: question {i}/{len(questions)}")
        response = agent.respond(question)
        correct, mode, votes = score_answer(question, response, expected, judge_samples)
        result.records.append(
            AnswerRecord(
                phase="final_eval",
                day_label=eval_label,
                turn_index=i,
                user=question,
                agent=response,
                expected=expected,
                abstain_reason=item.get("abstain_reason"),
                failure_mode=mode.value,
                correct=correct,
                judge_votes=votes,
                output_tokens_estimate=estimate_tokens(response),
                input_tokens_estimate=agent.last_input_tokens,
            )
        )

    return result


def _parse_ts(value: Any) -> float | None:
    """Parse an ISO date string from the scenario into epoch seconds, or None."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except ValueError:
        return None


def run_strong_rag_trial(
    scenario: dict[str, Any],
    days: list[dict[str, Any]],
    judge_samples: int = 1,
    verbose: bool = False,
    top_k: int = DEFAULT_VECTOR_TOP_K,
    candidate_k: int = 12,
    recency_weight: float = 0.30,
    recency_mode: str = "turn",
    recency_window_days: float | None = None,
    show_timestamps: bool = False,
    chronological: bool = False,
    agent_name: str = "strong_rag",
) -> AgentTrialResult:
    """Run the full scenario through an industry-standard RAG control.

    Query rewriting, hybrid dense+BM25 retrieval fused with RRF, a recency boost,
    and reranking, the stack a production team actually ships. It still has no
    validity state, so this measures how far engineered retrieval gets on the
    relocation chain before it hits the authority gap.

    Two configurations share this runner. The default (`strong_rag`) uses
    turn-order recency. The dated variant (`strong_rag_dated`) uses real date
    metadata: timestamp recency plus a date-window filter, with the clock set per
    day from the scenario dates. Comparing them shows whether metadata-by-date
    filtering closes the chain gap or just relocates it.
    """
    agent = StrongRAGAgent(
        top_k=top_k,
        candidate_k=candidate_k,
        recency_weight=recency_weight,
        recency_mode=recency_mode,
        recency_window_days=recency_window_days,
        show_timestamps=show_timestamps,
        chronological=chronological,
    )
    result = AgentTrialResult(agent_name=agent_name)

    for day in days:
        label = day.get("label", "day")
        turns = day.get("turns", [])
        agent.set_clock(_parse_ts(day.get("date")))
        for i, user_turn in enumerate(turns, start=1):
            if verbose:
                print(f"[{agent_name}] {label}: turn {i}/{len(turns)}")
            response = agent.respond(user_turn)
            result.records.append(
                AnswerRecord(
                    phase="chat",
                    day_label=label,
                    turn_index=i,
                    user=user_turn,
                    agent=response,
                    output_tokens_estimate=estimate_tokens(response),
                    input_tokens_estimate=agent.last_input_tokens,
                )
            )

    final_eval = scenario.get("final_eval", {})
    eval_label = final_eval.get("label", "final_eval")
    questions = final_eval.get("questions", [])
    # The eval happens after the last day; stamp it at the final-eval date if the
    # scenario gives one, else the last day's date, so "now" is the present.
    eval_ts = _parse_ts(final_eval.get("date")) or (_parse_ts(days[-1].get("date")) if days else None)
    agent.set_clock(eval_ts)
    for i, item in enumerate(questions, start=1):
        question = item["text"]
        expected = item["expected"]
        if verbose:
            print(f"[{agent_name}] {eval_label}: question {i}/{len(questions)}")
        response = agent.respond(question)
        correct, mode, votes = score_answer(question, response, expected, judge_samples)
        result.records.append(
            AnswerRecord(
                phase="final_eval",
                day_label=eval_label,
                turn_index=i,
                user=question,
                agent=response,
                expected=expected,
                abstain_reason=item.get("abstain_reason"),
                failure_mode=mode.value,
                correct=correct,
                judge_votes=votes,
                output_tokens_estimate=estimate_tokens(response),
                input_tokens_estimate=agent.last_input_tokens,
            )
        )

    return result


def run_episodic_trial(
    scenario: dict[str, Any],
    days: list[dict[str, Any]],
    judge_samples: int = 1,
    verbose: bool = False,
) -> AgentTrialResult:
    """Run the full scenario through the raw episodic read-time-judge control.

    Keeps every statement raw and injects the whole log each turn, then asks the
    model to work out the current answer. No consolidation runs. This is the
    "just keep everything" baseline from arxiv 2605.12978; the comparison of
    interest is accuracy at a growing input-token cost versus the brief's flat
    one.
    """
    agent = EpisodicJudgeAgent()
    result = AgentTrialResult(agent_name="episodic_judge")

    for day in days:
        label = day.get("label", "day")
        turns = day.get("turns", [])
        for i, user_turn in enumerate(turns, start=1):
            if verbose:
                print(f"[episodic] {label}: turn {i}/{len(turns)}")
            response = agent.respond(user_turn)
            result.records.append(
                AnswerRecord(
                    phase="chat",
                    day_label=label,
                    turn_index=i,
                    user=user_turn,
                    agent=response,
                    output_tokens_estimate=estimate_tokens(response),
                    input_tokens_estimate=agent.last_input_tokens,
                )
            )

    final_eval = scenario.get("final_eval", {})
    eval_label = final_eval.get("label", "final_eval")
    questions = final_eval.get("questions", [])
    for i, item in enumerate(questions, start=1):
        question = item["text"]
        expected = item["expected"]
        if verbose:
            print(f"[episodic] {eval_label}: question {i}/{len(questions)}")
        response = agent.respond(question)
        correct, mode, votes = score_answer(question, response, expected, judge_samples)
        result.records.append(
            AnswerRecord(
                phase="final_eval",
                day_label=eval_label,
                turn_index=i,
                user=question,
                agent=response,
                expected=expected,
                abstain_reason=item.get("abstain_reason"),
                failure_mode=mode.value,
                correct=correct,
                judge_votes=votes,
                output_tokens_estimate=estimate_tokens(response),
                input_tokens_estimate=agent.last_input_tokens,
            )
        )

    return result


def run_deterministic_trial(
    scenario: dict[str, Any],
    days: list[dict[str, Any]],
    judge_samples: int = 1,
    verbose: bool = False,
) -> AgentTrialResult:
    """Run the scenario through the deterministic latest-value resolver control.

    Extracts (attribute, value) pairs from each user turn, stamps each with the
    scenario date, and resolves the current value by max timestamp in Python.
    No validity decision is made. This is the "why not just take the latest?"
    baseline: if it matches Recall Lab on the relocation chain, validity state is
    over-engineered for this scenario; where it breaks isolates what a validity
    decision buys over a timestamp sort.
    """
    agent = DeterministicResolverAgent()
    result = AgentTrialResult(agent_name="deterministic")

    for day in days:
        label = day.get("label", "day")
        turns = day.get("turns", [])
        agent.set_clock(_parse_ts(day.get("date")))
        for i, user_turn in enumerate(turns, start=1):
            if verbose:
                print(f"[deterministic] {label}: turn {i}/{len(turns)}")
            response = agent.respond(user_turn)
            result.records.append(
                AnswerRecord(
                    phase="chat",
                    day_label=label,
                    turn_index=i,
                    user=user_turn,
                    agent=response,
                    output_tokens_estimate=estimate_tokens(response),
                    input_tokens_estimate=agent.last_input_tokens,
                )
            )

    final_eval = scenario.get("final_eval", {})
    eval_label = final_eval.get("label", "final_eval")
    questions = final_eval.get("questions", [])
    eval_ts = _parse_ts(final_eval.get("date")) or (_parse_ts(days[-1].get("date")) if days else None)
    agent.set_clock(eval_ts)
    for i, item in enumerate(questions, start=1):
        question = item["text"]
        expected = item["expected"]
        if verbose:
            print(f"[deterministic] {eval_label}: question {i}/{len(questions)}")
        response = agent.respond(question)
        correct, mode, votes = score_answer(question, response, expected, judge_samples)
        result.records.append(
            AnswerRecord(
                phase="final_eval",
                day_label=eval_label,
                turn_index=i,
                user=question,
                agent=response,
                expected=expected,
                abstain_reason=item.get("abstain_reason"),
                failure_mode=mode.value,
                correct=correct,
                judge_votes=votes,
                output_tokens_estimate=estimate_tokens(response),
                input_tokens_estimate=agent.last_input_tokens,
            )
        )

    return result


def run_recall_trial(
    scenario: dict[str, Any],
    days: list[dict[str, Any]],
    out_dir: Path,
    judge_samples: int = 1,
    verbose: bool = False,
) -> AgentTrialResult:
    """Run the full scenario through the brief-backed Recall agent."""
    window = int(scenario.get("working_window", 2))
    result = AgentTrialResult(agent_name=f"recall_lab_brief_window_{window}")

    agent_dir = out_dir / "recall_agent_state"
    agent_dir.mkdir(parents=True, exist_ok=True)
    log = EpisodicLog(db_path=agent_dir / "log.db")
    brief = Brief(path=agent_dir / "brief.md")
    trace_store = MemoryTraceStore(path=agent_dir / "memory_traces.jsonl")
    brief.load()
    brief.save()
    clock = TrialClock()
    agent = RecallAgent(brief=brief, log=log, working_window=window, clock=clock)
    fallback_time = datetime.now(UTC)

    for day_index, day in enumerate(days):
        label = day.get("label", "day")
        day_time = parse_time(day.get("date"), fallback_time + timedelta(days=day_index))
        turns = day.get("turns", [])
        for i, user_turn in enumerate(turns, start=1):
            if verbose:
                print(f"[recall] {label}: turn {i}/{len(turns)}")
            clock.set(day_time + timedelta(minutes=i))
            response = agent.respond(user_turn)
            result.records.append(
                AnswerRecord(
                    phase="chat",
                    day_label=label,
                    turn_index=i,
                    user=user_turn,
                    agent=response,
                    output_tokens_estimate=estimate_tokens(response),
                    input_tokens_estimate=agent.last_input_tokens,
                )
            )

        if day.get("sleep_after", True):
            if verbose:
                print(f"[recall] {label}: sleep job")
            sleep_time = day_time.replace(hour=23, minute=0, second=0, microsecond=0)
            summary = run_sleep_job(sleep_time, brief, log, trace_store=trace_store)
            summary["day_label"] = label
            result.sleep_summaries.append(summary)
            if verbose:
                print(f"[recall] {label}: sleep summary {summary}")

    final_eval = scenario.get("final_eval", {})
    eval_label = final_eval.get("label", "final_eval")
    eval_time = parse_time(final_eval.get("date"), fallback_time + timedelta(days=len(days)))
    questions = final_eval.get("questions", [])
    for i, item in enumerate(questions, start=1):
        question = item["text"]
        expected = item["expected"]
        if verbose:
            print(f"[recall] {eval_label}: question {i}/{len(questions)}")
        clock.set(eval_time + timedelta(minutes=i))
        response = agent.respond(question)
        correct, mode, votes = score_answer(question, response, expected, judge_samples)
        result.records.append(
            AnswerRecord(
                phase="final_eval",
                day_label=eval_label,
                turn_index=i,
                user=question,
                agent=response,
                expected=expected,
                abstain_reason=item.get("abstain_reason"),
                failure_mode=mode.value,
                correct=correct,
                judge_votes=votes,
                output_tokens_estimate=estimate_tokens(response),
                input_tokens_estimate=agent.last_input_tokens,
            )
        )

    result.brief_text = brief.path.read_text(encoding="utf-8")
    result.memory_traces = trace_store.to_dicts()
    return result


def result_payload(
    scenario: dict[str, Any],
    days: list[dict[str, Any]],
    results: list[AgentTrialResult],
) -> dict[str, Any]:
    """Create JSON-serializable output."""
    return {
        "scenario_name": scenario.get("name"),
        "description": scenario.get("description"),
        "working_window": scenario.get("working_window", 2),
        "days_run": [day.get("label") for day in days],
        "total_chat_exchanges": sum(len(day.get("turns", [])) for day in days),
        "final_eval_questions": len(scenario.get("final_eval", {}).get("questions", [])),
        "agents": [
            {
                "agent_name": result.agent_name,
                "recall_accuracy": result.recall_accuracy,
                "selective_accuracy": result.selective_accuracy,
                "coverage": result.coverage,
                "refusals_by_reason": result.refusals_by_reason,
                "output_tokens_estimate": result.output_tokens_estimate,
                "input_tokens_estimate": result.input_tokens_estimate,
                "mean_chat_input_tokens": round(result.mean_chat_input_tokens, 1),
                "final_eval": [asdict(record) for record in result.eval_records],
                "sleep_summaries": result.sleep_summaries,
                "brief_text": result.brief_text,
                "memory_traces": result.memory_traces,
            }
            for result in results
        ],
    }


def write_markdown_report(payload: dict[str, Any], path: Path) -> None:
    """Write a screenshot-friendly markdown report."""
    lines = [
        f"# Recall Lab trial: {payload['scenario_name']}",
        "",
        payload.get("description") or "",
        "",
        "## Setup",
        "",
        f"- Working window: {payload['working_window']} turns",
        f"- Days run: {', '.join(payload['days_run'])}",
        f"- Chat exchanges before final eval: {payload['total_chat_exchanges']}",
        f"- Final eval questions: {payload['final_eval_questions']}",
        "",
        "## Results",
        "",
        "| Agent | Recall accuracy | Mean input tokens/turn | Output token estimate | Coverage |",
        "|---|---:|---:|---:|---:|",
    ]

    for agent in payload["agents"]:
        lines.append(
            f"| {agent['agent_name']} | {agent['recall_accuracy']:.2f} | "
            f"{agent.get('mean_chat_input_tokens', 0)} | "
            f"{agent['output_tokens_estimate']} | "
            f"{agent.get('coverage', 1.0):.2f} |"
        )

    refusal_rows = [
        (agent["agent_name"], agent.get("refusals_by_reason") or {})
        for agent in payload["agents"]
    ]
    if any(reasons for _, reasons in refusal_rows):
        reason_names = sorted({r for _, reasons in refusal_rows for r in reasons})
        lines.extend(
            [
                "",
                "### Refusals caught, by reason",
                "",
                "A question tagged with an abstain reason is one the agent should "
                "decline. The reasons are counted apart because they fail apart.",
                "",
                "| Agent | " + " | ".join(reason_names) + " |",
                "|---|" + "---:|" * len(reason_names),
            ]
        )
        for name, reasons in refusal_rows:
            cells = []
            for reason in reason_names:
                bucket = reasons.get(reason)
                cells.append(
                    f"{bucket['refused']}/{bucket['total']}" if bucket else "-"
                )
            lines.append(f"| {name} | " + " | ".join(cells) + " |")

    lines.extend(["", "## Final evaluation", ""])
    for agent in payload["agents"]:
        lines.extend([f"### {agent['agent_name']}", ""])
        for record in agent["final_eval"]:
            status = "PASS" if record["correct"] else "FAIL"
            entry = [
                f"- {status}: {record['user']}",
                f"  - expected: {record['expected']}",
                f"  - mode: {record['failure_mode']}",
            ]
            if record.get("abstain_reason"):
                entry.append(f"  - abstain reason: {record['abstain_reason']}")
            votes = record.get("judge_votes") or []
            if len(votes) > 1:
                agree = "unanimous" if len(set(votes)) == 1 else "split"
                entry.append(f"  - judge votes ({agree}): {', '.join(votes)}")
            entry.append(f"  - answer: {record['agent']}")
            lines.extend(entry)
        lines.append("")

    for agent in payload["agents"]:
        if agent.get("memory_traces"):
            lines.extend(["## Final memory traces", ""])
            lines.append("| Status | Section | Memory | Supersedes | References |")
            lines.append("|---|---|---|---:|---:|")
            for trace in agent["memory_traces"]:
                lines.append(
                    f"| {trace['status']} | {trace['section']} | {trace['compression']} | "
                    f"{trace.get('supersedes') or ''} | {len(trace.get('references', []))} |"
                )
            lines.append("")

        if agent.get("brief_text"):
            lines.extend(
                ["## Final Recall Lab brief", "", "```markdown", agent["brief_text"], "```"]
            )

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_trial(
    scenario_path: Path = DEFAULT_SCENARIO,
    out_dir: Path = DEFAULT_OUT_DIR,
    agents: AgentChoice = "both",
    max_days: int | None = None,
    clean: bool = True,
    judge_samples: int = 1,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run a configurable trial and write JSON plus markdown outputs."""
    scenario = load_scenario(scenario_path)
    days = selected_days(scenario, max_days)
    out_dir = make_output_dir(out_dir, clean=clean)

    results: list[AgentTrialResult] = []
    if agents in {"sliding", "both", "all"}:
        results.append(
            run_sliding_trial(scenario, days, judge_samples=judge_samples, verbose=verbose)
        )
    if agents in {"vector", "all"}:
        results.append(
            run_vector_trial(scenario, days, judge_samples=judge_samples, verbose=verbose)
        )
    if agents in {"strong_rag", "all"}:
        results.append(
            run_strong_rag_trial(scenario, days, judge_samples=judge_samples, verbose=verbose)
        )
    if agents in {"strong_rag_dated", "all"}:
        results.append(
            run_strong_rag_trial(
                scenario,
                days,
                judge_samples=judge_samples,
                verbose=verbose,
                recency_mode="timestamp",
                recency_window_days=STRONG_RAG_RECENCY_WINDOW_DAYS,
                agent_name="strong_rag_dated",
            )
        )
    if agents in {"episodic", "all"}:
        results.append(
            run_episodic_trial(scenario, days, judge_samples=judge_samples, verbose=verbose)
        )
    if agents in {"deterministic", "all"}:
        results.append(
            run_deterministic_trial(scenario, days, judge_samples=judge_samples, verbose=verbose)
        )
    if agents in {"recall", "both", "all"}:
        results.append(
            run_recall_trial(
                scenario, days, out_dir, judge_samples=judge_samples, verbose=verbose
            )
        )

    payload = result_payload(scenario, days, results)
    (out_dir / "trial_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown_report(payload, out_dir / "trial_report.md")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a configurable multi-day Recall Lab trial.")
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--agents",
        choices=[
            "sliding", "recall", "vector", "strong_rag", "strong_rag_dated",
            "episodic", "deterministic", "both", "all",
        ],
        default="both",
        help="'both' is sliding+recall. 'all' adds the vector, strong-RAG, "
        "date-metadata strong-RAG, raw-episodic, and deterministic-resolver controls.",
    )
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument(
        "--judge-samples",
        type=int,
        default=1,
        help="Judge calls per answer. 1 ships. Use 3 to audit judge disagreement.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print progress while running.")
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not delete the output dir first.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_trial(
        scenario_path=args.scenario,
        out_dir=args.out_dir,
        agents=args.agents,
        max_days=args.max_days,
        clean=not args.no_clean,
        judge_samples=args.judge_samples,
        verbose=args.verbose,
    )

    print(f"trial: {payload['scenario_name']}")
    print(f"days: {', '.join(payload['days_run'])}")
    print()
    for agent in payload["agents"]:
        print(agent["agent_name"])
        print("  recall accuracy:", agent["recall_accuracy"])
        print("  coverage:", agent.get("coverage"))
        print("  refusals by reason:", agent.get("refusals_by_reason") or "-")
        print("  mean input tokens/turn:", agent.get("mean_chat_input_tokens", 0))
        print("  output tokens estimate:", agent["output_tokens_estimate"])
    print()
    print(f"wrote {args.out_dir / 'trial_result.json'}")
    print(f"wrote {args.out_dir / 'trial_report.md'}")


if __name__ == "__main__":
    main()
