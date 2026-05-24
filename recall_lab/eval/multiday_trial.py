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
from recall_lab.controls.sliding import SlidingWindowAgent
from recall_lab.eval.metrics import FailureMode, estimate_tokens, judge_failure_mode
from recall_lab.memory.brief import Brief
from recall_lab.memory.episodic import EpisodicLog
from recall_lab.memory.traces import MemoryTraceStore

DEFAULT_SCENARIO = Path("scenarios/retail_memory_week.json")
DEFAULT_OUT_DIR = Path("reports/retail_memory_week")
AgentChoice = Literal["sliding", "recall", "both"]


@dataclass
class AnswerRecord:
    """One model response captured during the trial."""

    phase: str
    day_label: str
    turn_index: int
    user: str
    agent: str
    expected: str | None = None
    failure_mode: str | None = None
    correct: bool | None = None
    output_tokens_estimate: int = 0


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
    def output_tokens_estimate(self) -> int:
        return sum(record.output_tokens_estimate for record in self.records)


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


def score_answer(question: str, response: str, expected: str) -> tuple[bool, FailureMode]:
    """Score one final-eval answer with the LLM judge."""
    mode = judge_failure_mode(question, response, expected)
    return mode == FailureMode.CORRECT, mode


def run_sliding_trial(
    scenario: dict[str, Any],
    days: list[dict[str, Any]],
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
        correct, mode = score_answer(question, response, expected)
        result.records.append(
            AnswerRecord(
                phase="final_eval",
                day_label=eval_label,
                turn_index=i,
                user=question,
                agent=response,
                expected=expected,
                failure_mode=mode.value,
                correct=correct,
                output_tokens_estimate=estimate_tokens(response),
            )
        )

    return result


def run_recall_trial(
    scenario: dict[str, Any],
    days: list[dict[str, Any]],
    out_dir: Path,
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
        correct, mode = score_answer(question, response, expected)
        result.records.append(
            AnswerRecord(
                phase="final_eval",
                day_label=eval_label,
                turn_index=i,
                user=question,
                agent=response,
                expected=expected,
                failure_mode=mode.value,
                correct=correct,
                output_tokens_estimate=estimate_tokens(response),
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
                "output_tokens_estimate": result.output_tokens_estimate,
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
        "| Agent | Recall accuracy | Output token estimate |",
        "|---|---:|---:|",
    ]

    for agent in payload["agents"]:
        lines.append(
            f"| {agent['agent_name']} | {agent['recall_accuracy']:.2f} | "
            f"{agent['output_tokens_estimate']} |"
        )

    lines.extend(["", "## Final evaluation", ""])
    for agent in payload["agents"]:
        lines.extend([f"### {agent['agent_name']}", ""])
        for record in agent["final_eval"]:
            status = "PASS" if record["correct"] else "FAIL"
            lines.extend(
                [
                    f"- {status}: {record['user']}",
                    f"  - expected: {record['expected']}",
                    f"  - mode: {record['failure_mode']}",
                    f"  - answer: {record['agent']}",
                ]
            )
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
    verbose: bool = False,
) -> dict[str, Any]:
    """Run a configurable trial and write JSON plus markdown outputs."""
    scenario = load_scenario(scenario_path)
    days = selected_days(scenario, max_days)
    out_dir = make_output_dir(out_dir, clean=clean)

    results: list[AgentTrialResult] = []
    if agents in {"sliding", "both"}:
        results.append(run_sliding_trial(scenario, days, verbose=verbose))
    if agents in {"recall", "both"}:
        results.append(run_recall_trial(scenario, days, out_dir, verbose=verbose))

    payload = result_payload(scenario, days, results)
    (out_dir / "trial_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown_report(payload, out_dir / "trial_report.md")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a configurable multi-day Recall Lab trial.")
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--agents", choices=["sliding", "recall", "both"], default="both")
    parser.add_argument("--max-days", type=int, default=None)
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
        verbose=args.verbose,
    )

    print(f"trial: {payload['scenario_name']}")
    print(f"days: {', '.join(payload['days_run'])}")
    print()
    for agent in payload["agents"]:
        print(agent["agent_name"])
        print("  recall accuracy:", agent["recall_accuracy"])
        print("  output tokens estimate:", agent["output_tokens_estimate"])
    print()
    print(f"wrote {args.out_dir / 'trial_result.json'}")
    print(f"wrote {args.out_dir / 'trial_report.md'}")


if __name__ == "__main__":
    main()
