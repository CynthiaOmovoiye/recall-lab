"""Variance runner: run the retail trial several times and report the spread.

A single trial is one draw. The agents answer above temperature zero, so the
exact wording, and what the sleep job promotes from it, changes between runs.
The salience judge and the contradiction classifier run at temperature zero.
One run is not a stable result.

This runner runs the full four-day both-agents trial N times, keeps every run
under reports/variance/<label>/run_K, then writes a summary with the per-run
accuracy, the min/max/mean spread, and the per-question pass count across runs.

The label keeps each campaign separate, so an earlier trial is never
overwritten by a rerun. Use a new label whenever the code under test changes.

The variance run scores each answer with 3 judge calls. That is an audit, not
the shipped default. The summary reports how often the 3 calls disagree. If
that rate is near zero, one judge call is enough and stays the default.

Run with:
    python -m recall_lab.eval.variance
    python -m recall_lab.eval.variance --runs 5 --label v3_ensemble_judge

Each run does roughly 250 API calls and takes a few minutes. Five runs is
about half an hour. Needs OPENROUTER_API_KEY in .env and the repo root as the
working directory.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any

from recall_lab.eval.equal_budget_trial import run_equal_budget_trial
from recall_lab.eval.multiday_trial import DEFAULT_SCENARIO, run_trial

VARIANCE_DIR = Path("reports/variance")


def _judge_audit_lines(payloads: list[dict[str, Any]]) -> list[str]:
    """Summarize how often the judge calls disagreed on the same answer.

    Each graded answer in a variance run is scored with 3 judge calls. A split
    verdict means the 3 calls did not agree. A near-zero split rate means one
    judge call is enough and the majority vote can be dropped.
    """
    graded = 0
    split = 0
    splits: list[str] = []
    for run_index, payload in enumerate(payloads, start=1):
        for agent in payload["agents"]:
            for record in agent["final_eval"]:
                votes = record.get("judge_votes") or []
                if len(votes) < 2:
                    continue
                graded += 1
                if len(set(votes)) > 1:
                    split += 1
                    splits.append(
                        f"- run {run_index}, {agent['agent_name']}: "
                        f"\"{record['user']}\" votes: {', '.join(votes)}"
                    )

    lines = ["## Judge audit", ""]
    if graded == 0:
        lines.append("Single-call scoring this campaign. No disagreement data.")
        return lines

    lines.append(
        f"The judge ran 3 calls per answer. {split} of {graded} graded answers "
        "got a split verdict."
    )
    if split == 0:
        lines += ["", "Every answer was unanimous. One judge call is enough."]
    else:
        lines += ["", "Split verdicts:", *splits]
    return lines


def write_summary(payloads: list[dict[str, Any]], path: Path) -> None:
    """Aggregate the runs into one markdown summary."""
    by_agent: dict[str, list[dict[str, Any]]] = {}
    for payload in payloads:
        for agent in payload["agents"]:
            by_agent.setdefault(agent["agent_name"], []).append(agent)

    scenario_name = payloads[0].get("scenario_name") or "unknown_scenario"
    lines = [
        f"# Recall Lab variance, {scenario_name}",
        "",
        f"{len(payloads)} runs of the both-agents trial.",
        "Same scenario, same code. The spread comes from the agents answering "
        "above temperature zero. The judge and classifier run at temperature zero.",
        "",
        "## Recall accuracy per run",
        "",
        "| Run | " + " | ".join(by_agent) + " |",
        "|" + "---|" * (len(by_agent) + 1),
    ]
    for i in range(len(payloads)):
        row = [f"run {i + 1}"]
        for runs in by_agent.values():
            row.append(f"{runs[i]['recall_accuracy']:.2f}")
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Spread", ""]
    for name, runs in by_agent.items():
        accs = [r["recall_accuracy"] for r in runs]
        mean = statistics.mean(accs)
        lines.append(
            f"- {name}: min {min(accs):.2f}, max {max(accs):.2f}, mean {mean:.2f}"
        )

    lines += ["", "## Per-question pass count across runs", ""]
    for name, runs in by_agent.items():
        lines += [f"### {name}", ""]
        counts: dict[str, list[int]] = {}
        for run in runs:
            for record in run["final_eval"]:
                question = record["user"]
                counts.setdefault(question, [0, 0])
                counts[question][1] += 1
                if record["correct"]:
                    counts[question][0] += 1
        for question, (passed, total) in counts.items():
            lines.append(f"- {passed}/{total}  {question}")
        lines.append("")

    lines += _judge_audit_lines(payloads)

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"wrote {path}")


def run_variance(
    runs: int,
    scenario_path: Path,
    label: str,
    mode: str = "multiday",
    agents: str = "both",
    budget_scale: float = 1.0,
) -> tuple[list[dict[str, Any]], Path]:
    """Run the trial `runs` times under reports/variance/<label>/run_K.

    Two modes share the same summary machinery, because both produce a payload
    with an `agents` list carrying `recall_accuracy` and `final_eval`:
    - multiday: the standard trial. `agents` selects the lineup (e.g. "all"
      adds the vector-retrieval control).
    - equal-budget: the budget-matched recency control vs Recall Lab.
    """
    base = VARIANCE_DIR / label
    base.mkdir(parents=True, exist_ok=True)
    payloads: list[dict[str, Any]] = []

    for n in range(1, runs + 1):
        out_dir = base / f"run_{n}"
        print(f"\n=== variance run {n}/{runs} ({mode}) -> {out_dir} ===", flush=True)
        try:
            if mode == "equal-budget":
                payload = run_equal_budget_trial(
                    scenario_path=scenario_path,
                    out_dir=out_dir,
                    budget_scale=budget_scale,
                    judge_samples=3,
                    verbose=True,
                )
            else:
                payload = run_trial(
                    scenario_path=scenario_path,
                    out_dir=out_dir,
                    agents=agents,
                    clean=True,
                    judge_samples=3,
                    verbose=True,
                )
            payloads.append(payload)
            for agent in payload["agents"]:
                print(
                    f"  {agent['agent_name']}: "
                    f"accuracy {agent['recall_accuracy']:.2f}",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001 - one bad run must not kill the batch
            print(f"  run {n} FAILED: {type(exc).__name__}: {exc}", flush=True)

    return payloads, base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the retail trial N times.")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument(
        "--label",
        default="v5_past_section",
        help="Campaign name. Output goes to reports/variance/<label>/.",
    )
    parser.add_argument(
        "--mode",
        choices=["multiday", "equal-budget"],
        default="multiday",
        help="'multiday' runs the standard trial; 'equal-budget' runs the "
        "budget-matched recency control vs Recall Lab.",
    )
    parser.add_argument(
        "--agents",
        choices=["sliding", "recall", "vector", "both", "all"],
        default="both",
        help="Multiday lineup. 'all' adds the vector-retrieval control. "
        "Ignored in equal-budget mode.",
    )
    parser.add_argument(
        "--budget-scale",
        type=float,
        default=1.0,
        help="Equal-budget multiplier on Recall Lab's mean input tokens.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads, base = run_variance(
        args.runs,
        args.scenario,
        args.label,
        mode=args.mode,
        agents=args.agents,
        budget_scale=args.budget_scale,
    )

    if not payloads:
        print("\nno runs completed, nothing to summarize")
        return

    summary_path = base / "variance_summary.md"
    write_summary(payloads, summary_path)
    print(f"\n{len(payloads)}/{args.runs} runs completed")
    print(f"trail: {base}/run_1 .. run_{args.runs}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
