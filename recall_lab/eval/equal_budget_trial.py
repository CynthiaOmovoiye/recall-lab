"""Equal-token-budget trial.

The protocol calls for a control that removes one objection to every Recall Lab
result so far: maybe the brief-backed agent only wins because the sliding-window
baseline was deliberately starved (a two-turn window), so it simply had fewer
tokens to get confused by. If that were the whole story, handing the baseline a
larger, equal context budget would close the gap.

This runner tests that directly:

1. Run the Recall agent on the scenario and measure its mean input-token cost
   per chat turn. That is the budget Recall Lab actually pays.
2. Run a budget-bounded sliding-window agent at the same input-token budget
   (optionally scaled up) on the same scenario.
3. Score both and report accuracy at equal token budget.

If Recall Lab still wins when the recency baseline is given an equal or larger
input-token budget, the win is about validity-aware consolidation, not prompt
length. If the gap closes, the selective-consolidation claim is weakened for
this setup, which is exactly the kind of result the protocol wants surfaced.

Run with:
    python -m recall_lab.eval.equal_budget_trial
    python -m recall_lab.eval.equal_budget_trial \
        --scenario scenarios/relocation_chain.json --budget-scale 1.0
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from recall_lab.controls.budgeted import BudgetedSlidingWindowAgent
from recall_lab.eval.metrics import estimate_tokens
from recall_lab.eval.multiday_trial import (
    AgentTrialResult,
    AnswerRecord,
    load_scenario,
    make_output_dir,
    run_recall_trial,
    score_answer,
    selected_days,
)

DEFAULT_SCENARIO = Path("scenarios/relocation_chain.json")
DEFAULT_OUT_DIR = Path("reports/equal_budget")


def run_budgeted_trial(
    scenario: dict[str, Any],
    days: list[dict[str, Any]],
    input_token_budget: int,
    judge_samples: int = 1,
    verbose: bool = False,
) -> AgentTrialResult:
    """Run the scenario through a sliding window bounded by an input-token budget."""
    agent = BudgetedSlidingWindowAgent(input_token_budget=input_token_budget)
    # Stable name: the exact budget drifts run to run (it tracks Recall Lab's
    # measured cost), so keep it out of the name or variance grouping breaks.
    # The actual budget is recorded in the payload's budget_target field.
    result = AgentTrialResult(agent_name="budgeted_window")

    for day in days:
        label = day.get("label", "day")
        turns = day.get("turns", [])
        for i, user_turn in enumerate(turns, start=1):
            if verbose:
                print(f"[budgeted] {label}: turn {i}/{len(turns)}")
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
            print(f"[budgeted] {eval_label}: question {i}/{len(questions)}")
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
                failure_mode=mode.value,
                correct=correct,
                judge_votes=votes,
                output_tokens_estimate=estimate_tokens(response),
                input_tokens_estimate=agent.last_input_tokens,
            )
        )

    return result


def equal_budget_payload(
    scenario: dict[str, Any],
    days: list[dict[str, Any]],
    recall: AgentTrialResult,
    budgeted: AgentTrialResult,
    budget: int,
    budget_scale: float,
) -> dict[str, Any]:
    """Assemble the JSON-serializable result for an equal-budget trial."""
    return {
        "scenario_name": scenario.get("name"),
        "description": scenario.get("description"),
        "budget_scale": budget_scale,
        "recall_mean_input_tokens": round(recall.mean_chat_input_tokens, 1),
        "budget_target": budget,
        "days_run": [day.get("label") for day in days],
        "final_eval_questions": len(scenario.get("final_eval", {}).get("questions", [])),
        "agents": [
            {
                "agent_name": result.agent_name,
                "recall_accuracy": result.recall_accuracy,
                "mean_chat_input_tokens": round(result.mean_chat_input_tokens, 1),
                "output_tokens_estimate": result.output_tokens_estimate,
                "final_eval": [asdict(record) for record in result.eval_records],
            }
            for result in (recall, budgeted)
        ],
    }


def write_markdown_report(payload: dict[str, Any], path: Path) -> None:
    """Write a screenshot-friendly equal-budget report."""
    recall_agent, budgeted_agent = payload["agents"]
    lines = [
        f"# Equal-token-budget trial: {payload['scenario_name']}",
        "",
        payload.get("description") or "",
        "",
        "## Why this trial exists",
        "",
        "Earlier results compared Recall Lab against a deliberately starved "
        "two-turn sliding window. This trial hands a sliding-window baseline an "
        "input-token budget matched to what Recall Lab actually spends, so the "
        "comparison is no longer about prompt length.",
        "",
        "## Setup",
        "",
        f"- Recall Lab mean input tokens/turn: {payload['recall_mean_input_tokens']}",
        f"- Budget scale applied: {payload['budget_scale']}",
        f"- Budget given to the recency baseline: {payload['budget_target']} input tokens",
        f"- Days run: {', '.join(payload['days_run'])}",
        f"- Final eval questions: {payload['final_eval_questions']}",
        "",
        "## Results",
        "",
        "| Agent | Recall accuracy | Mean input tokens/turn |",
        "|---|---:|---:|",
        f"| {recall_agent['agent_name']} | {recall_agent['recall_accuracy']:.2f} | "
        f"{recall_agent['mean_chat_input_tokens']} |",
        f"| {budgeted_agent['agent_name']} | {budgeted_agent['recall_accuracy']:.2f} | "
        f"{budgeted_agent['mean_chat_input_tokens']} |",
        "",
        "## Read",
        "",
    ]

    recall_acc = recall_agent["recall_accuracy"]
    budget_acc = budgeted_agent["recall_accuracy"]
    if recall_acc > budget_acc:
        lines.append(
            "Recall Lab beat the recency baseline at a comparable input-token "
            "budget. The advantage is not explained by prompt length; the "
            "remaining difference is what each agent chose to put in that budget."
        )
    elif recall_acc == budget_acc:
        lines.append(
            "Recall Lab and the budget-matched recency baseline tied. At equal "
            "input-token budget this scenario does not separate them. A harder "
            "scenario is needed to show a difference."
        )
    else:
        lines.append(
            "The budget-matched recency baseline beat Recall Lab. For this "
            "scenario, equal token budget closes or reverses the gap. The "
            "selective-consolidation claim is weakened here and needs scrutiny."
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

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_equal_budget_trial(
    scenario_path: Path = DEFAULT_SCENARIO,
    out_dir: Path = DEFAULT_OUT_DIR,
    budget_scale: float = 1.0,
    max_days: int | None = None,
    judge_samples: int = 1,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run Recall Lab, derive its token budget, then run the matched baseline."""
    scenario = load_scenario(scenario_path)
    days = selected_days(scenario, max_days)
    out_dir = make_output_dir(out_dir, clean=True)

    if verbose:
        print("=== phase 1: Recall Lab (measuring its token budget) ===")
    recall = run_recall_trial(
        scenario, days, out_dir, judge_samples=judge_samples, verbose=verbose
    )

    budget = max(1, round(recall.mean_chat_input_tokens * budget_scale))
    if verbose:
        print(
            f"=== phase 2: budgeted recency baseline at {budget} input tokens "
            f"(scale {budget_scale}) ==="
        )
    budgeted = run_budgeted_trial(
        scenario, days, budget, judge_samples=judge_samples, verbose=verbose
    )

    payload = equal_budget_payload(scenario, days, recall, budgeted, budget, budget_scale)
    (out_dir / "equal_budget_result.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    write_markdown_report(payload, out_dir / "equal_budget_report.md")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an equal-token-budget trial.")
    parser.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--budget-scale",
        type=float,
        default=1.0,
        help="Multiplier on Recall Lab's mean input tokens. 1.0 is equal budget; "
        "use >1.0 to hand the recency baseline an advantage.",
    )
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--judge-samples", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_equal_budget_trial(
        scenario_path=args.scenario,
        out_dir=args.out_dir,
        budget_scale=args.budget_scale,
        max_days=args.max_days,
        judge_samples=args.judge_samples,
        verbose=args.verbose,
    )

    print(f"trial: {payload['scenario_name']}")
    print(f"recall mean input tokens/turn: {payload['recall_mean_input_tokens']}")
    print(f"budget given to recency baseline: {payload['budget_target']}")
    print()
    for agent in payload["agents"]:
        print(agent["agent_name"])
        print("  recall accuracy:", agent["recall_accuracy"])
        print("  mean input tokens/turn:", agent["mean_chat_input_tokens"])
    print()
    print(f"wrote {args.out_dir / 'equal_budget_result.json'}")
    print(f"wrote {args.out_dir / 'equal_budget_report.md'}")


if __name__ == "__main__":
    main()
