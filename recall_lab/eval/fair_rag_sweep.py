"""Fair RAG sweep: is the strong-RAG result an artifact of two knobs?

Chapter 3 reported strong RAG at 0.76, failing the first city in the relocation
chain. That run used recency_weight=0.30 and top_k=5. Both are choices, and the
first-city miss was partly a coverage effect: with only 5 snippets in context,
the recency boost pushed the oldest city out of retrieval, so the model abstained.
This sweep removes the bias question by varying the two knobs and by giving the
pipeline a genuinely fair shot.

Configs (all on the same scenario, same query-rewrite + hybrid + RRF + rerank):

  recency ablation at top_k=5:   rw=0.0, rw=0.3, rw=0.6
  top_k sweep at rw=0.3:         k=5 (above), k=10, k=full
  fair shot:                     k=full + timestamps visible in the context

How to read it:

- If the first city stays at 0 with rw=0.0 and at small k, recency did not cause
  the failure; the pipeline simply cannot order the chain.
- If the chain is solved only at k=full with timestamps, then a fully loaded RAG
  can order it, but that config is the keep-everything agent under another name,
  and the cost column shows the token bill the bounded validity brief avoids.

Reference baselines from the v12 5-seed campaign (not re-run here): flat vector
0.52, raw episodic 1.00 at ~974 tokens/turn, validity brief 1.00 at ~438.

Run from the repo root, key from .env (see run_fair_rag.sh):

    python -m recall_lab.eval.fair_rag_sweep --runs 5
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from recall_lab.eval.multiday_trial import (
    load_scenario,
    run_strong_rag_trial,
    selected_days,
)

DEFAULT_SCENARIO = Path("scenarios/relocation_chain.json")
FULL_K = 100  # larger than the scenario length, so the whole log can reach context

# name -> kwargs for run_strong_rag_trial
CONFIGS: list[tuple[str, dict[str, Any]]] = [
    ("strong_rw0.0_k5", dict(recency_weight=0.0, top_k=5, candidate_k=12)),
    ("strong_rw0.3_k5", dict(recency_weight=0.3, top_k=5, candidate_k=12)),
    ("strong_rw0.6_k5", dict(recency_weight=0.6, top_k=5, candidate_k=12)),
    ("strong_rw0.3_k10", dict(recency_weight=0.3, top_k=10, candidate_k=20)),
    ("strong_rw0.3_kFULL", dict(recency_weight=0.3, top_k=FULL_K, candidate_k=FULL_K)),
    (
        "strong_fairshot",
        dict(recency_weight=0.3, top_k=FULL_K, candidate_k=FULL_K, show_timestamps=True),
    ),
    (
        "strong_fairshot_chrono",
        dict(
            recency_weight=0.3,
            top_k=FULL_K,
            candidate_k=FULL_K,
            show_timestamps=True,
            chronological=True,
        ),
    ),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fair RAG knob sweep on the relocation chain.")
    ap.add_argument("--runs", type=int, default=3, help="Seeds per config. 3 is cheap, 5 for final.")
    ap.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    ap.add_argument("--judge-samples", type=int, default=3)
    ap.add_argument("--label", default="v14_fair_rag")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    scenario = load_scenario(args.scenario)
    days = selected_days(scenario, None)

    rows: dict[str, dict[str, Any]] = {}
    for name, cfg in CONFIGS:
        accs: list[float] = []
        toks: list[float] = []
        perq: dict[str, list[bool]] = defaultdict(list)
        for seed in range(1, args.runs + 1):
            print(f"\n=== {name} seed {seed}/{args.runs} ===", flush=True)
            try:
                res = run_strong_rag_trial(
                    scenario,
                    days,
                    judge_samples=args.judge_samples,
                    verbose=False,
                    agent_name=name,
                    **cfg,
                )
            except Exception as exc:  # noqa: BLE001 - one bad run must not kill the sweep
                print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
                continue
            accs.append(res.recall_accuracy)
            toks.append(res.mean_chat_input_tokens)
            for r in res.eval_records:
                perq[r.user].append(bool(r.correct))
            print(f"  acc={res.recall_accuracy:.2f}  mean_chat_in_tok={res.mean_chat_input_tokens:.0f}", flush=True)
        rows[name] = {"accs": accs, "toks": toks, "perq": perq}

    out_dir = Path("reports") / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Fair RAG sweep, {scenario.get('name', 'scenario')}",
        "",
        f"{args.runs} seeds per config. Same scenario and pipeline; only the named knobs change.",
        "",
        "Reference (v12 5-seed, not re-run): flat vector 0.52, raw episodic 1.00 at ~974 tok, "
        "validity brief 1.00 at ~438 tok.",
        "",
        "## Accuracy and cost",
        "",
        "| Config | mean acc | mean chat input tokens | runs |",
        "|---|---|---|---|",
    ]
    for name, _ in CONFIGS:
        d = rows[name]
        if d["accs"]:
            lines.append(
                f"| {name} | {statistics.mean(d['accs']):.2f} | "
                f"{statistics.mean(d['toks']):.0f} | {len(d['accs'])} |"
            )
        else:
            lines.append(f"| {name} | (no runs completed) | - | 0 |")

    lines += ["", "## Per-question pass count", ""]
    for name, _ in CONFIGS:
        d = rows[name]
        if not d["perq"]:
            continue
        lines.append(f"### {name}")
        for q, vals in d["perq"].items():
            lines.append(f"- {sum(vals)}/{len(vals)}  {q}")
        lines.append("")

    path = out_dir / "fair_rag_summary.md"
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
