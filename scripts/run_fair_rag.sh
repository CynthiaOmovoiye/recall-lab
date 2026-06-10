#!/usr/bin/env bash
#
# Fair RAG sweep. Tests whether the strong-RAG result in Chapter 3 is an artifact
# of two knobs (recency weight, top_k), and whether a fully loaded production RAG
# can order a correction chain.
#
# Configs: recency ablation (rw 0.0 / 0.3 / 0.6 at k=5), top_k sweep
# (k 5 / 10 / full at rw=0.3), and a fair shot (full k + timestamps in context).
# Reference baselines (vector, episodic, brief) are cited from v12, not re-run.
#
# Run from the repo root, in a real terminal. The key is read from .env by
# config.py, the same as every campaign. No export needed.
#
#   bash scripts/run_fair_rag.sh
#
# Cost: the kFULL and fair-shot configs inject the whole log each turn, so they
# are token-heavy. Default RUNS=3 keeps the bill down; set RUNS=5 for final
# numbers. Six configs at 3 seeds is roughly 18 trials.

set -euo pipefail

RUNS="${RUNS:-3}"
SCENARIO="${SCENARIO:-scenarios/relocation_chain.json}"
LABEL="${LABEL:-v14_fair_rag}"
# The judge (Claude Sonnet) is the cost driver. Prior audits showed 0 split
# verdicts across 370+ graded answers, so 1 call is enough and ~3x cheaper than
# the 3-call audit. Raise to 3 only if you want the audit back.
JUDGE_SAMPLES="${JUDGE_SAMPLES:-1}"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON="${VIRTUAL_ENV}/bin/python"
  elif [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
  else
    echo "No Python interpreter found. Activate your venv or set PYTHON=." >&2
    exit 1
  fi
fi

echo "=== Fair RAG sweep ==="
echo "python:        ${PYTHON}"
echo "scenario:      ${SCENARIO}"
echo "runs:          ${RUNS}"
echo "judge_samples: ${JUDGE_SAMPLES}"
echo "label:         ${LABEL}"
echo

"${PYTHON}" -m recall_lab.eval.fair_rag_sweep \
  --runs "${RUNS}" \
  --scenario "${SCENARIO}" \
  --judge-samples "${JUDGE_SAMPLES}" \
  --label "${LABEL}"

echo
echo "=== done ==="
echo "summary: reports/${LABEL}/fair_rag_summary.md"
echo
echo "Next: read the per-question table. If the first city stays 0 at rw=0.0 and"
echo "small k, recency did not bias the result. If it is solved only at kFULL or"
echo "fair-shot, report that as 'RAG can order the chain, at keep-everything cost'."
