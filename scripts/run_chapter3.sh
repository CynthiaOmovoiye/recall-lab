#!/usr/bin/env bash
#
# Chapter 3 experiment campaign.
#
# Runs the full retrieval lineup against the relocation chain, plus the
# budget-matched recency control, under the pinned provider with the 3-call
# judge audit. This is the data behind Chapter 3: "What RAG gets right about
# human memory, and what it still misses."
#
# Lineup (--agents all): sliding window, flat vector (naive RAG), strong RAG
# (query rewrite + hybrid + recency + rerank), strong RAG with date metadata
# (timestamp recency + metadata-by-date filtering), raw episodic read-time judge,
# and Recall Lab's validity brief. The equal-budget pass adds the recency
# baseline matched to Recall Lab's token budget.
#
# The date-filter window for strong_rag_dated comes from .env
# (RECALL_STRONG_RAG_RECENCY_WINDOW_DAYS, default 1.5 days).
#
# Run from the repo root, in a real terminal (not a session-bound job). The
# research log notes that background runs get killed mid-campaign.
#
#   bash scripts/run_chapter3.sh
#
# The OpenRouter key is read from .env by recall_lab/config.py (load_dotenv),
# the same as every other campaign. No export step, no shell variable to set.
#
# Cost and time: each run is roughly 250 API calls. Two campaigns at 5 runs is
# about an hour of wall time and a few dollars of OpenRouter credit at
# gpt-4o-mini agent + claude judge prices. Tune RUNS down to smoke-test first.

set -euo pipefail

RUNS="${RUNS:-5}"
SCENARIO="${SCENARIO:-scenarios/relocation_chain.json}"
LABEL="${LABEL:-v13_chapter3}"

# Pick the interpreter where the deps live. Prefer an active venv, then a local
# .venv, then python3, then python. Override with PYTHON=/path/to/python.
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

echo "=== Chapter 3 campaign ==="
echo "python:   ${PYTHON}"
echo "scenario: ${SCENARIO}"
echo "runs:     ${RUNS}"
echo "label:    ${LABEL}"
echo

echo "--- 1/2: full lineup (sliding, vector, strong_rag, strong_rag_dated, episodic, recall) ---"
"${PYTHON}" -m recall_lab.eval.variance \
  --runs "${RUNS}" \
  --scenario "${SCENARIO}" \
  --agents all \
  --label "${LABEL}_lineup"

echo
echo "--- 2/2: budget-matched recency control vs Recall Lab ---"
"${PYTHON}" -m recall_lab.eval.variance \
  --runs "${RUNS}" \
  --scenario "${SCENARIO}" \
  --mode equal-budget \
  --label "${LABEL}_equal_budget"

echo
echo "=== done ==="
echo "summaries:"
echo "  reports/variance/${LABEL}_lineup/variance_summary.md"
echo "  reports/variance/${LABEL}_equal_budget/variance_summary.md"
echo
echo "Next: fold the headline table into the Chapter 3 draft, replacing the"
echo "'what I have not tested yet' section with the measured numbers."
