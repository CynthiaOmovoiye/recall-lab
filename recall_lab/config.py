"""Runtime configuration for Recall Lab.

Reads environment from .env in the repo root. Centralises every tunable so
experiments are reproducible from a single file.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
EPISODIC_DB_PATH = DATA_DIR / "episodic" / "log.db"
BRIEF_PATH = REPO_ROOT / "brief.md"
TRACE_STORE_PATH = DATA_DIR / "memory_traces.jsonl"
RESEARCH_LOG_PATH = REPO_ROOT / "research-log.md"

# LLM (OpenRouter via the OpenAI client)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
AGENT_MODEL = os.environ.get("RECALL_AGENT_MODEL", "openai/gpt-4o-mini")
JUDGE_MODEL = os.environ.get("RECALL_JUDGE_MODEL", "anthropic/claude-sonnet-4.6")
# Spotting an implicit correction is the hardest reasoning call in the system.
# It defaults to the judge model, not the cheap agent model.
CONTRADICTION_MODEL = os.environ.get("RECALL_CONTRADICTION_MODEL", JUDGE_MODEL)
# Cap on output tokens per call. Every Recall Lab call returns a short result:
# a chat turn or a small JSON verdict. Without a cap, OpenRouter pre-authorizes
# the model's full output budget and rejects the call when credit runs low.
MAX_OUTPUT_TOKENS = int(os.environ.get("RECALL_MAX_OUTPUT_TOKENS", "1024"))
# Resilience for the OpenRouter client. A transient connection blip during a
# long variance batch should be absorbed by retries, not fail a whole run.
OPENROUTER_MAX_RETRIES = int(os.environ.get("RECALL_OPENROUTER_MAX_RETRIES", "6"))
OPENROUTER_TIMEOUT_SECONDS = float(os.environ.get("RECALL_OPENROUTER_TIMEOUT", "60"))

# Provider routing. OpenRouter routes a model across providers
# non-deterministically. That adds variance unrelated to the memory
# architecture, and once killed a run outright when Azure's content filter
# false-flagged a benign prompt as a jailbreak. So routing is pinned here for
# reproducibility.
#
# The agent model openai/gpt-4o-mini is OpenAI-proprietary: on OpenRouter only
# OpenAI and Azure serve it. Azure was the false-positive. OpenAI direct runs no
# extra jailbreak filter on top of OpenAI's own moderation, so it is the pinned
# provider. The pin is OpenAI, fallbacks are off, and Azure stays excluded.
#
# - ORDER pins a preferred provider order (comma-separated slugs). The order is
#   applied per model family: llm.py applies it only to a call whose model
#   vendor matches the order's provider (openai/* -> OpenAI). PROVIDER_ORDER
#   pins the agent model; JUDGE_PROVIDER_ORDER pins the Anthropic judge and
#   contradiction classifier. A model matching neither keeps its own routing.
# - ALLOW_FALLBACKS off means a pinned call that the provider cannot serve
#   errors instead of silently rerouting. That is the point: no silent provider
#   switch mid-campaign. Transient blips are still absorbed by client retries.
# - IGNORE drops named providers entirely, on every call. Azure stays ignored
#   so a non-OpenAI model can never land on it either. Ignoring a provider that
#   does not serve a given model is a harmless no-op.
#
# Both model families are pinned so the 30-conversation campaign is fully
# reproducible: a reviewer can say which provider scored which run, not
# "whichever Anthropic node the load balancer sent it to."
OPENROUTER_PROVIDER_ORDER = os.environ.get("RECALL_OPENROUTER_PROVIDER_ORDER", "OpenAI")
OPENROUTER_JUDGE_PROVIDER_ORDER = os.environ.get(
    "RECALL_OPENROUTER_JUDGE_PROVIDER_ORDER", "Anthropic"
)
OPENROUTER_ALLOW_FALLBACKS = os.environ.get(
    "RECALL_OPENROUTER_ALLOW_FALLBACKS", "false"
).strip().lower() in {"1", "true", "yes"}
OPENROUTER_IGNORE_PROVIDERS = os.environ.get("RECALL_OPENROUTER_IGNORE_PROVIDERS", "Azure")

# Memory
WORKING_MAX_TURNS = 6  # recent turns injected alongside the brief
SLIDING_WINDOW_TURNS = 10  # for the sliding-window control
BRIEF_MAX_TOKENS = 2000  # cap on the consolidated brief size

# Salience scoring
SALIENCE_THRESHOLD = 0.5  # below this, exchange is not promoted to brief
DECAY_HALF_LIFE_DAYS = 14  # used in Forgetting Curves Lab (secondary experiment)

# Strong RAG control. The `strong_rag` agent uses turn-order recency. The
# `strong_rag_dated` agent uses real date metadata: timestamp recency plus a
# date-window filter (metadata filtering by date added). The window is how many
# days of history survive the filter; older candidates are dropped before
# reranking. Empty disables the hard filter, leaving timestamp recency only.
_window = os.environ.get("RECALL_STRONG_RAG_RECENCY_WINDOW_DAYS", "1.5").strip()
STRONG_RAG_RECENCY_WINDOW_DAYS = float(_window) if _window else None
