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
# false-flagged a benign prompt as a jailbreak. So routing is constrained here.
#
# - IGNORE drops named providers entirely. Azure is ignored by default to stop
#   the content-filter false-positive. Ignoring a provider that does not serve
#   a given model is a harmless no-op, so this is safe to apply to every call.
# - ORDER pins a preferred provider order (comma-separated slugs). Leave empty
#   by default: a global order would break calls to a model that provider does
#   not serve (e.g. forcing "OpenAI" onto the Anthropic judge). Set it only when
#   every model in play is served by the listed providers.
# - ALLOW_FALLBACKS lets routing fall back past the constraints when a provider
#   is down. Set false for strict reproducibility, accepting that a provider
#   outage then fails the call instead of silently rerouting.
OPENROUTER_IGNORE_PROVIDERS = os.environ.get("RECALL_OPENROUTER_IGNORE_PROVIDERS", "Azure")
OPENROUTER_PROVIDER_ORDER = os.environ.get("RECALL_OPENROUTER_PROVIDER_ORDER", "")
OPENROUTER_ALLOW_FALLBACKS = os.environ.get(
    "RECALL_OPENROUTER_ALLOW_FALLBACKS", "true"
).strip().lower() in {"1", "true", "yes"}

# Memory
WORKING_MAX_TURNS = 6  # recent turns injected alongside the brief
SLIDING_WINDOW_TURNS = 10  # for the sliding-window control
BRIEF_MAX_TOKENS = 2000  # cap on the consolidated brief size

# Salience scoring
SALIENCE_THRESHOLD = 0.5  # below this, exchange is not promoted to brief
DECAY_HALF_LIFE_DAYS = 14  # used in Forgetting Curves Lab (secondary experiment)
