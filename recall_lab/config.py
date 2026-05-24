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

# Memory
WORKING_MAX_TURNS = 6  # recent turns injected alongside the brief
SLIDING_WINDOW_TURNS = 10  # for the sliding-window control
BRIEF_MAX_TOKENS = 2000  # cap on the consolidated brief size

# Salience scoring
SALIENCE_THRESHOLD = 0.5  # below this, exchange is not promoted to brief
DECAY_HALF_LIFE_DAYS = 14  # used in Forgetting Curves Lab (secondary experiment)
CONTRADICTION_COMPARE_LIMIT = int(os.environ.get("RECALL_CONTRADICTION_COMPARE_LIMIT", "3"))
