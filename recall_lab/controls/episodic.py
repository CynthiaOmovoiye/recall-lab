"""Episodic read-time-judge baseline.

Keep every statement raw. No compression, no supersede, no consolidation. On
each turn, dump the entire conversation history into the prompt and ask the
model to work out the current answer at read time.

This is the control that paper arxiv 2605.12978, "Useful Memories Become Faulty
When Continuously Updated by LLMs", says should win. It found that repeatedly
rewriting memory degrades it, and that keeping raw traces and deciding at read
time beat the rewriting approaches. Recall Lab's sleep job is exactly the
rewriting pattern that paper punishes, so this baseline is the sharpest test of
whether consolidation earns its place.

The question it answers: does validity-state consolidation actually beat just
keeping everything raw? Two ways the brief can still win:

- Accuracy. If raw history confuses the model on a long correction chain, the
  brief's explicit `Past, no longer current` labelling wins on correctness.
- Cost. Even if raw ties on accuracy here, it pays a growing input-token bill,
  the full transcript every turn, while the brief stays bounded. The crossover,
  where long logs make raw too expensive and consolidation starts to pay, is
  the Chapter 3 result.

The agent reports `last_input_tokens` so the runner can chart that growing bill
against the brief's flat one.
"""

from __future__ import annotations

from recall_lab.config import (
    AGENT_MODEL,
    MAX_OUTPUT_TOKENS,
    OPENROUTER_API_KEY,
)
from recall_lab.eval.metrics import estimate_tokens
from recall_lab.llm import chat_client, complete

EPISODIC_PREAMBLE = (
    "You are a helpful assistant. Below is the complete, raw history of this "
    "conversation in time order, oldest first. Treat it as your memory. Facts "
    "can change over time: if a later turn corrects an earlier one, the later "
    "statement is what is currently true, but the earlier one is still the "
    "answer to questions about the past. Work out the current answer from the "
    "full history. If a personal fact the user asks for is not in the history, "
    "say you do not know rather than guessing."
)


class EpisodicJudgeAgent:
    """Raw-history baseline with read-time reasoning.

    Keeps every exchange verbatim and injects the whole log each turn. No
    consolidation pass ever runs. Implements the `.respond(str) -> str` agent
    protocol used by the harness.
    """

    def __init__(self) -> None:
        self.history: list[dict] = []
        self.last_input_tokens = 0

    def _render_history(self) -> str:
        """Render the full raw log oldest-first, or a placeholder if empty."""
        if not self.history:
            return "(no history yet)"
        lines = []
        for i, exchange in enumerate(self.history, start=1):
            lines.append(f"[{i}] User: {exchange['user']}")
            lines.append(f"[{i}] Assistant: {exchange['agent']}")
        return "\n".join(lines)

    def respond(self, user_message: str) -> str:
        """Inject the entire raw history, answer at read time, then append."""
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is required to run EpisodicJudgeAgent.")

        prompt = "\n".join(
            [
                EPISODIC_PREAMBLE,
                "",
                "## Full conversation history",
                self._render_history(),
                "",
                "## Current user message",
                user_message,
            ]
        ).strip()
        self.last_input_tokens = estimate_tokens(prompt)

        client = chat_client()
        completion = complete(
            client,
            model=AGENT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        response = completion.choices[0].message.content or ""

        self.history.append({"user": user_message, "agent": response})
        return response
