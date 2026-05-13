"""Run a multi-turn conversation through any agent and capture outputs.

Used to compare the Recall Lab agent against the controls on the same task.
The harness is agent-agnostic: any object with a `.respond(str) -> str` method
can be run through it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol


class Agent(Protocol):
    def respond(self, user_message: str) -> str: ...


@dataclass
class TurnRecord:
    """One turn of a recorded run."""

    index: int
    user: str
    agent: str
    elapsed_seconds: float
    tokens: int | None = None


@dataclass
class RunResult:
    """One full conversation run through one agent."""

    agent_name: str
    turns: list[TurnRecord] = field(default_factory=list)
    total_elapsed_seconds: float = 0.0


def run_conversation(agent: Agent, user_turns: list[str], agent_name: str) -> RunResult:
    """Feed a list of user turns through the agent, collect responses."""
    result = RunResult(agent_name=agent_name)
    started = time.time()
    for i, user_turn in enumerate(user_turns):
        turn_started = time.time()
        response = agent.respond(user_turn)
        elapsed = time.time() - turn_started
        result.turns.append(
            TurnRecord(
                index=i,
                user=user_turn,
                agent=response,
                elapsed_seconds=elapsed,
            )
        )
    result.total_elapsed_seconds = time.time() - started
    return result
