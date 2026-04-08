"""Typed models for the CLI Auto Fixer environment."""

from typing import Any

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


class CliAutoFixerAction(Action):
    """A single bash command issued by the agent."""

    command: str = Field(..., description="A single bash command to execute.")


class CliAutoFixerReward(BaseModel):
    """Reward bookkeeping for the current step and full episode."""

    step_reward: float = Field(..., description="Incremental reward for this step.")
    total_score: float = Field(
        ..., description="Episode score clamped to the inclusive range [0.0, 1.0]."
    )


class CliAutoFixerObservation(Observation):
    """Observed shell state after executing the last command."""

    cwd: str = Field(..., description="Current working directory of the sandbox.")
    last_command: str = Field(..., description="The bash command executed in the last step.")
    stdout: str = Field(..., description="Stdout truncated to the last 200 lines.")
    stderr: str = Field(..., description="Stderr truncated to the last 200 lines.")
    exit_code: int = Field(..., description="Exit code from the previous command.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Task metadata and reward bookkeeping.",
    )
