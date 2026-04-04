"""Client helpers for the CLI Auto Fixer environment."""

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import CliAutoFixerAction, CliAutoFixerObservation


class CliAutoFixerEnv(
    EnvClient[CliAutoFixerAction, CliAutoFixerObservation, State]
):
    """Typed HTTP client for a running CLI Auto Fixer environment."""

    def _step_payload(self, action: CliAutoFixerAction) -> dict[str, Any]:
        return {"command": action.command}

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[CliAutoFixerObservation]:
        obs_data = payload.get("observation", {})
        observation = CliAutoFixerObservation(
            cwd=obs_data.get("cwd", ""),
            last_command=obs_data.get("last_command", ""),
            stdout=obs_data.get("stdout", ""),
            stderr=obs_data.get("stderr", ""),
            exit_code=obs_data.get("exit_code", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
