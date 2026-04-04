"""Baseline runner for the CLI Auto Fixer environment."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Protocol

from openai import OpenAI

try:
    from .client import CliAutoFixerEnv
    from .env import CliAutoFixerEnvironment
    from .models import CliAutoFixerAction, CliAutoFixerObservation
except ImportError:
    from client import CliAutoFixerEnv
    from env import CliAutoFixerEnvironment
    from models import CliAutoFixerAction, CliAutoFixerObservation


SYSTEM_PROMPT = """You are operating a terminal repair environment.
Return JSON only with this schema: {"command": "<single bash command>"}.
Rules:
- Use exactly one bash command per response.
- Prefer short exploratory commands first when the error is unclear.
- Avoid destructive commands.
- Finish each task by running its grader command once the fix is in place.
"""


@dataclass
class StepEnvelope:
    observation: CliAutoFixerObservation
    done: bool


class SupportsEnv(Protocol):
    def reset(self) -> StepEnvelope: ...
    def step(self, action: CliAutoFixerAction) -> StepEnvelope: ...
    def close(self) -> None: ...


class LocalEnvAdapter:
    """Small adapter to align direct environment calls with EnvClient results."""

    def __init__(self) -> None:
        self._env = CliAutoFixerEnvironment()

    def reset(self) -> StepEnvelope:
        observation = self._env.reset()
        return StepEnvelope(observation=observation, done=observation.done)

    def step(self, action: CliAutoFixerAction) -> StepEnvelope:
        observation = self._env.step(action)
        return StepEnvelope(observation=observation, done=observation.done)

    def close(self) -> None:
        self._env.close()


def build_env(base_url: str | None) -> SupportsEnv:
    if base_url:
        return CliAutoFixerEnv(base_url=base_url)
    return LocalEnvAdapter()


def propose_command_with_model(
    client: OpenAI,
    observation: CliAutoFixerObservation,
    model: str,
) -> str:
    reward_model = observation.metadata.get("reward_model", {})
    prompt = {
        "task_id": observation.metadata.get("task_id"),
        "task_name": observation.metadata.get("task_name"),
        "goal": observation.metadata.get("goal"),
        "cwd": observation.cwd,
        "last_command": observation.last_command,
        "stdout": observation.stdout,
        "stderr": observation.stderr,
        "exit_code": observation.exit_code,
        "score": reward_model,
    }
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(prompt)},
        ],
        temperature=0,
    )
    text = response.output_text.strip()
    payload = json.loads(text)
    command = payload["command"].strip()
    if not command:
        raise ValueError("Model returned an empty command.")
    return command


def scripted_command(observation: CliAutoFixerObservation) -> str:
    task_id = observation.metadata.get("task_id")
    stderr = observation.stderr
    stdout = observation.stdout

    if task_id == "task_1":
        if "No module named 'requests'" in stderr:
            return "pip install requests"
        return "python3 main.py"

    if task_id == "task_2":
        if not observation.last_command:
            return "cat package.json"
        if observation.last_command == "cat package.json":
            return "npm install"
        if "ERESOLVE" in stderr or "conflicting peer dependency" in stderr:
            return "npm install react-dom@18.2.0"
        return "npm run build"

    if task_id == "task_3":
        if "No module named 'psycopg2'" in stderr:
            return "pip install psycopg2"
        if "pg_config executable not found" in stderr or "libpq-fe.h" in stderr:
            return "sudo apt-get update && sudo apt-get install -y libpq-dev"
        if "task_3_ok" in stdout:
            return "python3 main.py"
        if not observation.last_command:
            return "python3 main.py"
        return "pip install psycopg2"

    return "pwd"


def run_episode(
    env: SupportsEnv,
    *,
    planner: str,
    model: str,
    max_steps: int,
) -> dict[str, object]:
    result = env.reset()
    observation = result.observation

    client = OpenAI() if planner == "openai" else None
    history: list[dict[str, object]] = []

    for _ in range(max_steps):
        if result.done:
            break

        if planner == "openai":
            command = propose_command_with_model(client, observation, model)  # type: ignore[arg-type]
        else:
            command = scripted_command(observation)

        result = env.step(CliAutoFixerAction(command=command))
        observation = result.observation
        history.append(
            {
                "command": command,
                "exit_code": observation.exit_code,
                "reward": observation.reward,
                "total_score": observation.metadata.get("reward_model", {}).get("total_score"),
                "done": result.done,
            }
        )

        if result.done:
            break

    return {
        "task_id": observation.metadata.get("task_id"),
        "task_name": observation.metadata.get("task_name"),
        "done": result.done,
        "final_exit_code": observation.exit_code,
        "final_score": observation.metadata.get("reward_model", {}).get("total_score"),
        "steps": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CLI Auto Fixer baseline.")
    parser.add_argument("--base-url", default=os.environ.get("OPENENV_BASE_URL"))
    parser.add_argument("--planner", choices=("openai", "scripted"), default="openai")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args()

    if args.planner == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for planner=openai. Use --planner scripted for offline smoke tests.")

    env = build_env(args.base_url)
    try:
        results = [
            run_episode(env, planner=args.planner, model=args.model, max_steps=args.max_steps)
            for _ in range(3)
        ]
    finally:
        env.close()

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
