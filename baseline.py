"""Baseline runner for the CLI Auto Fixer environment."""

from __future__ import annotations

import argparse
import asyncio
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


SYSTEM_PROMPT = """You are an autonomous AI Site Reliability Engineer (SRE) tasked with fixing broken development environments. You will receive the current state of a terminal sandbox, including the last executed command, standard output, standard error, and the exit code.

Your goal is to diagnose missing dependencies, resolve version conflicts, and successfully execute the target script to achieve the task objective.

CRITICAL DIRECTIVES:
1. RECONNAISSANCE FIRST: Never blindly guess. If you do not know the exact error, your very first command MUST be to execute the target script (e.g., `python3 main.py` or `npm run build`) to generate an error log in `stderr`.
2. READ THE LOGS CAREFULLY: 
   - Python: If a module is missing, run `pip install <module>`.
   - Node.js: If a dependency conflicts (e.g., ERESOLVE), read the config with `cat package.json` to understand the required versions before installing.
   - OS/C++: If a package build fails due to missing system headers (e.g., `libpq-fe.h`), you MUST chain your command: `sudo apt-get update && sudo apt-get install -y <package>`.
3. NO REPETITION: Never repeat a command that just returned a non-zero exit code. If a fix failed, read the new error output and change your approach completely.
4. VERIFICATION: Once you believe the dependencies are fixed, you must execute the target script one last time to verify exit code 0.

OUTPUT FORMAT:
You must respond with raw, valid JSON only. Do not wrap the response in markdown blocks (no ```json). Use exactly this schema:
{
  "reasoning": "Briefly explain what the last error was and why you are choosing the next command.",
  "command": "<single bash command to execute>"
}
"""


@dataclass
class StepEnvelope:
    observation: CliAutoFixerObservation
    done: bool


@dataclass
class EpisodeContext:
    task_id: str | None = None
    task_name: str | None = None
    goal: str | None = None
    total_score: float | None = None


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
        async_env = CliAutoFixerEnv(base_url=base_url)

        class SyncRemoteAdapter:
            def __init__(self) -> None:
                self._loop = asyncio.new_event_loop()

            def reset(self) -> StepEnvelope:
                res = self._loop.run_until_complete(async_env.reset())
                # Unpack the HTTP client's wrapper if it exists
                obs = getattr(res, 'observation', res)
                done = getattr(res, 'done', getattr(obs, 'done', False))
                return StepEnvelope(observation=obs, done=done)

            def step(self, action: CliAutoFixerAction) -> StepEnvelope:
                res = self._loop.run_until_complete(async_env.step(action))
                # Unpack the HTTP client's wrapper if it exists
                obs = getattr(res, 'observation', res)
                done = getattr(res, 'done', getattr(obs, 'done', False))
                return StepEnvelope(observation=obs, done=done)

            def close(self) -> None:
                try:
                    self._loop.run_until_complete(async_env.close())
                finally:
                    self._loop.close()

        return SyncRemoteAdapter()

    return LocalEnvAdapter()


def infer_episode_context(
    observation: CliAutoFixerObservation,
    previous: EpisodeContext | None = None,
) -> EpisodeContext:
    metadata = observation.metadata or {}
    task_id = metadata.get("task_id")
    task_name = metadata.get("task_name")
    goal = metadata.get("goal")
    total_score = metadata.get("reward_model", {}).get("total_score")

    lines = observation.stdout.splitlines()
    if lines and lines[0].startswith("Task "):
        header = lines[0][len("Task ") :]
        parsed_task_id, _, parsed_task_name = header.partition(": ")
        task_id = task_id or parsed_task_id or None
        task_name = task_name or parsed_task_name or None
    for line in lines:
        if line.startswith("Goal: "):
            goal = goal or line[len("Goal: ") :]
            break

    if total_score is None and previous is not None:
        reward = observation.reward
        if isinstance(reward, (int, float)):
            if observation.done and float(reward) == 1.0:
                total_score = 1.0
            else:
                prior = previous.total_score or 0.0
                total_score = max(0.0, min(0.9, prior + float(reward)))
        else:
            total_score = previous.total_score

    return EpisodeContext(
        task_id=task_id or (previous.task_id if previous else None),
        task_name=task_name or (previous.task_name if previous else None),
        goal=goal or (previous.goal if previous else None),
        total_score=total_score if total_score is not None else (previous.total_score if previous else None),
    )


def propose_command_with_model(
    client: OpenAI,
    observation: CliAutoFixerObservation,
    model: str,
    command_history: list[str],
    context: EpisodeContext,
) -> str:
    reward_model = observation.metadata.get("reward_model", {})
    if context.total_score is not None and "total_score" not in reward_model:
        reward_model = {**reward_model, "total_score": context.total_score}
    prompt = {
        "task_id": observation.metadata.get("task_id") or context.task_id,
        "task_name": observation.metadata.get("task_name") or context.task_name,
        "goal": observation.metadata.get("goal") or context.goal,
        "cwd": observation.cwd,
        "last_command": observation.last_command,
        "stdout": observation.stdout,
        "stderr": observation.stderr,
        "exit_code": observation.exit_code,
        "score": reward_model,
        "previously_tried_commands": command_history,
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


def scripted_command(
    observation: CliAutoFixerObservation,
    context: EpisodeContext | None = None,
) -> str:
    task_id = observation.metadata.get("task_id") or (context.task_id if context else None)
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
    context = infer_episode_context(observation)

    client = OpenAI() if planner == "openai" else None
    history: list[dict[str, object]] = []

    for _ in range(max_steps):
        if result.done:
            break

        if planner == "openai":
            past_commands = [str(step["command"]) for step in history]
            command = propose_command_with_model(  # type: ignore[arg-type]
                client, observation, model, past_commands, context
            )
        else:
            command = scripted_command(observation, context)

        result = env.step(CliAutoFixerAction(command=command))
        observation = result.observation
        context = infer_episode_context(observation, context)
        history.append(
            {
                "command": command,
                "exit_code": observation.exit_code,
                "reward": observation.reward,
                "total_score": context.total_score,
                "done": result.done,
            }
        )

        if result.done:
            break

    return {
        "task_id": context.task_id,
        "task_name": context.task_name,
        "done": result.done,
        "final_exit_code": observation.exit_code,
        "final_score": context.total_score,
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
