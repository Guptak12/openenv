---
title: CLI Auto Fixer
emoji: "🛠️"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
tags:
  - openenv
  - reinforcement-learning
  - docker
---

# CLI Auto Fixer

`cli_auto_fixer` is an OpenEnv reinforcement learning environment that simulates a broken developer terminal. The agent receives shell output, issues a single bash command per step, and must repair dependency failures until the target script succeeds.

## Tasks

The environment cycles deterministically through three task templates on each `reset()`:

1. `task_1` (Easy): a Python script imports `requests` with no dependency installed.
2. `task_2` (Medium): a Node.js project contains a React / React DOM peer dependency conflict.
3. `task_3` (Hard): a Python script needs `psycopg2`, which initially fails to build because `libpq-dev` is missing from the container.

Each episode copies the selected broken project into a fresh temporary directory and executes every agent command inside that isolated sandbox.

## Spaces

Action space:

- `command: str`

Observation space:

- `cwd: str`
- `last_command: str`
- `stdout: str` (truncated to the last 200 lines)
- `stderr: str` (truncated to the last 200 lines)
- `exit_code: int`

Reward model:

- `step_reward: float`
- `total_score: float`

`step_reward` is exposed through the OpenEnv `reward` field, and the full reward model is attached under `observation.metadata["reward_model"]`.

## Reward Shaping

- `-0.1` for non-zero exit codes.
- `-0.2` when the exact same failed command is repeated twice in a row.
- `-1.0` with immediate termination for destructive commands such as `rm -rf /` or `reboot`.
- `+0.1` for exploratory commands such as `ls`, `cat`, `find`, or `rg`.
- `+0.3` for the first successful installation command in an episode.
- On success, the episode terminates and `total_score` is set to `1.0`.

## Files

- [models.py](/home/stackedshadow/openenv/models.py)
- [env.py](/home/stackedshadow/openenv/env.py)
- [baseline.py](/home/stackedshadow/openenv/baseline.py)
- [openenv.yaml](/home/stackedshadow/openenv/openenv.yaml)
- [Dockerfile](/home/stackedshadow/openenv/Dockerfile)

## Local Usage

Install dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

Run the server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Validate the environment:

```bash
openenv validate .
```

Run the baseline:

```bash
python3 baseline.py --planner scripted
```

For the OpenAI-driven baseline, set `OPENAI_API_KEY` and use the default `--planner openai`, which queries `gpt-4o-mini`.

## Baseline Scores

Expected baseline behavior:

- `scripted` baseline: solves task 1 and task 2 in local smoke tests; task 3 is intended to complete inside the Docker image where `apt-get install libpq-dev` is safe to run.
- `openai` baseline: depends on model behavior, but the prompt is constrained to return one bash command as JSON for each step.
