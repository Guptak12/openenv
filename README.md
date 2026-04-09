# CLI Auto Fixer: Autonomous SRE Environment

**A containerized OpenEnv benchmark for training and evaluating autonomous agents that repair broken developer environments from terminal feedback.**

## The Problem

Code generation is only part of real software work. Agents also need to survive broken environments: missing Python packages, Node dependency conflicts, and system-level build failures. These are not one-shot prompt problems. The agent has to inspect terminal output, choose a command, observe the result, and adapt.

That makes environment repair a sequential decision-making problem, which is exactly what OpenEnv is designed to benchmark.

## The Environment

`cli_auto_fixer` is an OpenEnv environment where the action space is one bash command per step. On every reset, the environment creates a fresh isolated workspace, drops the agent into a broken project, and asks it to recover the system until the target command succeeds.

Each step returns:

- `cwd`
- `last_command`
- `stdout`
- `stderr`
- `exit_code`
- `reward`
- `metadata`, including task info and reward bookkeeping

The environment is deterministic across resets and rotates through three task templates.

## Tasks

### 1. Missing Python Package

The agent encounters a Python script that imports `requests` without the dependency being installed. It must inspect the traceback and repair the environment by installing the missing package before rerunning the target script.

### 2. Node.js Version Conflict

The agent enters a broken JavaScript project with an `npm` dependency conflict. It must inspect `package.json`, reason about the dependency mismatch, and apply the correct fix so the build succeeds.

### 3. OS-Level Dependency Failure

The hardest task simulates a Python package build failure caused by missing system headers for `psycopg2`. The agent must diagnose the failure, use `sudo apt-get` inside the container, install `libpq-dev`, and then complete the Python setup.

## Reward Model

The environment exposes `step_reward` through the OpenEnv `reward` field and keeps the full reward state in `observation.metadata["reward_model"]`.

Current shaping:

- `+1.0` on successful task completion
- `-0.1` for a non-zero exit code
- `-0.2` for repeating the same failed command twice in a row
- `-1.0` and immediate termination for destructive commands
- `+0.1` for exploratory commands such as `ls`, `cat`, `find`, or `rg`
- `+0.3` for the first successful install command in an episode

Episode `total_score` is clamped into `[0.0, 1.0]`.

## Why This Benchmark Matters

This benchmark targets a failure mode common in coding agents: they can generate plausible fixes, but they struggle to navigate multi-step operational debugging. CLI Auto Fixer tests whether an agent can:

- read terminal output precisely
- avoid repeating failed actions
- explore before acting
- perform both application-level and OS-level remediation
- verify that the repair actually worked

## Included Agents

### `baseline.py`

The repository includes a baseline runner with two planner modes:

- `scripted`: deterministic hand-written policy for smoke testing
- `openai`: model-based planner using the OpenAI Python client

The model-based baseline sends the current shell state plus command history to the model and expects one bash command back as raw JSON.

### `inference.py`

The repository also includes an evaluation-style inference script that:

- uses the OpenAI Python client for model calls
- uses the evaluator-provided `API_BASE_URL` and `API_KEY`
- can run against a local Docker image via `LOCAL_IMAGE_NAME`
- can run against a remote OpenEnv server via `OPENENV_BASE_URL`
- emits strict `[START]`, `[STEP]`, and `[END]` lines for benchmarking

## Project Files

- [env.py](/home/stackedshadow/openenv/env.py): core environment logic
- [models.py](/home/stackedshadow/openenv/models.py): action and observation schemas
- [client.py](/home/stackedshadow/openenv/client.py): typed OpenEnv HTTP client
- [baseline.py](/home/stackedshadow/openenv/baseline.py): baseline runner
- [inference.py](/home/stackedshadow/openenv/inference.py): benchmark-style inference runner
- [openenv.yaml](/home/stackedshadow/openenv/openenv.yaml): OpenEnv metadata
- [Dockerfile](/home/stackedshadow/openenv/Dockerfile): deployment container

## Local Setup

Install dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

Run the environment server locally:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Validate the environment:

```bash
openenv validate .
```

## Running the Baseline

Run the scripted baseline locally:

```bash
python3 baseline.py --planner scripted
```

Run the model-based baseline:

```bash
export OPENAI_API_KEY=your_key
python3 baseline.py --planner openai --model gpt-4o-mini
```

If the environment is already running elsewhere:

```bash
OPENENV_BASE_URL=http://localhost:8000 python3 baseline.py --planner scripted
```

## Running `inference.py`

For evaluator submission, the script must use the injected LiteLLM proxy credentials:

```bash
export API_KEY=your_proxy_key
export API_BASE_URL=your_proxy_base_url
export MODEL_NAME=your_model_name
python3 inference.py
```

To run the environment from a local Docker image:

```bash
export LOCAL_IMAGE_NAME=cli-auto-fixer
python3 inference.py
```

To run against an existing OpenEnv server:

```bash
export OPENENV_BASE_URL=http://localhost:8000
python3 inference.py
```

## Docker

Build the image:

```bash
docker build -t cli-auto-fixer .
```

Run the OpenEnv server:

```bash
docker run -p 8000:8000 cli-auto-fixer
```

This container is the intended runtime for the hardest task because it safely supports package installation and passwordless `sudo` inside the sandbox.

## Hugging Face / OpenEnv Push

Validate before pushing:

```bash
openenv validate .
```

Push to Hugging Face Spaces with OpenEnv:

```bash
openenv push --repo-id stackedshadow/cli-auto-fixer
```

Notes:

- the Hugging Face Space repo name should use hyphens, for example `cli-auto-fixer`
- the internal OpenEnv environment name remains `cli_auto_fixer`
- if push fails with `403 Forbidden`, the usual cause is token or namespace permissions on Hugging Face

## Expected Behavior

- The scripted baseline is intended for quick smoke tests and deterministic validation.
- The Docker runtime is important for the OS-level dependency task, because it allows the environment to install system packages safely.
- The model-based runner depends on the model following the command-only or JSON-only contract in the prompt.

## Summary

CLI Auto Fixer is a practical benchmark for agents that need to act like junior SREs instead of code autocomplete systems. It measures whether a model can debug, explore, repair, and verify in a real command-line loop.
