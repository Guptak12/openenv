"""Core environment logic for the CLI Auto Fixer OpenEnv task suite."""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import tempfile
import venv
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from .models import CliAutoFixerAction, CliAutoFixerObservation, CliAutoFixerReward
except ImportError:
    from models import CliAutoFixerAction, CliAutoFixerObservation, CliAutoFixerReward

MAX_LOG_LINES = 200
COMMAND_TIMEOUT_SECONDS = 30
PWD_MARKER = "__OPENENV_CWD__:"


@dataclass(frozen=True)
class TaskSpec:
    """Deterministic task definition."""

    task_id: str
    name: str
    template_dir: Path
    grader_command: str
    goal: str
    difficulty: str


class CliAutoFixerEnvironment(Environment):
    """An OpenEnv environment that simulates fixing broken CLI projects."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self) -> None:
        self._root_dir = Path(__file__).resolve().parent
        self._tasks = self._load_tasks()
        self._task_index = 0
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._episode_dir: Path | None = None
        self._cwd: Path | None = None
        self._task: TaskSpec | None = None
        self._total_score = 0.0
        self._last_command = ""
        self._last_failed_command = ""
        self._last_exit_code = 0
        self._install_reward_granted = False

    def reset(self) -> CliAutoFixerObservation:
        """Reset the environment into the next deterministic task."""
        self._cleanup_episode_dir()

        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._total_score = 0.0
        self._last_command = ""
        self._last_failed_command = ""
        self._last_exit_code = 0
        self._install_reward_granted = False

        self._task = self._tasks[self._task_index % len(self._tasks)]
        self._task_index += 1

        self._episode_dir = Path(tempfile.mkdtemp(prefix=f"{self._task.task_id}_"))
        shutil.copytree(self._task.template_dir, self._episode_dir, dirs_exist_ok=True)
        self._prepare_task_runtime()
        self._cwd = self._episode_dir

        reward = CliAutoFixerReward(step_reward=0.0, total_score=self._total_score)
        return self._build_observation(
            stdout=(
                f"Task {self._task.task_id}: {self._task.name}\n"
                f"Goal: {self._task.goal}\n"
                f"Sandbox: {self._episode_dir}"
            ),
            stderr="",
            exit_code=0,
            done=False,
            reward=reward,
        )

    def step(self, action: CliAutoFixerAction) -> CliAutoFixerObservation:  # type: ignore[override]
        """Execute a single bash command inside the isolated episode directory."""
        if self._task is None or self._cwd is None or self._episode_dir is None:
            return self.reset()

        command = action.command.strip()
        self._state.step_count += 1

        if self._is_destructive(command):
            reward = self._apply_reward(-1.0)
            self._last_command = command
            self._last_failed_command = command
            self._last_exit_code = 137
            return self._build_observation(
                stdout="",
                stderr="Blocked destructive command.",
                exit_code=137,
                done=True,
                reward=reward,
                last_command=command,
            )

        try:
            result = self._run_shell_command(command)
            stdout, stderr, exit_code, new_cwd = result
            self._cwd = new_cwd
        except subprocess.TimeoutExpired:
            stdout = ""
            stderr = f"Command timed out after {COMMAND_TIMEOUT_SECONDS} seconds."
            exit_code = 124

        step_reward = 0.0
        if exit_code != 0:
            step_reward -= 0.1
            if command and command == self._last_failed_command:
                step_reward -= 0.2
            self._last_failed_command = command
        else:
            self._last_failed_command = ""

        if self._is_exploratory(command):
            step_reward += 0.1

        if (
            exit_code == 0
            and not self._install_reward_granted
            and self._is_successful_install(command)
        ):
            step_reward += 0.3
            self._install_reward_granted = True

        success = exit_code == 0 and self._grade_current_task()
        if success:
            reward = CliAutoFixerReward(step_reward=1.0, total_score=1.0)
            self._total_score = 1.0
            self._last_command = command
            self._last_exit_code = exit_code
            return self._build_observation(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                done=True,
                reward=reward,
                last_command=command,
            )

        reward = self._apply_reward(step_reward)
        self._last_command = command
        self._last_exit_code = exit_code
        return self._build_observation(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            done=False,
            reward=reward,
            last_command=command,
        )

    @property
    def state(self) -> State:
        """Return the OpenEnv state object."""
        return self._state

    def _load_tasks(self) -> list[TaskSpec]:
        tasks_dir = self._root_dir / "tasks"
        return [
            TaskSpec(
                task_id="task_1",
                name="Missing Python Package",
                template_dir=tasks_dir / "task_1",
                grader_command="python3 main.py",
                goal="Install the missing Python dependency and run python3 main.py.",
                difficulty="Easy",
            ),
            TaskSpec(
                task_id="task_2",
                name="Node.js Version Conflict",
                template_dir=tasks_dir / "task_2",
                grader_command="npm run build",
                goal="Resolve the dependency conflict and run npm run build.",
                difficulty="Medium",
            ),
            TaskSpec(
                task_id="task_3",
                name="OS-Level Missing Dependency",
                template_dir=tasks_dir / "task_3",
                grader_command="python3 main.py",
                goal=(
                    "Diagnose the psycopg2 build failure, install the missing OS package, "
                    "then run python3 main.py."
                ),
                difficulty="Hard",
            ),
        ]

    def _run_shell_command(self, command: str) -> tuple[str, str, int, Path]:
        current_cwd = self._cwd or self._episode_dir
        assert current_cwd is not None

        wrapped_command = (
            "set +e\n"
            f"{command}\n"
            "status=$?\n"
            f"printf '\\n{PWD_MARKER}%s\\n' \"$PWD\"\n"
            "exit $status\n"
        )
        process = subprocess.Popen(
            ["bash", "-lc", wrapped_command],
            cwd=str(current_cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=self._build_subprocess_env(),
            start_new_session=True,
        )
        try:
            stdout_raw, stderr_raw = process.communicate(timeout=COMMAND_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as exc:
            os.killpg(process.pid, signal.SIGKILL)
            stdout_raw, stderr_raw = process.communicate()
            raise subprocess.TimeoutExpired(
                process.args,
                COMMAND_TIMEOUT_SECONDS,
                output=stdout_raw,
                stderr=stderr_raw,
            ) from exc

        stdout, parsed_cwd = self._extract_cwd_marker(stdout_raw)
        stderr = self._truncate_log(stderr_raw)
        return stdout, stderr, process.returncode, parsed_cwd

    def _prepare_task_runtime(self) -> None:
        assert self._task is not None
        assert self._episode_dir is not None
        if self._task.task_id not in {"task_1", "task_3"}:
            return

        env_dir = self._episode_dir / ".venv"
        venv.EnvBuilder(with_pip=True, clear=True).create(str(env_dir))

    def _build_subprocess_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["PYTHONNOUSERSITE"] = "1"
        if self._task and self._episode_dir and self._task.task_id in {"task_1", "task_3"}:
            venv_dir = self._episode_dir / ".venv"
            bin_dir = venv_dir / "bin"
            env["VIRTUAL_ENV"] = str(venv_dir)
            env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
        return env

    def _extract_cwd_marker(self, stdout: str) -> tuple[str, Path]:
        current_cwd = self._cwd or self._episode_dir
        assert current_cwd is not None

        lines = stdout.splitlines()
        for index in range(len(lines) - 1, -1, -1):
            line = lines[index]
            if line.startswith(PWD_MARKER):
                next_cwd = Path(line[len(PWD_MARKER) :].strip())
                cleaned_stdout = "\n".join(lines[:index]).rstrip("\n")
                return self._truncate_log(cleaned_stdout), next_cwd
        return self._truncate_log(stdout), current_cwd

    def _grade_current_task(self) -> bool:
        assert self._task is not None
        assert self._cwd is not None
        try:
            _, _, exit_code, _ = self._run_shell_command(self._task.grader_command)
        except subprocess.TimeoutExpired:
            return False
        return exit_code == 0

    def _apply_reward(self, step_reward: float) -> CliAutoFixerReward:
        if self._total_score < 1.0:
            next_score = self._total_score + step_reward
            self._total_score = max(0.0, min(0.9, next_score))
        return CliAutoFixerReward(
            step_reward=round(step_reward, 4),
            total_score=round(self._total_score, 4),
        )

    def _build_observation(
        self,
        *,
        stdout: str,
        stderr: str,
        exit_code: int,
        done: bool,
        reward: CliAutoFixerReward,
        last_command: str | None = None,
    ) -> CliAutoFixerObservation:
        command = self._last_command if last_command is None else last_command
        cwd = str(self._cwd or self._episode_dir or self._root_dir)
        metadata: dict[str, Any] = {
            "task_id": self._task.task_id if self._task else "",
            "task_name": self._task.name if self._task else "",
            "difficulty": self._task.difficulty if self._task else "",
            "goal": self._task.goal if self._task else "",
            "last_action_error": self._truncate_log(stderr) if stderr else None,
            "reward_model": reward.model_dump(),
        }
        return CliAutoFixerObservation(
            cwd=cwd,
            last_command=command,
            stdout=self._truncate_log(stdout),
            stderr=self._truncate_log(stderr),
            exit_code=exit_code,
            done=done,
            reward=reward.step_reward,
            metadata=metadata,
        )

    def _truncate_log(self, content: str) -> str:
        lines = content.splitlines()
        if len(lines) <= MAX_LOG_LINES:
            return content.strip("\n")
        return "\n".join(lines[-MAX_LOG_LINES:]).strip("\n")

    def _is_exploratory(self, command: str) -> bool:
        return bool(
            re.match(r"^\s*(ls|cat|pwd|find|grep|rg|sed|head|tail|npm view|pip show)\b", command)
        )

    def _is_successful_install(self, command: str) -> bool:
        return bool(
            re.search(
                r"\b(pip3?\s+install|python3?\s+-m\s+pip\s+install|npm\s+install|apt(-get)?\s+install)\b",
                command,
            )
        )

    def _is_destructive(self, command: str) -> bool:
        destructive_patterns = (
            r"\brm\s+-rf\s+/(?:\s|$)",
            r"\breboot\b",
            r"\bshutdown\b",
            r"\bpoweroff\b",
            r"\bmkfs\b",
            r":\(\)\s*\{\s*:\|:\s*&\s*\};:",
        )
        return any(re.search(pattern, command) for pattern in destructive_patterns)

    def _cleanup_episode_dir(self) -> None:
        if self._episode_dir and self._episode_dir.exists():
            shutil.rmtree(self._episode_dir, ignore_errors=True)
        self._episode_dir = None
        self._cwd = None

    def close(self) -> None:
        """Release temp directories when the environment is discarded."""
        self._cleanup_episode_dir()
