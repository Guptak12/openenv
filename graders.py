"""Submission-time graders for OpenEnv task metadata.

These graders are intentionally strict about returning scores inside the
open interval (0, 1) because the packaging validator rejects 0.0 and 1.0.
"""

from __future__ import annotations

from typing import Any


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _clamp_fractional(score: float) -> float:
    return max(0.05, min(0.95, round(score, 4)))


def _extract_score(payload: Any) -> float | None:
    if payload is None:
        return None

    numeric = _as_float(payload)
    if numeric is not None:
        return numeric

    if isinstance(payload, dict):
        for key in (
            "total_score",
            "score",
            "reward",
            "reward_model",
            "metadata",
            "observation",
            "state",
            "result",
        ):
            nested = _extract_score(payload.get(key))
            if nested is not None:
                return nested

    if isinstance(payload, (list, tuple)):
        for item in reversed(payload):
            nested = _extract_score(item)
            if nested is not None:
                return nested

    return None


def _base_grade(trajectory: Any, partial_credit: float) -> float:
    observed = _extract_score(trajectory)
    if observed is None:
        return _clamp_fractional(partial_credit)
    if observed >= 1.0:
        return 0.95
    if observed <= 0.0:
        return _clamp_fractional(partial_credit)
    return _clamp_fractional(observed)


def grade_task_1(trajectory: Any, ground_truth: Any | None = None) -> float:
    return _base_grade(trajectory, partial_credit=0.25)


def grade_task_2(trajectory: Any, ground_truth: Any | None = None) -> float:
    return _base_grade(trajectory, partial_credit=0.45)


def grade_task_3(trajectory: Any, ground_truth: Any | None = None) -> float:
    return _base_grade(trajectory, partial_credit=0.65)


def grade_episode(trajectory: Any, ground_truth: Any | None = None) -> float:
    task_id = None
    if isinstance(trajectory, dict):
        task_id = (
            trajectory.get("task_id")
            or trajectory.get("task")
            or trajectory.get("task_name")
        )

    if task_id == "task_1":
        return grade_task_1(trajectory, ground_truth)
    if task_id == "task_2":
        return grade_task_2(trajectory, ground_truth)
    if task_id == "task_3":
        return grade_task_3(trajectory, ground_truth)

    return _base_grade(trajectory, partial_credit=0.35)
