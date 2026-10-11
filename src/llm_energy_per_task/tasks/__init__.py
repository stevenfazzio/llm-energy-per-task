"""The tasks: what each one asks, and how an answer is scored.

Every model sees the same frozen items (data/tasks/*.jsonl), each as a single
user turn. A single turn because several chat templates reject a system role,
and one prompt shape for every model matters more than using each one's
features.
"""

from __future__ import annotations

from .. import TASK_DATA
from ..files import read_jsonl
from . import extraction, intent, toolcall
from .base import Task, strip_thinking

TASKS: dict[str, Task] = {task.name: task for task in (intent.TASK, extraction.TASK, toolcall.TASK)}


def load_items(name: str) -> list[dict]:
    path = TASK_DATA / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run: uv run python -m llm_energy_per_task.tasks.prepare"
        )
    return read_jsonl(path)


__all__ = ["TASKS", "Task", "load_items", "strip_thinking"]
