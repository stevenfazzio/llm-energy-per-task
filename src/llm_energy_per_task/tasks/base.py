"""What a task is: a prompt for each item and a score for each answer."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    name: str
    # What `score` measures, so a results file says what its number means.
    metric: str
    # Generous for a right answer, tight enough that a rambling one is cut off.
    max_tokens: int
    prompt: Callable[[dict], str]
    score: Callable[[dict, str], float]


_THINKING = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Drop a reasoning block, closed or cut off, so only the answer is scored."""
    return _THINKING.sub("", text).strip()
