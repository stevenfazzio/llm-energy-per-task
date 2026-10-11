"""GPU energy that small open-weight LLMs need per task, at measured accuracy."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASK_DATA = ROOT / "data" / "tasks"
RESULTS = ROOT / "results"
