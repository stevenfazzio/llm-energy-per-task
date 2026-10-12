"""Score the stored answers again, after a scorer changes.

Every raw answer is kept in results/outputs, so changing how an answer is
scored needs no GPU. This works each answer's scores out again and rewrites
them, there and in the run files. Energy, tokens and everything else that was
measured are left exactly as they were.

    uv run python -m llm_energy_per_task.rescore --dry-run   # say what would change
    uv run python -m llm_energy_per_task.rescore
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import RESULTS
from .files import read_jsonl, write_json, write_jsonl
from .run import answer_row, task_scores
from .tasks import TASKS, load_items

# What an answer's row holds besides its scores: the text and what was measured.
NOT_SCORES = ("prompt_tokens", "output_tokens", "finish_reason")


def rescore(root: Path, write: bool = True) -> list[tuple[str, str, str, float, float]]:
    """Rescore every run under `root`.

    Returns (model, task, metric, score before, score after) for each score that moved.
    """
    items = {name: {item["id"]: item for item in load_items(name)} for name in TASKS}
    moved = []
    for path in sorted((root / "runs").glob("*.json")):
        run = json.loads(path.read_text())
        for name, entry in run["tasks"].items():
            task = TASKS[name]
            answers = root / "outputs" / path.stem / f"{name}.jsonl"
            before_rows = read_jsonl(answers)
            if len(before_rows) != entry["items"]:
                raise ValueError(f"{answers}: {len(before_rows)} answers, not {entry['items']}")
            rows = [
                answer_row(
                    task, items[name][row["id"]], row["output"], **{k: row[k] for k in NOT_SCORES}
                )
                for row in before_rows
            ]
            scores = task_scores(task, rows)
            before = entry["score"]
            if (entry["metric"], before) != (scores["metric"], scores["score"]):
                moved.append((run["model"], name, scores["metric"], before, scores["score"]))
            entry.update(scores)
            # Answers first, as when measuring: a run file never describes
            # answers that are not there.
            if write and rows != before_rows:
                write_jsonl(answers, rows)
        if write:
            write_json(path, run)
    return moved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="report, and write nothing")
    args = parser.parse_args()

    moved = rescore(RESULTS, write=not args.dry_run)
    for model, name, metric, before, after in moved:
        print(f"{name:11s} {model:44s} {before:.4f} -> {after:.4f}  ({metric})")
    print(f"{len(moved)} scores {'would move' if args.dry_run else 'moved'}")


if __name__ == "__main__":
    main()
