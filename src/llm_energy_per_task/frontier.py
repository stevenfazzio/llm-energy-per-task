"""From measurements to the chart's lines.

For a task and a score bar, the line is the least energy per item among models
released to date that score at least the bar: a running minimum, stepping down
only when something more frugal clears it. Fixing the score is what makes
energy comparable over time, since the most frugal model is always a weak one.

    uv run python -m llm_energy_per_task.frontier --bars 0.7,0.8,0.9
"""

from __future__ import annotations

import argparse
import csv
import io
import json

from . import RESULTS
from .files import write_atomic
from .models import load_models

COLUMNS = [
    "task",
    "model",
    "released",
    "params",
    "metric",
    "score",
    "token_f1",
    "first_line",
    "joules_per_item",
    "wh_per_1000_items",
    "prompt_tokens_per_item",
    "output_tokens_per_item",
    "mean_watts",
    "joules_per_billion_params_per_1000_tokens",
    "energy_source",
    "gpu",
]


def per_unit_of_work(joules_per_item: float, params: int | None, tokens: float) -> float | None:
    """Energy per item with model size and tokens processed divided out.

    Roughly level across models when energy tracks the work done. A model well
    above the rest is paying for something that is not the model, which is what
    happens at the small end: the card and the serving stack waiting on it.
    """
    if not params:
        return None
    return joules_per_item / (params / 1e9) / (tokens / 1000)


def summary() -> list[dict]:
    """One row per model and task, from every run file of a model in the dataset."""
    # The registry's date wins over the one stamped at measurement, so a release
    # date corrected in models.toml takes effect without measuring again.
    models = load_models()
    released = {m.id: m.released for m in models}
    excluded = {m.id for m in models if m.excluded}
    rows = []
    for path in sorted((RESULTS / "runs").glob("*.json")):
        run = json.loads(path.read_text())
        if run["model"] in excluded:
            continue
        for name, task in run["tasks"].items():
            tokens = task["prompt_tokens_per_item"] + task["output_tokens_per_item"]
            rows.append(
                {
                    "task": name,
                    "model": run["model"],
                    "released": released.get(run["model"], run["released"]),
                    "params": run["params"],
                    "metric": task["metric"],
                    "score": task["score"],
                    # Each kept beside one task's score, and blank for the others:
                    # SQuAD's metric for extraction, the stricter match for intent.
                    "token_f1": task.get("also", {}).get("token_f1"),
                    "first_line": task.get("also", {}).get("first_line"),
                    "joules_per_item": task["joules_per_item"],
                    "wh_per_1000_items": task["joules_per_item"] / 3.6,
                    "prompt_tokens_per_item": task["prompt_tokens_per_item"],
                    "output_tokens_per_item": task["output_tokens_per_item"],
                    "mean_watts": task["mean_watts"],
                    "joules_per_billion_params_per_1000_tokens": per_unit_of_work(
                        task["joules_per_item"], run["params"], tokens
                    ),
                    "energy_source": task["energy_source"],
                    "gpu": run["environment"]["gpu"],
                }
            )
    return sorted(rows, key=lambda r: (r["task"], r["released"], r["joules_per_item"]))


def frontier(rows: list[dict], bar: float) -> list[dict]:
    """The rows of one task that set a new low in energy at or above `bar`."""
    steps = []
    best = float("inf")
    for row in sorted(rows, key=lambda r: (r["released"], r["joules_per_item"])):
        if row["score"] >= bar and row["joules_per_item"] < best:
            best = row["joules_per_item"]
            steps.append(row)
    return steps


def _csv(rows: list[dict], columns: list[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bars", default="0.7,0.8,0.9", help="comma-separated score bars")
    args = parser.parse_args()
    bars = [float(bar) for bar in args.bars.split(",")]

    rows = summary()
    if not rows:
        raise SystemExit("no run files in results/runs yet")
    for model in load_models():
        if model.excluded:
            print(f"left out: {model.id} ({model.excluded})")
    gpus = sorted({row["gpu"] for row in rows})
    if len(gpus) > 1:
        print(f"WARNING: runs span {gpus}; energy is only comparable on one card\n")

    print(f"{'task':11s} {'released':10s} {'model':42s} {'score':>6s} {'J/item':>9s}")
    for row in rows:
        print(
            f"{row['task']:11s} {row['released']:10s} {row['model']:42s} "
            f"{row['score']:6.3f} {row['joules_per_item']:9.3f}"
        )
    write_atomic(RESULTS / "summary.csv", _csv(rows, COLUMNS))

    steps = []
    for task in sorted({row["task"] for row in rows}):
        of_task = [row for row in rows if row["task"] == task]
        for bar in bars:
            for row in frontier(of_task, bar):
                steps.append({"bar": bar, **row})
    print(f"\n{'task':11s} {'bar':>5s} {'released':10s} {'model':42s} {'J/item':>9s}")
    for step in steps:
        print(
            f"{step['task']:11s} {step['bar']:5.2f} {step['released']:10s} {step['model']:42s} "
            f"{step['joules_per_item']:9.3f}"
        )
    write_atomic(RESULTS / "frontier.csv", _csv(steps, ["bar", *COLUMNS]))


if __name__ == "__main__":
    main()
