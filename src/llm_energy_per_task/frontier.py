"""From measurements to the chart's lines.

For a task and a score bar, the line is the least energy per item among models
released to date that score at least the bar: a running minimum, stepping down
only when something more frugal clears it. Fixing the score is what makes
energy comparable over time, since the most frugal model is always a weak one.

Which bars, and what counts as clearing one or as a step, were settled before
the models were measured, so that the result could not choose them:

- frontier.csv has the line at every bar from 0.05 to 0.95.
- Two bars a task are marked for the chart: the highest that some model released
  in 2023 clears, and the highest that at least three models clear.
- A model clears a bar when its score is at or above it. A step by a model whose
  score interval reaches below the bar is marked as straddling it. A model just
  under a bar, whose interval reaches it and which would have been a step, is
  recorded as a near miss and left off the line.
- A new low is a step only when it is 3% or more below the last one. Energy
  repeats to within 1%.

    uv run python -m llm_energy_per_task.frontier
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math

from . import RESULTS
from .files import write_atomic
from .models import load_models

GRID = [n / 20 for n in range(1, 20)]
# The first year with models in the dataset, and so where a line can start.
BASE_YEAR = "2023"
# A bar only one or two models clear says more about them than about a trend.
MIN_CLEARING = 3
# A new low has to be under this share of the last step to count as one.
STEP = 0.97

COLUMNS = [
    "task",
    "model",
    "released",
    "params",
    "metric",
    "items",
    "score",
    "score_low",
    "score_high",
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


def interval(score: float, items: int, z: float = 1.96) -> tuple[float, float]:
    """The 95% Wilson interval for a share of `items` answers that were right."""
    scale = 1 + z * z / items
    middle = (score + z * z / (2 * items)) / scale
    half = z * math.sqrt(score * (1 - score) / items + z * z / (4 * items * items)) / scale
    return middle - half, middle + half


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
            low, high = interval(task["score"], task["items"])
            rows.append(
                {
                    "task": name,
                    "model": run["model"],
                    "released": released.get(run["model"], run["released"]),
                    "params": run["params"],
                    "metric": task["metric"],
                    "items": task["items"],
                    "score": task["score"],
                    "score_low": low,
                    "score_high": high,
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


def line(rows: list[dict], bar: float) -> list[tuple[str, dict]]:
    """One task's line at `bar`, in order of release, with the near misses beside it.

    Each row comes with its kind: "step", "step_straddling" for a step whose
    score interval reaches below the bar, or "near_miss" for a model under the
    bar whose interval reaches it and which would otherwise have been a step.
    A near miss does not move the line.
    """
    marked = []
    best = float("inf")
    for row in sorted(rows, key=lambda r: (r["released"], r["joules_per_item"])):
        if row["joules_per_item"] >= best * STEP:
            continue
        if row["score"] >= bar:
            best = row["joules_per_item"]
            marked.append(("step" if row["score_low"] >= bar else "step_straddling", row))
        elif row["score_high"] >= bar:
            marked.append(("near_miss", row))
    return marked


def chart_bars(rows: list[dict]) -> dict[float, str]:
    """The bars of one task that a chart shows, each with what picked it."""
    base_year = [row["score"] for row in rows if row["released"].startswith(BASE_YEAR)]
    picks = {
        BASE_YEAR: [bar for bar in GRID if any(score >= bar for score in base_year)],
        "three_models": [
            bar for bar in GRID if sum(row["score"] >= bar for row in rows) >= MIN_CLEARING
        ],
    }
    bars: dict[float, str] = {}
    for why, cleared in picks.items():
        if cleared:
            bars[max(cleared)] = f"{bars[max(cleared)]}+{why}" if max(cleared) in bars else why
    return bars


def _csv(rows: list[dict], columns: list[str]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()

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

    marked = []
    for task in sorted({row["task"] for row in rows}):
        of_task = [row for row in rows if row["task"] == task]
        shown = chart_bars(of_task)
        for bar in GRID:
            at_bar = line(of_task, bar)
            head = {"bar": f"{bar:.2f}", "chart": shown.get(bar, "")}
            marked += [{**head, "kind": kind, **row} for kind, row in at_bar]
            # Only the bars a chart shows are printed; frontier.csv has the rest.
            if bar in shown:
                print(f"\n{task}, at least {bar:.2f} ({shown[bar]})")
                for kind, row in at_bar:
                    print(
                        f"  {row['released']:10s} {row['model']:42s} {row['score']:.3f} "
                        f"[{row['score_low']:.3f}, {row['score_high']:.3f}] "
                        f"{row['joules_per_item']:8.3f} J  {kind}"
                    )
    write_atomic(RESULTS / "frontier.csv", _csv(marked, ["bar", "chart", "kind", *COLUMNS]))


if __name__ == "__main__":
    main()
