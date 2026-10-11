"""What the card draws when nothing is asked of it.

Run once on the pod, after the measurements: a property of the card and the
engine, not of any model. It traces the card with no engine on it, then with
the engine loaded and quiet after the same warm-up every measurement begins
with. That shows three things: the empty card, the loaded card once settled,
and what `idle_watts` in each run file is catching, which is read too soon
after the warm-up to be a settled idle.

It cannot show what the card draws in the gaps inside a task, which are far
shorter than a power reading.

    uv run python -m llm_energy_per_task.calibrate
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

from . import RESULTS, run
from .energy import GpuMeter
from .files import write_json
from .models import Model, load_models
from .tasks import TASKS, load_items

# The small pilot model: the loaded card is the same whichever model it holds,
# since the engine reserves the same memory each time, and this one loads fastest.
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
# How much of the end of a trace is taken as settled.
SETTLED_S = 20.0

Trace = list[tuple[float, float, float | None]]


def mean_watts(rows: Trace, start: float, end: float) -> float | None:
    """Mean draw between two moments of a trace, from the counter where there is one."""
    span = [row for row in rows if start <= row[0] <= end]
    if len(span) < 2:
        return None
    (began, _, counted_from), (ended, _, counted_to) = span[0], span[-1]
    if counted_from is not None and counted_to is not None and counted_to >= counted_from:
        return (counted_to - counted_from) / (ended - began)
    sampled = sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(span, span[1:], strict=False))
    return sampled / (ended - began)


def settled(rows: Trace) -> float | None:
    return mean_watts(rows, rows[-1][0] - SETTLED_S, rows[-1][0])


def calibrate(model: Model, empty_s: float, quiet_s: float) -> dict:
    import torch
    import vllm

    meter = GpuMeter()
    empty = meter.trace(empty_s)

    answer = run.load_engine(model)
    for name, task in TASKS.items():
        prompts = [task.prompt(item) for item in load_items(name)[: run.WARMUP_ITEMS]]
        answer(prompts, task.max_tokens)
    quiet = meter.trace(quiet_s)

    def rounded(rows: Trace) -> list[list]:
        return [
            [round(at, 3), round(watts, 2), None if counted is None else round(counted, 3)]
            for at, watts, counted in rows
        ]

    return {
        "model": model.id,
        "measured": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": {**meter.describe(), "vllm": vllm.__version__, "torch": torch.__version__},
        "empty_watts": settled(empty),
        "loaded_watts": settled(quiet),
        # The stretch each run records as idle_watts.
        "idle_watts_as_recorded": mean_watts(quiet, run.SETTLE_S, run.SETTLE_S + run.IDLE_S),
        "columns": ["seconds", "watts", "counter_joules"],
        "empty_trace": rounded(empty),
        "loaded_trace": rounded(quiet),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=MODEL, help="a model id from models.toml")
    parser.add_argument("--empty-seconds", type=float, default=120.0)
    parser.add_argument("--quiet-seconds", type=float, default=120.0)
    args = parser.parse_args()

    model = next((m for m in load_models() if m.id == args.model), None)
    if model is None:
        parser.error(f"{args.model} is not in models.toml")

    record = calibrate(model, args.empty_seconds, args.quiet_seconds)
    write_json(RESULTS / "calibration.json", record)
    for key in ("empty_watts", "loaded_watts", "idle_watts_as_recorded"):
        print(f"{key}: {record[key]:.1f} W" if record[key] is not None else f"{key}: not read")


if __name__ == "__main__":
    main()
