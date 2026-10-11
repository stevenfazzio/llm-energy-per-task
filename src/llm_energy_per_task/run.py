"""Measure one model: answer every task's items on the GPU while metering it.

Runs on the pod. One model per process, because the simplest way to hand a GPU
back clean is to exit; run_all.py is the loop over models.

    uv run python -m llm_energy_per_task.run --model Qwen/Qwen2.5-0.5B-Instruct
    uv run python -m llm_energy_per_task.run --model ... --limit 20   # a smoke test, not a result
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from datetime import UTC, datetime
from statistics import fmean

from . import RESULTS
from .energy import GpuMeter, Reading
from .files import write_json, write_jsonl
from .models import Model, load_models
from .tasks import TASKS, load_items, strip_thinking

# The protocol. Changing any of these changes what the numbers mean, so they are
# written into every result.
DTYPE = "bfloat16"
# The shortest context window among the models listed; every prompt fits in it.
MAX_MODEL_LEN = 2048
# How many items the engine works on at once, and how many tokens it takes per
# step. Left unset, vLLM picks both from the card's memory; these are what it
# picks for a 48 GB card, set here so a different card cannot change them.
MAX_NUM_SEQS = 256
MAX_NUM_BATCHED_TOKENS = 8192
# A tiny model clears a task in a second or two, too short to meter. The batch
# is repeated until the window is at least this long.
MIN_WINDOW_S = 30.0
# Let the card settle after warm-up before reading its idle draw.
SETTLE_S = 2.0
IDLE_S = 5.0
WARMUP_ITEMS = 16
# Thinking is switched off wherever a template has the switch; templates
# without one ignore it. Reasoning variants would be a separate entry.
TEMPLATE_KWARGS = {"enable_thinking": False}


def timed_passes(
    generate: Callable[[], list],
    meter: GpuMeter,
    min_window_s: float,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[list, int, Reading]:
    """Run the batch under the meter, repeating it until the window is long enough.

    Returns the first pass's outputs, the number of passes, and the reading over
    all of them.
    """
    meter.start()
    try:
        began = clock()
        first = generate()
        passes = 1
        while clock() - began < min_window_s:
            generate()
            passes += 1
    except BaseException:
        meter.stop()
        raise
    return first, passes, meter.stop()


def measure(
    model: Model, task_names: list[str], limit: int | None, min_window_s: float
) -> tuple[dict, dict[str, list[dict]]]:
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    meter = GpuMeter()

    loading = time.perf_counter()
    llm = LLM(
        model=model.id,
        revision=model.revision,
        dtype=DTYPE,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        # On by default, and it would make every repeated pass nearly free.
        enable_prefix_caching=False,
        trust_remote_code=model.trust_remote_code,
        seed=0,
    )
    load_seconds = time.perf_counter() - loading

    def answer(prompts: list[str], max_tokens: int) -> list:
        return llm.chat(
            [[{"role": "user", "content": prompt}] for prompt in prompts],
            SamplingParams(temperature=0.0, max_tokens=max_tokens),
            use_tqdm=False,
            chat_template_kwargs=TEMPLATE_KWARGS,
        )

    batches = {}
    for name in task_names:
        items = load_items(name)[:limit]
        batches[name] = (items, [TASKS[name].prompt(item) for item in items])

    for name, (_, prompts) in batches.items():
        answer(prompts[:WARMUP_ITEMS], TASKS[name].max_tokens)
    time.sleep(SETTLE_S)
    idle_watts = meter.idle_watts(IDLE_S)

    measured, outputs = {}, {}
    for name, (items, prompts) in batches.items():
        task = TASKS[name]
        results, passes, reading = timed_passes(
            lambda prompts=prompts, task=task: answer(prompts, task.max_tokens), meter, min_window_s
        )

        rows = []
        for item, result in zip(items, results, strict=True):
            completion = result.outputs[0]
            rows.append(
                {
                    "id": item["id"],
                    "output": completion.text,
                    "score": task.score(item, strip_thinking(completion.text)),
                    "prompt_tokens": len(result.prompt_token_ids),
                    "output_tokens": len(completion.token_ids),
                    "finish_reason": completion.finish_reason,
                }
            )
        outputs[name] = rows
        measured[name] = {
            "metric": task.metric,
            "score": fmean(row["score"] for row in rows),
            "items": len(rows),
            "max_tokens": task.max_tokens,
            "joules_per_item": reading.joules / (passes * len(rows)),
            "energy_source": reading.source,
            "passes": passes,
            "window_seconds": reading.seconds,
            "counter_joules": reading.counter_joules,
            "sampled_joules": reading.sampled_joules,
            "mean_watts": reading.mean_watts,
            "peak_watts": reading.peak_watts,
            "prompt_tokens_per_item": fmean(row["prompt_tokens"] for row in rows),
            "output_tokens_per_item": fmean(row["output_tokens"] for row in rows),
            "truncated": fmean(row["finish_reason"] == "length" for row in rows),
            # Should be zero with prefix caching off; recorded so a nonzero value is seen.
            "cached_prompt_tokens": sum(result.num_cached_tokens or 0 for result in results),
        }
        print(
            f"  {name}: {task.metric} {measured[name]['score']:.3f}, "
            f"{measured[name]['joules_per_item']:.3f} J/item "
            f"({passes} passes, {reading.seconds:.0f} s, {reading.mean_watts:.0f} W)",
            flush=True,
        )

    record = {
        "model": model.id,
        "revision": model.revision,
        "released": model.released,
        "params": model.params,
        "measured": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": {
            **meter.describe(),
            "vllm": vllm.__version__,
            "torch": torch.__version__,
            "dtype": DTYPE,
            "max_model_len": MAX_MODEL_LEN,
            "max_num_seqs": MAX_NUM_SEQS,
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "prefix_caching": False,
            "min_window_s": min_window_s,
        },
        "load_seconds": load_seconds,
        "idle_watts": idle_watts,
        "tasks": measured,
    }
    return record, outputs


def results_root(limit: int | None):
    """Smoke runs go somewhere they cannot be mistaken for measurements."""
    return RESULTS / "smoke" if limit else RESULTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="a model id from models.toml")
    parser.add_argument("--tasks", default=",".join(TASKS), help="comma-separated task names")
    parser.add_argument("--limit", type=int, help="use only the first N items (smoke test)")
    parser.add_argument("--min-window", type=float, default=MIN_WINDOW_S)
    args = parser.parse_args()

    model = next((m for m in load_models() if m.id == args.model), None)
    if model is None:
        parser.error(f"{args.model} is not in models.toml")
    task_names = args.tasks.split(",")
    for name in task_names:
        if name not in TASKS:
            parser.error(f"unknown task {name!r}; choose from {list(TASKS)}")

    print(f"{model.id} ({model.released})", flush=True)
    record, outputs = measure(model, task_names, args.limit, args.min_window)

    # Raw answers first: the run file is what marks a model as done, so it must
    # never exist without the answers it was scored from.
    root = results_root(args.limit)
    for name, rows in outputs.items():
        write_jsonl(root / "outputs" / model.slug / f"{name}.jsonl", rows)
    write_json(root / "runs" / f"{model.slug}.json", record)


if __name__ == "__main__":
    main()
