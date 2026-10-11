"""Measure a set of models, one process each, smallest first.

Resumable: a model with a run file is skipped, so a pod that dies partway costs
only the model it was on. A model that fails is recorded and passed over rather
than ending the run, and stays passed over until --retry-failed.

    uv run python -m llm_energy_per_task.run_all --set pilot
    uv run python -m llm_energy_per_task.run_all --set all --max-hours 8
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from .files import write_atomic
from .models import Model, load_models
from .run import results_root

# What fits on one 48 GB card at 16 bits per weight, with room to work.
MAX_PARAMS = 15_000_000_000


def weights_dir(model: Model) -> Path:
    home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    return home / "hub" / f"models--{model.id.replace('/', '--')}"


def run_model(command: list[str], log: Path, timeout_s: float) -> int | str:
    """Run one measurement in a process group of its own and leave none of it behind.

    vLLM serves from a child process of its own. Killing only run.py on a timeout
    could leave that child holding the GPU, and every model after it would fail
    to load.
    """
    # Without this, huggingface_hub keeps weights in a store shared by every repo,
    # where deleting weights_dir() does not reach them and the disk fills.
    env = {**os.environ, "HF_HUB_DISABLE_SHARED_BLOBS": "1"}
    with log.open("w") as stream:
        process = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, env=env
        )
        try:
            return process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return "timeout"
        finally:
            # After a clean exit the group is already empty and there is nothing to find.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def skip_reason(model: Model, root: Path, max_params: int, retry_failed: bool) -> str | None:
    if (root / "runs" / f"{model.slug}.json").exists():
        return "already measured"
    if (root / "failed" / f"{model.slug}.txt").exists() and not retry_failed:
        return "failed before (--retry-failed to try again)"
    if model.params and model.params > max_params:
        return f"over {max_params / 1e9:g}B parameters"
    if not model.chat_template:
        return "no chat template published"
    if model.gated and not os.environ.get("HF_TOKEN"):
        return "gated, and HF_TOKEN is not set"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--set", choices=["pilot", "all"])
    choice.add_argument("--models", help="comma-separated model ids")
    parser.add_argument("--max-params", type=float, default=MAX_PARAMS)
    parser.add_argument("--max-hours", type=float, help="start no new model after this long")
    parser.add_argument("--timeout-min", type=float, default=40, help="per model")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--keep-weights", action="store_true", help="leave downloads in place")
    parser.add_argument("--limit", type=int, help="passed to run.py: smoke test on N items")
    args = parser.parse_args()

    models = load_models()
    if args.models:
        wanted = args.models.split(",")
        unknown = sorted(set(wanted) - {m.id for m in models})
        if unknown:
            parser.error(f"not in models.toml: {unknown}")
        models = [m for m in models if m.id in wanted]
    elif args.set == "pilot":
        models = [m for m in models if m.pilot]
    # Smallest first: if the budget runs out, it runs out on the costly ones.
    models.sort(key=lambda m: (m.params is None, m.params or 0))

    root = results_root(args.limit)
    todo = []
    for model in models:
        reason = skip_reason(model, root, int(args.max_params), args.retry_failed)
        if reason:
            print(f"skip {model.id}: {reason}")
        else:
            todo.append(model)
    print(f"{len(todo)} to measure, {len(models) - len(todo)} skipped\n", flush=True)

    began = time.monotonic()
    failed = []
    for n, model in enumerate(todo, start=1):
        hours = (time.monotonic() - began) / 3600
        if args.max_hours and hours >= args.max_hours:
            print(f"stopping at {hours:.1f} h with {len(todo) - n + 1} models unmeasured")
            break

        command = [sys.executable, "-m", "llm_energy_per_task.run", "--model", model.id]
        if args.limit:
            command += ["--limit", str(args.limit)]
        log = root / "logs" / f"{model.slug}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        code = run_model(command, log, args.timeout_min * 60)
        minutes = (time.monotonic() - started) / 60

        if code == 0:
            (root / "failed" / f"{model.slug}.txt").unlink(missing_ok=True)
            outcome = "ok"
        else:
            failed.append(model.id)
            tail = "\n".join(log.read_text().splitlines()[-40:])
            write_atomic(root / "failed" / f"{model.slug}.txt", f"exit {code}\n\n{tail}\n")
            outcome = f"FAILED ({code}); see {log}"
        if not args.keep_weights:
            shutil.rmtree(weights_dir(model), ignore_errors=True)
        total = (time.monotonic() - began) / 3600
        print(f"[{n}/{len(todo)}] {model.id}: {outcome}, {minutes:.1f} min ({total:.2f} h so far)")
        sys.stdout.flush()

    if failed:
        print(f"\n{len(failed)} failed: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
