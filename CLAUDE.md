# llm-energy-per-task

Measures the GPU energy small open-weight LLMs need per task, to chart the least
energy that reaches a fixed score over time. The README has the method and its
caveats; this file has what a session needs that the README does not.

## State (2026-10-11)

Nothing has been measured. The harness, frozen task items and model list exist
and pass their tests, but the GPU path has only run against fakes
(`tests/fakes.py`, `tests/test_run.py`). The vLLM arguments in `run.py` were
checked against the 0.31 docs and the NVML calls against the library source;
neither has touched real hardware.

## Budget

Steven has set **$20 total** of RunPod spend for this project. Before anything
billable: state the hourly price and the estimate, and wait for a yes. Give every
pod a terminate-after guard. Track spend so far here once there is any.

## Next step: the pilot

Three ungated models, flagged `pilot = true` in `models.toml`:
Mistral-7B-Instruct-v0.2, Qwen2.5-3B-Instruct, Qwen2.5-0.5B-Instruct.

Pod: one L40S (48 GB), host CUDA 12.9 or newer (what the pinned vLLM wheels are
built for), about 60 GB of disk. On 2026-10-11 an L40S was $0.79/hr community,
$1.09/hr secure. Then `bash scripts/pod_run.sh pilot` and copy `results/` back.

The pilot exists to confirm these before the full run, so check each one:

1. vLLM loads all three with the arguments `run.py` passes.
2. `environment.energy_counter` is true, and `counter_joules` agrees with
   `sampled_joules` to within a few percent.
3. `cached_prompt_tokens` is 0 for every task.
4. `truncated` is near zero and `results/outputs/` reads like answers, not
   preamble or reasoning.
5. Minutes per model, end to end. Re-derive the full-run cost from this; the
   current "10 minutes each" is a guess.
6. Scores spread sensibly across the three sizes. The score bars for the chart
   get chosen from this data; `frontier.py` defaults to 0.7/0.8/0.9 as
   placeholders.

## The full run

81 models, so about $11-15 at the guessed pace: inside the budget with little
slack. If the pilot shows it will not fit, prune rather than overspend. Only
models that could set a record matter, meaning ones smaller than whatever
already clears a bar at their release date.

14 models are gated (every Llama, Gemma 1 through 3). They are skipped unless
`HF_TOKEN` is set on the pod and the licences are accepted on that account; ask
Steven how he wants to handle that.

## Rules that protect the measurements

- **Do not change the protocol constants in `run.py` or the items in
  `data/tasks/` once real results exist.** Either one makes old and new rows
  incomparable. If one must change, every model is measured again.
- **One GPU model for every run.** Energy on different cards is not comparable;
  `frontier.py` warns if runs span more than one.
- **Fix scoring bugs by rescoring, not rerunning.** `results/outputs/` holds
  every raw answer for that reason. There is no rescore command yet; write one
  when it is first needed.
- Smoke runs (`--limit N`) write to `results/smoke/`, which is ignored by git.

## Where this is going

`ai-trends` (`~/repos/ai-trends`) will plot the result as a chart, ingesting a
CSV published from this repo like any other upstream. That site's rule is that
it plots other people's published numbers, so this repo is the publisher and
needs to stand on its own: method, caveats and data all here.

Still open: the chart itself, the score bars, a licence file (the SQuAD sample
is CC BY-SA), whether to publish on GitHub, and a summarisation task (left out
because a fixed quality bar needs a judge model or a weak overlap metric).

## Commands

```sh
uv sync
uv run pytest
uv run ruff check . && uv run ruff format .
```
