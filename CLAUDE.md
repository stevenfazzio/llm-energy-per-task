# llm-energy-per-task

Measures the GPU energy small open-weight LLMs need per task, to chart the least
energy that reaches a fixed score over time. The README has the method and its
caveats; this file has what a session needs that the README does not.

## State (2026-10-11)

The pilot has run. Four models are measured on one L40S and sit in
`results/runs/`; the other 77 are not. The 0.5B model was measured three times
for noise, and the two extra runs are in `results/repeat/`, which `frontier.py`
does not read. The pilot rows were measured under the protocol now in `run.py`,
so they stand unless it changes.

What the pilot confirmed:

- vLLM 0.31.0 loads all four with the arguments `run.py` passes, including
  Mistral-7B-Instruct-v0.3 through the Mistral path described below.
- The energy counter exists on the L40S and agrees with sampled power to within
  2% on every task. No prompt tokens were served from cache.
- Energy repeats: `joules_per_item` spread by under 1% across three fresh runs.
  Scores do not quite: tool-call accuracy ran 0.805 to 0.815, because greedy
  decoding in a batch is not exactly repeatable. A bar needs that much margin.
- The card is busy whatever the model's size: mean draw was 305 to 350 W for all
  four, so the 0.5B model's figure is not mostly idling.
- A model takes 2 to 3.5 minutes end to end. The pod answered SSH 30 seconds
  after creation and the environment installed in about 2 minutes.
- `scripts/pod_guard.sh` works: the pilot pod terminated itself with it.

What it left open, for Steven to decide:

- **Score bars.** No pilot model reaches 0.7 on intent (best 0.556), while the
  0.5B model already scores 0.815 on tool calls. The bars have to be per task;
  `frontier.py` still defaults to 0.7/0.8/0.9 for all three.
- **Extraction rewards short answers more than right ones.** Both Mistrals
  answer in sentences (a median of 11 to 15 words against 2 in the references)
  and score below the 0.5B model: 0.366 and 0.474 against 0.527. v0.2 hits the
  48-token cap on 17% of items. The answers are kept, so trying another scorer
  needs no GPU.
- **`idle_watts` is probably not a settled idle.** It read 108 to 113 W, but the
  same card drew 35 W before anything loaded and 56 W just after a run exited.
  It is read 2 seconds after warm-up. It does not enter `joules_per_item`.

What reading vLLM 0.31.0 showed, since it shapes the code:

- The engine runs in a child process of `run.py`, so `run_all.py` kills the
  whole process group. That the child would outlive a timeout was inferred from
  finding no parent-death handling; no model has timed out yet.
- Batch limits default from the card's memory, so `run.py` sets them.
- A repo holding `consolidated*.safetensors` and a Mistral tokenizer file is
  loaded through `mistral_common`, which ignores the repo's chat template and
  `enable_thinking`. That is Mistral-7B-Instruct-v0.3, Mistral-Nemo and the
  three Ministrals; v0.1 and v0.2 take the ordinary path.

## Budget

Steven has set **$20 total** of RunPod spend for this project. Before anything
billable: state the hourly price and the estimate, and wait for a yes. Give every
pod a guard that terminates it. The account carries other projects' spend, so
count by pod, not by account total.

Spent so far: **about $0.37**. The pilot pod (`rm2ki4tybbbtyc`, secure cloud,
$1.09/hr) lived 20 minutes on 2026-10-11. That is price times lifetime; the
billing API had not posted the charge when this was written.

The guard is two timers, because nothing RunPod offers does it: `runpodctl pod
create` has no terminate-after flag through 2.15.0 (the RunPod plugin's docs say
it does), and neither has the MCP `create-pod`. The API's GraphQL input does
define `terminateAfter`; that is untried. So `scripts/pod_guard.sh` runs on the
pod and a `sleep && runpodctl pod delete` runs here. Terminating deletes the
pod's disk, so on a long run copy `results/` back as it goes.

## Running on a pod

The commands are in the README under "Running it", and all of them have now run
once. What they do not say:

- Template `runpod-torch-v280` works: it has `rsync`, `uv` and `runpodctl`
  1.14.15. That `runpodctl` gets its key from PID 1's environment, which
  `pod_guard.sh` reads. The key can terminate its own pod but cannot list pods.
- Steven's SSH key has a passphrase and works through the agent. If the agent
  does not hold it, SSH fails without prompting.
- Start the run with `cd DIR; setsid ... &`, not `cd DIR && setsid ... &`. The
  second form leaves a shell holding the SSH channel and `ssh` never returns.
- `run_all.py` sets `HF_HUB_DISABLE_SHARED_BLOBS`. Without it huggingface_hub
  1.33 keeps weights in a store shared across repos, deleting a model's folder
  frees nothing, and four models left 34 GB behind.
- On 2026-10-11 an L40S was $0.79/hr community and $1.09/hr secure, on hosts
  reporting CUDA 13.0 or 13.2. Community had none free above the CUDA floor.

## Next step: the full run

80 models can load (`microsoft/phi-2` publishes no chat template), 66 of them
without a token. At the pilot's pace, about 2.1 minutes plus 0.2 per billion
parameters, all 80 take about 4.2 hours: roughly $3.30 at $0.79/hr or $4.50 at
$1.09/hr. That is an extrapolation from four models, none above 7B. 60 GB of
volume was ample; the largest download is about 30 GB.

14 models are gated (every Llama, Gemma 1 through 3). They are skipped unless
`HF_TOKEN` is set on the pod and the licences are accepted on that account; ask
Steven how he wants to handle that.

Settle the open questions above first where they touch `run.py`: changing the
idle reading there means measuring the four pilot models again.

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
