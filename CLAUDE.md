# llm-energy-per-task

Measures the GPU energy small open-weight LLMs need per task, to chart the least
energy that reaches a fixed score over time. The README has the method and its
caveats; this file has what a session needs that the README does not.

The question, in Steven's words: are models getting more energy efficient at
doing the same tasks at the same level of quality? Quality is held fixed and
energy is what moves. It is not "are models getting better at the task", and a
choice that lets quality drift with a model's style fails it.

## State (2026-10-12)

The dataset is measured. All 80 loadable models ran on one pod on 2026-10-11
(`hr1nmckg7ts46y`: one L40S, driver 580.126.09, vLLM 0.31.0), on 2,000 intent
items, 2,000 extraction items and the 400 tool calls. None failed and none
timed out. `results/runs/` has the 80 records, `results/outputs/` every raw
answer, and `results/calibration.json` what the card draws doing nothing.
Two of the 80 are marked `excluded` in `models.toml` (see below), so the
dataset is 78 models and `frontier.summary()` gives 234 rows.
`frontier.py` has not been run on them: the bar rule below is not built.

A four-model pilot came first, on 500 items a task. Its rows are not
comparable and sit under `results/pilot/`, which `frontier.py` does not read.
That folder also holds the two extra runs of the 0.5B model made for noise.

What the pilot and the full run confirmed:

- vLLM 0.31.0 loads all 80 with the arguments `run.py` passes, including the
  five repos that go through the Mistral path described below.
- The energy counter agrees with sampled power to within 2.1% on every task of
  every model. No prompt tokens were served from cache.
- Energy repeats on one pod: `joules_per_item` spread by under 1% across three
  fresh runs in the pilot. Scores do not quite: tool-call accuracy ran 0.805 to
  0.815, because greedy decoding in a batch is not exactly repeatable. A bar
  needs that much margin.
- **A different pod reads about 3% differently.** Qwen2.5-0.5B on the same 400
  tool calls, same software and same throughput, cost 1.063 J an item on the
  full run's pod against 1.096 to 1.100 J across the pilot's three runs. That
  is one comparison, and the driver differed too (580 against 595). It does
  not touch this dataset, which is all one card. It does mean a model measured
  later, on another pod, needs two or three models from this run measured
  beside it, since 3% is the size of a step.
- The card is busy for nearly everything: mean draw was 276 to 350 W, median
  341 W. The exception is the 90M Falcon, at 200 W on extraction and 212 W on
  tool calls.
- Pace: 5.13 hours for the 80. About 2.5 minutes a model under 2B, 3 at 3B, 4
  at 7 to 8B, 6 at 14B; the slowest was Llama-2-13b at 8.7. The first model on
  a pod takes about two minutes longer, half of it a one-time kernel build.
  Disk was never seen above 36 GB.
- `scripts/pod_guard.sh` works: both pods were terminated with it.

Decided, with what the full run added:

- **Extraction scorer, 2026-10-11.** The score is whether a reference answer
  appears in the reply (`contains_answer`); SQuAD's token F1 is kept as the
  `token_f1` column. Under F1 both Mistrals, which answer in sentences, scored
  below the 0.5B model in the pilot though their replies held a reference 93%
  of the time against its 76%. The prompt and the 48-token cap stay. The
  pilot's own files still carry F1 as their score.
- **Intent scorer, 2026-10-12.** The score is whether the reply names the right
  intent and no other (`names_intent`). A name counts wherever it is written as
  the label is, underscores and all, or when the first line is nothing but the
  name. The rule it replaced, first line only, is kept as the `first_line`
  column. It gave Llama-2-7b, Llama-2-13b and gemma-7b-it 0.045, 0.000 and
  0.000 for wrapping the label in a sentence; they now have 0.399, 0.257 and
  0.351. Over the 78 models 2,127 answers gained and 50 lost, the losses being
  replies that list several intents. Steven chose this over counting the
  label's words anywhere, which also credits replies that only echo the
  customer ("the current exchange rate is...") and so favours rambling.
  The 24-token cap still costs two of them: Llama-2-13b is cut off before
  naming any intent in 27% of replies and gemma-7b-it in 36%. Only measuring
  again with a longer cap would fix that, and neither would set a line.
  On the lines: zephyr-7b-beta fell from 0.5515 to 0.541, so the 0.55 bar now
  rests on one 2023 model, Mistral-7B v0.2 at 0.5545 (1,109 of 2,000).
- **`idle_watts` is a fair reading after all.** The calibration at the end of
  the run (`bash scripts/pod_run.sh calibrate`, which now has run on hardware)
  read 33.3 W with the card empty, 94.2 W with a model loaded and quiet, and
  92.7 W over the stretch `idle_watts` reads. The loaded card is steady within
  10 seconds. Across the 80 runs `idle_watts` was 95 to 104 W. The earlier
  worry came from comparing it with the empty card: a loaded engine holds the
  card about 60 W above empty. Nothing is subtracted from any figure.
- **Reasoning is out of scope for the moment, 2026-10-12.** Steven's call; he
  may take up reasoning models or tasks later. `LiquidAI/LFM2.5-8B-A1B` and
  `LiquidAI/LFM2.5-2.6B` always reason and have no switch: their cards say so,
  and the 2.6B's template opens `<think>` itself. The token caps cut off every
  intent and extraction answer mid-thought, so their rows said "could not
  answer inside the cap", not "cannot do the task". They are marked `excluded`
  in `models.toml`, which keeps them out of `summary.csv` and out of any later
  run; their records stay in `results/`. Neither set a step at any bar on the
  0.05 grid, so excluding them moved no line. The 8B is capable given room: of
  the 155 tool calls it finished inside 192 tokens, 96.8% were right. No other
  model is a case of this: 17 templates take `enable_thinking` and ran with it
  off, and none of the other 78 put a thinking marker in an answer.
- **The small end is overcharged.** In the pilot, energy per item followed
  model size times tokens to within 12% across the 3B and 7B models, and the
  0.5B model sat 8 to 37% above them. That biases against the smallest models,
  so it understates the trend. `summary.csv` carries `mean_watts` and
  `joules_per_billion_params_per_1000_tokens` to show it. Not yet looked at
  across the 80.

Open, for Steven to decide:

- **Score bars: the rule is agreed, the numbers wait for the code.** Agreed
  with Steven on 2026-10-11, before the full run, so the trend cannot steer it:
  1. `frontier.csv` carries every bar on a 0.05 grid.
  2. The chart shows two bars per task: the highest that some model released
     in 2023 clears, and the highest that at least three models clear.
  3. A model clears a bar when its score is at or above it. A record-setter
     whose score interval straddles the bar is flagged.
  4. A new low counts as a step only if it is about 3% below the last; energy
     repeats to within 1%.
  None of this is in `frontier.py` yet, which still takes `--bars` and
  defaults to 0.7/0.8/0.9. Why a rule was needed: on 500 items a score was
  only good to 3 or 4 points either way. The larger sets halve that for intent
  and extraction; tool calls stay at 400 items. It will bite at once: on
  intent the best 2023 model scores 0.5545, against a 0.55 bar.

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

Spent so far: **about $6.25**, by pod:

- `rm2ki4tybbbtyc`, the pilot: $0.37 billed. Secure cloud, 20 minutes on
  2026-10-11.
- `hr1nmckg7ts46y`, the full run: about $5.88. Secure cloud, $1.09/hr plus
  disk, alive 5.33 hours from 19:47 UTC on 2026-10-11. The billing API had
  posted $4.89 of it, for 4.43 hours, when this was written; `runpodctl
  billing pods --grouping podId` has the final figure.

The guard is two timers, because nothing RunPod offers does it: `runpodctl pod
create` has no terminate-after flag through 2.15.0 (the RunPod plugin's docs say
it does), and neither has the MCP `create-pod`. The API's GraphQL input does
define `terminateAfter`; that is untried. So `scripts/pod_guard.sh` runs on the
pod and a `sleep && runpodctl pod delete` runs here. Terminating deletes the
pod's disk, so on a long run copy `results/` back as it goes.

## Running on a pod

The commands are in the README under "Running it", and all of them have now run.
What they do not say:

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
- The Bash tool here is zsh, which does not split a command kept in a
  variable. Put the pod's host and port in an ssh config file and use
  `ssh -F`.
- `pgrep -f run_all` sent over SSH matches the remote shell carrying it, so it
  never reports the run as over. Write the pattern as `"[r]un_all"`.
- For a long run, a local loop copied `results/` back every 10 minutes. The
  machine has to stay awake for it; `caffeinate -i` on the local guard does
  that while the lid is open.

## Next steps

1. Build the bar rule into `frontier.py` and run it, for `summary.csv` and
   `frontier.csv`.
2. Then the chart.

All 14 gated models are measured. The read-only token went to the pod as a
file piped over SSH, so its value was never shown, and it went with the pod's
disk. Meta approved the last three Llama 3.x requests while the run was under
way.

Adding a model later means another pod, and so another physical card: measure
two or three models from this run beside it, and see the 3% above.

## Rules that protect the measurements

- **Do not change the protocol constants in `run.py` or the items in
  `data/tasks/` once real results exist.** Either one makes old and new rows
  incomparable. If one must change, every model is measured again.
- **One GPU model for every run.** Energy on different cards is not comparable;
  `frontier.py` warns if runs span more than one.
- **Fix scoring bugs by rescoring, not rerunning.** `results/outputs/` holds
  every raw answer for that reason, and `python -m llm_energy_per_task.rescore`
  redoes the scores from them (`--dry-run` first). Run with no scorer changed
  it reports nothing moved and leaves every file byte for byte as it was; that
  was checked on all 80 records before its first use. It does not touch
  `results/pilot/`.
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
