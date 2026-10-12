# llm-energy-per-task

How much GPU energy an open-weight LLM needs to do an everyday task, and
whether that has kept falling.

For each task and each score bar, the question is: among open models released
by a given date, what is the least energy per item that still scores at least
the bar? Plotted over time that is a running minimum, stepping down only when a
more frugal model clears it. Fixing the score is what makes the comparison mean
something, since the most frugal model is always a weak one.

**Status: measured, and the lines are drawn; the chart is not.** All 80 models
ran on one NVIDIA L40S on 11 October 2026. Two of them turned out to be
reasoning-only and are left out, which makes the dataset 78 models.
`results/summary.csv` has every model's scores and energy, and
`results/frontier.csv` the lines. A four-model pilot ran first, on 500 items a
task; the item sets were then enlarged, so its numbers are not comparable.
They are kept under `results/pilot/` for the record.

## What gets measured

Every model in `models.toml` answers the same frozen items on the same GPU while
the card's energy is metered. One row comes out per model and task: its score,
and joules per item.

| Task | Items | Source | Scored by |
|---|---|---|---|
| `intent` — route a customer message to one of 77 intents | 2,000 | Banking77, test split | the reply names the right intent and no other |
| `extraction` — answer a question with a span from a passage | 2,000 | SQuAD v1.1, validation split | a reference answer appears in the reply |
| `toolcall` — turn a request into one call to a described function | 400 | Berkeley Function-Calling Leaderboard v3, "simple" | name and arguments match |

These stand in for everyday work a language model is put to: routing, pulling an
answer out of retrieved text, and filling in a function call. The tool-call
scorer is this repo's own, modelled on the leaderboard's argument check, so a
score here is not a leaderboard score.

The extraction score is not SQuAD's either. SQuAD's metric, token F1, was built
for models that could only return a span of the passage. A chat model that
wraps the right answer in a sentence loses most of its credit there: in the
pilot, 93% of Mistral-7B-Instruct-v0.2's replies contained a reference answer
and its F1 was 0.37, below a 0.5B model's. Holding quality fixed needs a score
that does not move with how a model talks, and the extra words are already paid
for in energy. So the score is whether a reference answer appears in the reply,
word for word. F1 is published beside it as `token_f1`.

That score errs both ways. A long reply can mention the reference in passing:
in the pilot the wordiest model's replies also held the answer to a different
question on the same passage 13% of the time, against 3% for the tersest. And a
right answer in other words is missed. The first favours older, wordier models,
so it understates how far energy has fallen.

The intent score was also changed for the way models talk, after they had run,
by rescoring their stored answers. The first rule read only the first line of
the reply. Llama-2-7b scored 0.045 under it, because it wraps the label in a
sentence: `The intent of the customer message is "card_arrival".` The score is
now whether the reply names the right intent and no other. A name counts
wherever it is written as the label is, underscores and all, or when the first
line is nothing but the name. The same words as ordinary words elsewhere do
not count, because the labels read like phrases a reply can use without
classifying anything, and a reply that names two intents gets nothing. Four of
the 78 models moved by more than 0.02: Llama-2-7b to 0.399, gemma-7b-it from 0
to 0.351, Llama-2-13b from 0 to 0.257 and gemma-2b-it from 0.170 to 0.207. The
first rule's score is published beside the new one as `first_line`.

Rescoring cannot give back what the 24-token cap took. Llama-2-13b was cut off
before naming any intent in 27% of its replies and gemma-7b-it in 36%, so those
two are still understated on intent.

## The protocol

Held fixed for every model, and written into every result file:

- **One GPU, one serving stack.** vLLM, pinned by `uv.lock`, on a single card.
- **16-bit weights** (`bfloat16`), whatever precision the model was published in.
- **One prompt shape.** Each item is a single user turn through the model's own
  chat template. No system prompt, because several templates reject one.
- **Thinking off** wherever a template has the switch, as 17 of them do. A model
  that cannot answer without thinking first is out of scope for now.
- **Greedy decoding**, with a per-task cap on answer length.
- **Batched hard.** All of a task's items are submitted together and the engine
  works on up to 256 of them at a time. That is vLLM's own choice for a 48 GB
  card, set explicitly so a different card cannot change it. This is the most
  favourable regime; serving one request at a time would cost several times
  more per item.
- **Prefix caching off.** It is on by default in vLLM and would make every
  repeated pass nearly free.
- **A window of at least 30 seconds.** A tiny model clears a task in a second or
  two, too short to meter, so the batch is repeated and the energy divided over
  every pass.

Energy comes from NVML: the driver's cumulative energy counter, cross-checked by
integrating sampled power over the same window. Both are recorded.

Nothing is subtracted for idling: joules per item is everything the card drew
during the window. For scale, the card that measured the dataset drew 33 W
empty and 94 W with a model loaded and nothing to do, against about 340 W
while answering. Each run also records `idle_watts`, read for five seconds
shortly after warm-up; in a test at the end of the run that reading caught
93 W (`results/calibration.json`), and across the 80 models it ran from 95 to
104 W.

## Bars and lines

`results/summary.csv` has one row per model and task. `results/frontier.csv`
has the lines: for each task and each bar from 0.05 to 0.95, the models that
set a new low in energy at or above the bar, in order of release.

How a bar is chosen and what counts as clearing one were settled before the
models were measured, so the result could not pick its own bars:

- **Two bars a task are meant for a chart**, named in the `chart` column. One
  is the highest bar that some model released in 2023 clears (`2023`), so the
  line starts where the dataset does. The other is the highest that at least
  three models clear (`three_models`). They came out as 0.55 and 0.70 for
  intent, 0.90 and 0.95 for extraction, and 0.85 and 0.95 for tool calls.
- **A score is only good to a few points.** Each has a 95% interval
  (`score_low`, `score_high`): one to two points either way on 2,000 items,
  two to five on the 400 tool calls. A model clears a bar when its score is at
  or above it. A step by a model whose interval reaches below the bar has the
  kind `step_straddling`.
- **A step has to be 3% or more below the last one.** Energy repeats to within
  1% on one card.

One thing was added after the models ran: near misses. A model just under a
bar, whose interval reaches it and which would otherwise have been a step, is
recorded with the kind `near_miss`. It does not move the line. There are 61 of
them across the grid against 36 straddling steps, so a line is as likely to be
missing a step by a hair as to have one by a hair.

The tool-call lines want the most care. On the two bars a chart shows, 9 of
their 10 steps straddle, because 400 items leave an interval nearly as wide as
the gap between bars.

## What the number is not

- **Not the whole machine.** GPU only. CPU, memory, cooling and the rest of the
  server are left out, so every figure is a floor.
- **Not a trend in total energy.** Old and new models all run on today's
  hardware and software, which isolates one thing: models getting more capable
  for their size. Gains from better chips are a separate story.
- **Not the cheapest way to do the task.** A fine-tuned classifier a hundredth
  the size handles intent routing for far less. This is about general-purpose
  models, prompted.
- **Not every open model.** Size is no part of the question: if the most frugal
  model to clear a bar is a large one, it belongs on the line. But one 48 GB
  card holds about 15B parameters at 16 bits a weight, and nothing larger was
  measured. That shows in two ways. Three of the six chart lines begin on the
  largest models measured (Llama-2-13b, Phi-3-medium and gemma-3-12b), so a
  larger model released earlier may have cleared those bars sooner and at a
  higher cost, and the 2023 bars were picked from 2023 models under the limit.
  The lines probably start late, and understate the fall. And size is not
  energy: granite-4.0-h-tiny, a mixture-of-experts model with 7B parameters,
  costs about what a 2B to 3B model does here. A model of that kind above 15B
  could sit on a line, and is missing.
- **Not hardware-neutral at the small end.** A very small model cannot keep a
  48 GB card and its serving stack fully busy, so part of its figure is not the
  model's. In the four-model pilot, energy per item followed model size times
  tokens processed to within 12% across the 3B and 7B models on each task. The
  0.5B model came out 8 to 37% above them, most on tool calls, where the card's
  draw also dipped, which points to the card and the serving stack rather than
  the model. The error runs one way: the smallest models are overcharged, so
  the lines understate how far energy has fallen at the small end.
  `summary.csv` has two columns for spotting it: `mean_watts`, and
  `joules_per_billion_params_per_1000_tokens`, which is level across models
  when energy tracks the work done.
- **Not a measure of reasoning.** Every model answers directly, and the caps on
  answer length are sized for that. Two models in `models.toml` always think
  before they answer and have no switch to stop them:
  `LiquidAI/LFM2.5-8B-A1B` and `LiquidAI/LFM2.5-2.6B`. The cap cut them off
  mid-thought, so they are left out of `summary.csv`; their records stay in
  `results/`. Both date from mid-2026. If models that only reason become
  common, this dataset stops seeing the newest ones, and a line going flat
  would not mean the gains had stopped.
- **Not free of contamination.** All three datasets are public and old enough to
  be in training data, more so for newer models.

Release dates are the creation date of the model's Hugging Face repo unless
`models.toml` overrides it. Publishers often create a repo privately some weeks
before launch, so a date can run early.

## Layout

```
models.toml          the models to measure -- the file you edit to add one
models.lock.json     what Hugging Face says about each: date, size, gating, pinned commit
data/tasks/          the frozen items every model is measured on
src/llm_energy_per_task/
  models.py          registry and lock
  tasks/             one module per task (prompt + scorer), and prepare.py to freeze items
  energy.py          the NVML meter
  run.py             measure one model            (pod)
  run_all.py         loop over models, resumable  (pod)
  calibrate.py       what the card draws doing nothing, once per pod  (pod)
  rescore.py         score the stored answers again, after a scorer changes
  frontier.py        results -> summary.csv and frontier.csv
scripts/pod_run.sh   install and run, on the pod
scripts/pod_guard.sh has the pod terminate itself after a time limit
results/
  runs/              one JSON per model: scores, energy, and the environment it ran in
  outputs/           every raw answer, so scoring can be redone without a GPU
  pilot/             the four-model pilot, measured on smaller item sets
```

## Running it

Locally, no GPU needed:

```sh
uv sync
uv run pytest
uv run python -m llm_energy_per_task.models      # lock any models added to models.toml
uv run python -m llm_energy_per_task.frontier    # once there are results
uv run python -m llm_energy_per_task.rescore     # after changing how an answer is scored
```

`data/tasks/` is already frozen. `tasks/prepare.py` regenerates it and refuses to
overwrite without `--force`, because changing the items invalidates every
measurement made on them.

On a rented GPU (run twice on RunPod's secure cloud: the pilot, then all 80
models in a little over five hours):

1. Start a pod with one 48 GB card, a host CUDA of 12.9 or newer (what the
   pinned vLLM wheels are built for), about 60 GB of volume at `/workspace`, and
   SSH on a public IP, which copying files needs:

   ```sh
   runpodctl pod create --name llm-energy --template-id runpod-torch-v280 \
     --gpu-id "NVIDIA L40S" --cloud-type COMMUNITY --public-ip \
     --min-cuda-version 12.9 --ports 22/tcp --volume-in-gb 60 --wait
   runpodctl ssh info POD_ID      # HOST and PORT for the steps below
   ```

2. Arm the guard before anything else. `runpodctl` has no terminate-after
   setting, so the pod is told to terminate itself, and a second timer runs
   locally in case the pod cannot:

   ```sh
   ssh -p PORT root@HOST 'bash -s -- 90' < scripts/pod_guard.sh
   (sleep 5400 && runpodctl pod delete POD_ID) &
   ```

3. Copy this repo over and start the run detached, so it outlives the SSH
   session (`apt-get install -y rsync` on the pod first if the image lacks it):

   ```sh
   rsync -az -e "ssh -p PORT" --exclude .git --exclude .venv --exclude results \
     ./ root@HOST:/workspace/llm-energy-per-task/
   ssh -p PORT root@HOST 'cd /workspace/llm-energy-per-task; \
     setsid bash scripts/pod_run.sh pilot > run.log 2>&1 < /dev/null &'
   ```

   To measure the gated models, leave a Hugging Face token on the pod before
   starting the run. Piped like this it is never shown or put in a command
   line, and `pod_run.sh` picks it up:

   ```sh
   printf '%s' "$HF_TOKEN" | ssh -p PORT root@HOST \
     'umask 077; mkdir -p /workspace/hf-cache; cat > /workspace/hf-cache/token'
   ```

4. Copy `results/` back. Then, with the measurements safe, read what the card
   draws doing nothing (about six minutes), copy that back too, and terminate
   the pod, which deletes its disk:

   ```sh
   rsync -az -e "ssh -p PORT" root@HOST:/workspace/llm-energy-per-task/results/ results/
   ssh -p PORT root@HOST 'cd /workspace/llm-energy-per-task; \
     bash scripts/pod_run.sh calibrate > calibrate.log 2>&1'
   rsync -az -e "ssh -p PORT" root@HOST:/workspace/llm-energy-per-task/results/ results/
   ssh -p PORT root@HOST 'bash -s -- now' < scripts/pod_guard.sh
   ```

`run_all.py` measures smallest models first, skips any already measured, records
a failure and moves on, and deletes each model's weights when it is done.
`--max-hours` stops it starting new models after a time budget. Gated models
(Llama, and Gemma before version 4) are skipped unless a token is on the pod,
and so is a model marked `excluded` in `models.toml`.
One whose licence that account has not accepted is recorded as failed, and
`--retry-failed` picks it up once it has.

## Adding a model

Add an `id` to `models.toml`, then run `python -m llm_energy_per_task.models`.
Everything else about it is fetched.

## Data sources

- **Banking77** — Casanueva et al., 2020, PolyAI. CC BY 4.0. Read from the
  `mteb/banking77` parquet mirror.
- **SQuAD v1.1** — Rajpurkar et al., 2016. CC BY-SA 4.0. The sample in
  `data/tasks/extraction.jsonl` is redistributed under the same licence.
- **Berkeley Function-Calling Leaderboard** — Patil et al. Apache-2.0.

Exact dataset commits are in `data/tasks/manifest.json`.

## Licence

The code and this repo's own measurements (`results/summary.csv`,
`results/frontier.csv`, `results/runs/` and `results/calibration.json`) are
under the MIT licence in `LICENSE`.

Two things here are not this repo's to license:

- **The task items** in `data/tasks/` keep the licences of the datasets they
  were drawn from, listed above. The SQuAD sample is CC BY-SA 4.0, which the
  MIT licence does not replace.
- **The raw answers** in `results/outputs/` are the models' outputs, published
  so the scoring can be checked. Each model's own licence governs what its
  outputs may be used for, and several restrict using them to train or improve
  another model. Extraction answers also quote the SQuAD passages.

The models' licences were read for anything against publishing scores or
outputs, and none has it. Four Qwen models (Qwen1.5 0.5B, 1.8B and 4B, and
Qwen2.5 3B) are licensed for research and evaluation only, which is what was
done with them here.
