# llm-energy-per-task

How much GPU energy a small open-weight LLM needs to do an everyday task, and
whether that has kept falling.

For each task and each score bar, the question is: among open models released
by a given date, what is the least energy per item that still scores at least
the bar? Plotted over time that is a running minimum, stepping down only when a
more frugal model clears it. Fixing the score is what makes the comparison mean
something, since the most frugal model is always a weak one.

**Status: a four-model pilot has run; the other 77 models have not.** The pilot
checked the harness on real hardware. Four models are too few to draw a line
through, so there is no chart yet.

## What gets measured

Every model in `models.toml` answers the same frozen items on the same GPU while
the card's energy is metered. One row comes out per model and task: its score,
and joules per item.

| Task | Items | Source | Scored by |
|---|---|---|---|
| `intent` — route a customer message to one of 77 intents | 500 | Banking77, test split | exact label match |
| `extraction` — answer a question with a span from a passage | 500 | SQuAD v1.1, validation split | token F1 against the closest reference |
| `toolcall` — turn a request into one call to a described function | 400 | Berkeley Function-Calling Leaderboard v3, "simple" | name and arguments match |

These stand in for what small models are actually used for: routing, pulling an
answer out of retrieved text, and filling in a function call. The tool-call
scorer is this repo's own, modelled on the leaderboard's argument check, so a
score here is not a leaderboard score.

## The protocol

Held fixed for every model, and written into every result file:

- **One GPU, one serving stack.** vLLM, pinned by `uv.lock`, on a single card.
- **16-bit weights** (`bfloat16`), whatever precision the model was published in.
- **One prompt shape.** Each item is a single user turn through the model's own
  chat template. No system prompt, because several templates reject one.
- **Thinking off** wherever a template has the switch. A reasoning variant would
  be a separate entry, charged for its thinking tokens.
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

## What the number is not

- **Not the whole machine.** GPU only. CPU, memory, cooling and the rest of the
  server are left out, so every figure is a floor.
- **Not a trend in total energy.** Old and new models all run on today's
  hardware and software, which isolates one thing: models getting more capable
  for their size. Gains from better chips are a separate story.
- **Not the cheapest way to do the task.** A fine-tuned classifier a hundredth
  the size handles intent routing for far less. This is about general-purpose
  models, prompted.
- **Not hardware-neutral at the small end.** A 300M-parameter model cannot keep
  a 48 GB card busy, so part of its figure is the card idling. Idle draw is
  recorded alongside each run.
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
  frontier.py        results -> summary.csv and frontier.csv
scripts/pod_run.sh   install and run, on the pod
scripts/pod_guard.sh has the pod terminate itself after a time limit
results/
  runs/              one JSON per model: scores, energy, and the environment it ran in
  outputs/           every raw answer, so scoring can be redone without a GPU
```

## Running it

Locally, no GPU needed:

```sh
uv sync
uv run pytest
uv run python -m llm_energy_per_task.models      # lock any models added to models.toml
uv run python -m llm_energy_per_task.frontier    # once there are results
```

`data/tasks/` is already frozen. `tasks/prepare.py` regenerates it and refuses to
overwrite without `--force`, because changing the items invalidates every
measurement made on them.

On a rented GPU (run once, for the pilot, on RunPod's secure cloud):

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

4. Copy `results/` back, then terminate the pod, which deletes its disk:

   ```sh
   rsync -az -e "ssh -p PORT" root@HOST:/workspace/llm-energy-per-task/results/ results/
   ssh -p PORT root@HOST 'bash -s -- now' < scripts/pod_guard.sh
   ```

`run_all.py` measures smallest models first, skips any already measured, records
a failure and moves on, and deletes each model's weights when it is done.
`--max-hours` stops it starting new models after a time budget. Gated models
(Llama, and Gemma before version 4) are skipped unless `HF_TOKEN` is set and the
licence has been accepted on that account.

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
