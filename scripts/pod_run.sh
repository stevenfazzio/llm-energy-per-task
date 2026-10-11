#!/usr/bin/env bash
# Runs ON THE POD, from the repo root: installs the pinned GPU environment, then
# measures a set of models. Safe to run again after an interruption -- models
# already measured are skipped.
#
#   bash scripts/pod_run.sh pilot
#   bash scripts/pod_run.sh all --max-hours 8
set -euo pipefail

SET="${1:-pilot}"
shift || true

# Weights and caches on the volume when there is one, so a restart keeps them.
if [ -d /workspace ]; then
  export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/uv-cache}"
fi

# Gated models need a Hugging Face token. One left where huggingface_hub keeps
# its own is picked up here, so it never has to appear in a command line.
TOKEN_FILE="${HF_HOME:-$HOME/.cache/huggingface}/token"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$TOKEN_FILE" ]; then
  HF_TOKEN="$(cat "$TOKEN_FILE")"
  export HF_TOKEN
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# The lockfile is the environment of record; --frozen refuses to drift from it.
uv sync --extra gpu --no-dev --frozen

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

uv run --no-sync python -m llm_energy_per_task.run_all --set "$SET" "$@"
