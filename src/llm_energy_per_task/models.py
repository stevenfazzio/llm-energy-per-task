"""The model registry.

models.toml is the hand-edited list of what to measure. models.lock.json holds
what Hugging Face says about each entry -- creation date, parameter count,
gating, chat template, and the commit -- so those facts are fetched rather than
typed, and a run pins the exact weights it measured.

    uv run python -m llm_energy_per_task.models          # fetch entries not yet locked
    uv run python -m llm_energy_per_task.models --all    # re-fetch everything, moving the pins
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from dataclasses import dataclass

from . import ROOT
from .files import write_json

REGISTRY = ROOT / "models.toml"
LOCK = ROOT / "models.lock.json"


@dataclass(frozen=True)
class Model:
    id: str
    # The repo's creation date unless models.toml overrides it. Publishers often
    # create a repo privately some weeks before launch, so this can run early.
    released: str
    params: int | None
    revision: str
    gated: bool
    chat_template: bool
    license: str | None
    pilot: bool = False
    trust_remote_code: bool = False

    @property
    def slug(self) -> str:
        return self.id.replace("/", "__")


def _entries() -> list[dict]:
    entries = tomllib.loads(REGISTRY.read_text())["model"]
    ids = [entry["id"] for entry in entries]
    repeated = sorted({i for i in ids if ids.count(i) > 1})
    if repeated:
        raise ValueError(f"models.toml lists these more than once: {repeated}")
    return entries


def load_models() -> list[Model]:
    if not LOCK.exists():
        raise FileNotFoundError(
            "models.lock.json is missing; run: uv run python -m llm_energy_per_task.models"
        )
    lock = json.loads(LOCK.read_text())

    models = []
    for entry in _entries():
        facts = lock.get(entry["id"])
        if facts is None:
            raise KeyError(
                f"{entry['id']} is in models.toml but not models.lock.json; "
                "run: uv run python -m llm_energy_per_task.models"
            )
        released = entry.get("released")
        models.append(
            Model(
                id=entry["id"],
                released=released.isoformat() if released else facts["created"],
                params=entry.get("params", facts["params"]),
                revision=facts["revision"],
                gated=facts["gated"],
                chat_template=facts["chat_template"],
                license=facts["license"],
                pilot=entry.get("pilot", False),
                trust_remote_code=entry.get("trust_remote_code", False),
            )
        )
    return models


def _facts(api, model_id: str) -> dict:
    info = api.model_info(model_id)
    config = info.config or {}
    tokenizer = config.get("tokenizer_config") or {}
    files = {sibling.rfilename for sibling in info.siblings or []}
    return {
        "created": info.created_at.date().isoformat(),
        "params": info.safetensors.total if info.safetensors else None,
        "revision": info.sha,
        "gated": bool(info.gated),
        # Older repos keep the template in the tokenizer config, newer ones in a file.
        "chat_template": bool(tokenizer.get("chat_template")) or "chat_template.jinja" in files,
        "license": next(
            (tag.split(":", 1)[1] for tag in info.tags or [] if tag.startswith("license:")), None
        ),
    }


def refresh(everything: bool = False) -> list[str]:
    """Fetch facts for unlocked entries, or all of them. Returns the ids that failed."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    api = HfApi()
    lock = json.loads(LOCK.read_text()) if LOCK.exists() else {}
    wanted = [e["id"] for e in _entries() if everything or e["id"] not in lock]

    failed = []
    for model_id in wanted:
        try:
            facts = _facts(api, model_id)
        except HfHubHTTPError as error:
            failed.append(model_id)
            print(f"  FAILED {model_id}: {error.response.status_code}", file=sys.stderr)
            continue
        previous = lock.get(model_id)
        if previous and previous["revision"] != facts["revision"]:
            print(
                f"  pin moved for {model_id}: {previous['revision'][:8]} -> {facts['revision'][:8]}"
            )
        lock[model_id] = facts

    listed = {entry["id"] for entry in _entries()}
    write_json(LOCK, {model_id: lock[model_id] for model_id in sorted(lock) if model_id in listed})
    print(f"fetched {len(wanted) - len(failed)} of {len(wanted)}; {len(lock)} locked")
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--all", action="store_true", help="re-fetch every entry, moving the pins")
    args = parser.parse_args()

    failed = refresh(everything=args.all)

    listed = {entry["id"] for entry in _entries()}
    locked = set(json.loads(LOCK.read_text()))
    if listed - locked:
        print(f"not locked, so not loadable: {sorted(listed - locked)}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'model':44s} {'released':10s} {'params':>8s}  gated  template  licence")
    for m in sorted(load_models(), key=lambda m: m.released):
        size = f"{m.params / 1e9:.2f}B" if m.params else "?"
        flags = f"{'yes' if m.gated else '-':5s}  {'yes' if m.chat_template else 'NO':8s}"
        print(f"{m.id:44s} {m.released:10s} {size:>8s}  {flags}  {m.license or '?'}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
