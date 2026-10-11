"""Freeze the task items.

Draws a fixed sample from each source dataset, at a pinned commit, into
data/tasks/. Those files are the experiment: every model is measured on exactly
these items, so once anything has been measured they must not change. Hence the
refusal to overwrite without --force.

    uv run python -m llm_energy_per_task.tasks.prepare
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from .. import TASK_DATA
from ..files import write_json, write_jsonl
from . import TASKS

SEED = 0
SAMPLE = 500

# The longest context any model here must fit is 2,048 tokens. At a pessimistic
# three characters per token, a prompt this long still leaves room for the answer.
MAX_PROMPT_CHARS = 5_500

SOURCES = {
    "intent": {
        "dataset": "mteb/banking77",
        "revision": "18072d2685ea682290f7b8924d94c62acc19c0b2",
        "files": ["data/test-00000-of-00001.parquet"],
        "origin": "Banking77, PolyAI (CC BY 4.0), read from the mteb parquet mirror",
    },
    "extraction": {
        "dataset": "rajpurkar/squad",
        "revision": "7b6d24c440a36b6815f21b70d25016731768db1f",
        "files": ["plain_text/validation-00000-of-00001.parquet"],
        "origin": "SQuAD v1.1, Rajpurkar et al. (CC BY-SA 4.0)",
    },
    "toolcall": {
        "dataset": "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        "revision": "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda",
        "files": ["BFCL_v3_simple.json", "possible_answer/BFCL_v3_simple.json"],
        "origin": "Berkeley Function-Calling Leaderboard v3, simple category (Apache-2.0)",
    },
}


def _download(task: str) -> list[Path]:
    from huggingface_hub import hf_hub_download

    source = SOURCES[task]
    return [
        Path(
            hf_hub_download(
                source["dataset"], name, repo_type="dataset", revision=source["revision"]
            )
        )
        for name in source["files"]
    ]


def _parquet_rows(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def _json_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _draw(rows: list[dict], rng: random.Random) -> list[dict]:
    """A seeded sample, kept in source order so the file diffs cleanly."""
    if len(rows) <= SAMPLE:
        return rows
    return [rows[i] for i in sorted(rng.sample(range(len(rows)), SAMPLE))]


def intent_items(rng: random.Random) -> tuple[list[dict], list[str]]:
    (path,) = _download("intent")
    rows = _parquet_rows(path)
    labels = sorted({row["label_text"] for row in rows})
    assert len(labels) == 77, f"expected 77 intents, found {len(labels)}"
    indexed = [
        {"id": f"banking77-test-{i}", "text": row["text"], "label": row["label_text"]}
        for i, row in enumerate(rows)
    ]
    return _draw(indexed, rng), labels


def extraction_items(rng: random.Random) -> list[dict]:
    (path,) = _download("extraction")
    rows = _parquet_rows(path)
    return [
        {
            "id": row["id"],
            "context": row["context"],
            "question": row["question"],
            "answers": list(dict.fromkeys(row["answers"]["text"])),
        }
        for row in _draw(rows, rng)
    ]


def toolcall_items(rng: random.Random) -> list[dict]:
    questions_path, answers_path = _download("toolcall")
    answers = {row["id"]: row["ground_truth"] for row in _json_lines(answers_path)}

    items = []
    for row in _json_lines(questions_path):
        truth = answers[row["id"]]
        # "Simple" means one function, one turn, one call. Anything else would
        # be a different task, so stop here and look at it.
        assert len(row["function"]) == 1 and len(truth) == 1 and len(row["question"]) == 1, row[
            "id"
        ]
        (function,) = row["function"]
        # The name to match is the one the function is described under. The key
        # in the answer file disagrees with it on one row (simple_363, an
        # unqualified name), so only the accepted arguments are read from there.
        (expected,) = truth[0].values()
        request = "\n".join(m["content"] for m in row["question"][0] if m["role"] == "user")
        items.append(
            {"id": row["id"], "request": request, "function": function, "expected": expected}
        )
    return _draw(items, rng)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="replace an existing sample")
    args = parser.parse_args()

    existing = sorted(p.name for p in TASK_DATA.glob("*.json*"))
    if existing and not args.force:
        sys.exit(
            f"data/tasks already holds {existing}. Replacing the sample invalidates every "
            "measurement made on it; pass --force if that is what you mean."
        )

    # One generator per task, so adding or resizing a task leaves the others' samples alone.
    intent, labels = intent_items(random.Random(SEED))
    samples = {
        "intent": intent,
        "extraction": extraction_items(random.Random(SEED)),
        "toolcall": toolcall_items(random.Random(SEED)),
    }

    # The labels are read by the intent prompt, so they must land before it is built.
    write_json(TASK_DATA / "intent_labels.json", labels)

    manifest = {}
    for name, items in samples.items():
        ids = [item["id"] for item in items]
        assert len(set(ids)) == len(ids), f"{name}: duplicate ids"
        longest = max(len(TASKS[name].prompt(item)) for item in items)
        assert longest <= MAX_PROMPT_CHARS, f"{name}: a prompt is {longest} characters"
        write_jsonl(TASK_DATA / f"{name}.jsonl", items)
        manifest[name] = {
            **SOURCES[name],
            "items": len(items),
            "seed": SEED,
            "longest_prompt_chars": longest,
        }
        print(f"{name}: {len(items)} items, longest prompt {longest} characters")
    write_json(TASK_DATA / "manifest.json", manifest)


if __name__ == "__main__":
    main()
