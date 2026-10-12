"""Rescoring redoes the scores from the stored answers and moves nothing that was measured."""

import json

import pytest

from llm_energy_per_task import TASK_DATA
from llm_energy_per_task.files import read_jsonl, write_json, write_jsonl
from llm_energy_per_task.rescore import rescore
from llm_energy_per_task.tasks import load_items

pytestmark = pytest.mark.skipif(
    not (TASK_DATA / "manifest.json").exists(), reason="tasks not prepared"
)


@pytest.fixture
def stale(tmp_path):
    """One model's intent run, scored by a rule that gave every answer nothing."""
    right, wrapped, wrong = load_items("intent")[:3]
    answers = [right["label"], f'The intent is "{wrapped["label"]}".', "no idea"]
    rows = [
        {
            "id": item["id"],
            "output": output,
            "score": 0.0,
            "prompt_tokens": 600,
            "output_tokens": 9,
            "finish_reason": "stop",
        }
        for item, output in zip((right, wrapped, wrong), answers, strict=True)
    ]
    write_jsonl(tmp_path / "outputs" / "example__m" / "intent.jsonl", rows)
    entry = {"metric": "old_rule", "score": 0.0, "also": {}, "items": 3, "joules_per_item": 1.5}
    write_json(
        tmp_path / "runs" / "example__m.json", {"model": "example/m", "tasks": {"intent": entry}}
    )
    return tmp_path


def test_scores_are_redone_and_measurements_left_alone(stale):
    assert rescore(stale) == [("example/m", "intent", "names_intent", 0.0, 2 / 3)]

    rows = read_jsonl(stale / "outputs" / "example__m" / "intent.jsonl")
    assert [row["score"] for row in rows] == [1.0, 1.0, 0.0]
    assert [row["first_line"] for row in rows] == [1.0, 0.0, 0.0]
    # A row reads as run.py would have written it: text, scores, then what was measured.
    assert list(rows[0]) == [
        "id",
        "output",
        "score",
        "first_line",
        "prompt_tokens",
        "output_tokens",
        "finish_reason",
    ]
    assert all((row["prompt_tokens"], row["output_tokens"]) == (600, 9) for row in rows)

    entry = json.loads((stale / "runs" / "example__m.json").read_text())["tasks"]["intent"]
    assert entry["metric"] == "names_intent"
    assert entry["score"] == 2 / 3
    assert entry["also"] == {"first_line": 1 / 3}
    assert entry["joules_per_item"] == 1.5


def test_rescoring_twice_changes_nothing_the_second_time(stale):
    rescore(stale)
    files = [stale / "runs" / "example__m.json", stale / "outputs" / "example__m" / "intent.jsonl"]
    before = [path.read_bytes() for path in files]
    assert rescore(stale) == []
    assert [path.read_bytes() for path in files] == before


def test_a_dry_run_reports_and_writes_nothing(stale):
    files = [stale / "runs" / "example__m.json", stale / "outputs" / "example__m" / "intent.jsonl"]
    before = [path.read_bytes() for path in files]
    assert len(rescore(stale, write=False)) == 1
    assert [path.read_bytes() for path in files] == before
