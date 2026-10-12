"""Checks on the frozen items and the registry, skipped until they have been generated."""

import pytest

from llm_energy_per_task import RESULTS, TASK_DATA
from llm_energy_per_task.models import LOCK, load_models
from llm_energy_per_task.tasks import TASKS, intent, load_items
from llm_energy_per_task.tasks.prepare import MAX_PROMPT_CHARS

prepared = pytest.mark.skipif(
    not (TASK_DATA / "manifest.json").exists(), reason="tasks not prepared"
)
locked = pytest.mark.skipif(not LOCK.exists(), reason="models not locked")


@prepared
@pytest.mark.parametrize("name", list(TASKS))
def test_items_are_unique_and_every_prompt_fits(name):
    items = load_items(name)
    assert len(items) >= 400
    assert len({item["id"] for item in items}) == len(items)
    assert max(len(TASKS[name].prompt(item)) for item in items) <= MAX_PROMPT_CHARS


@prepared
def test_every_intent_label_is_one_the_prompt_offers():
    offered = set(intent.labels())
    assert len(offered) == 77
    assert {item["label"] for item in load_items("intent")} <= offered


@prepared
def test_a_perfect_answer_scores_one():
    for item in load_items("intent")[:50]:
        assert intent.score(item, item["label"]) == 1.0
    for item in load_items("extraction")[:50]:
        assert TASKS["extraction"].score(item, item["answers"][0]) == 1.0


@locked
def test_the_pilot_is_ungated_models_of_three_sizes():
    pilot = [m for m in load_models() if m.pilot]
    assert len(pilot) == 4
    assert all(m.chat_template and not m.gated and m.params for m in pilot)
    # Two of them are 7B: one for each of the ways vLLM loads a Mistral repo.
    small, mid, large, also_large = sorted(m.params for m in pilot)
    assert small * 2 < mid and mid * 2 < large
    assert also_large < large * 1.1


@locked
def test_the_models_left_out_were_measured_and_their_records_kept():
    left_out = [m for m in load_models() if m.excluded]
    assert {m.id for m in left_out} == {"LiquidAI/LFM2.5-8B-A1B", "LiquidAI/LFM2.5-2.6B"}
    if not (RESULTS / "runs").exists():
        pytest.skip("no results here")
    for model in left_out:
        assert (RESULTS / "runs" / f"{model.slug}.json").exists()
        assert (RESULTS / "outputs" / model.slug / "intent.jsonl").exists()
