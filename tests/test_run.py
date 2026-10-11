"""The measuring path end to end, with the engine and the GPU faked.

This cannot say whether vLLM or NVML behave as assumed -- only a pod can -- but
it does run every line between them, so a typo is found here and not on rented
hardware.
"""

import json
import sys
import types

import pytest
from fakes import FakeNvml

from llm_energy_per_task import TASK_DATA, run
from llm_energy_per_task.energy import GpuMeter
from llm_energy_per_task.models import Model

pytestmark = pytest.mark.skipif(
    not (TASK_DATA / "manifest.json").exists(), reason="tasks not prepared"
)

MODEL = Model(
    id="example/tiny-1B-instruct",
    released="2025-01-01",
    params=1_000_000_000,
    revision="abc123",
    gated=False,
    chat_template=True,
    license="apache-2.0",
)


class FakeLLM:
    """Answers every prompt the same way, and remembers how it was configured."""

    engine_args: dict = {}
    calls: list = []

    def __init__(self, **engine_args):
        FakeLLM.engine_args = engine_args
        FakeLLM.calls = []

    def chat(self, conversations, params, use_tqdm, chat_template_kwargs):
        FakeLLM.calls.append((len(conversations), params.max_tokens, chat_template_kwargs))
        completion = types.SimpleNamespace(
            text="<think>hm</think>card_arrival", token_ids=[1, 2, 3], finish_reason="stop"
        )
        return [
            types.SimpleNamespace(
                prompt_token_ids=list(range(len(c[0]["content"]) // 4)),
                outputs=[completion],
                num_cached_tokens=0,
            )
            for c in conversations
        ]


@pytest.fixture
def fake_gpu(monkeypatch):
    vllm = types.ModuleType("vllm")
    vllm.LLM = FakeLLM
    vllm.SamplingParams = lambda **kwargs: types.SimpleNamespace(**kwargs)
    vllm.__version__ = "0.0-fake"
    torch = types.ModuleType("torch")
    torch.__version__ = "0.0-fake"
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(run, "GpuMeter", lambda: GpuMeter(nvml=FakeNvml(), interval=0.01))
    monkeypatch.setattr(run, "SETTLE_S", 0.0)
    monkeypatch.setattr(run, "IDLE_S", 0.05)


def test_measure_builds_a_complete_record(fake_gpu):
    record, outputs = run.measure(MODEL, list(run.TASKS), limit=8, min_window_s=0.05)

    assert FakeLLM.engine_args["enable_prefix_caching"] is False
    assert FakeLLM.engine_args["revision"] == "abc123"
    assert all(kwargs == {"enable_thinking": False} for _, _, kwargs in FakeLLM.calls)

    json.dumps(record)  # everything in it must be writable
    assert record["idle_watts"] == pytest.approx(200, rel=0.05)
    assert record["environment"]["gpu"] == "Fake L40S"
    assert set(record["tasks"]) == set(run.TASKS) == set(outputs)

    for name, task in record["tasks"].items():
        assert task["items"] == len(outputs[name]) == 8
        assert task["passes"] >= 1
        # Energy is spread over every pass, not just the one that was scored.
        expected = task["counter_joules"] / (task["passes"] * 8)
        assert task["joules_per_item"] == pytest.approx(expected)
        assert task["energy_source"] == "counter"
        assert task["output_tokens_per_item"] == 3
        assert task["cached_prompt_tokens"] == 0
        assert 0.0 <= task["score"] <= 1.0

    # The reasoning block is stripped before scoring but kept in the raw output.
    labelled = {row["id"]: row for row in outputs["intent"]}
    assert all(row["output"].startswith("<think>") for row in labelled.values())
    assert record["tasks"]["toolcall"]["score"] == 0.0


def test_a_smoke_run_writes_outside_the_real_results(fake_gpu, monkeypatch, tmp_path):
    monkeypatch.setattr(run, "RESULTS", tmp_path)
    monkeypatch.setattr(run, "load_models", lambda: [MODEL])
    monkeypatch.setattr(
        sys, "argv", ["run", "--model", MODEL.id, "--limit", "4", "--min-window", "0.05"]
    )
    run.main()

    written = json.loads((tmp_path / "smoke" / "runs" / f"{MODEL.slug}.json").read_text())
    assert written["model"] == MODEL.id
    for name in run.TASKS:
        lines = (tmp_path / "smoke" / "outputs" / MODEL.slug / f"{name}.jsonl").read_text()
        assert len(lines.splitlines()) == 4
    assert not (tmp_path / "runs").exists()
