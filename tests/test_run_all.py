import json
import os
import signal
import sys
import time
from dataclasses import replace

import pytest
from test_run import MODEL

from llm_energy_per_task import frontier, run_all


def test_skip_reasons(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    limit = 15_000_000_000

    assert run_all.skip_reason(MODEL, tmp_path, limit, False) is None
    assert "15B" in run_all.skip_reason(
        replace(MODEL, params=30_000_000_000), tmp_path, limit, False
    )
    assert "template" in run_all.skip_reason(
        replace(MODEL, chat_template=False), tmp_path, limit, False
    )

    left_out = replace(MODEL, excluded="reasons, with no switch")
    assert "reasons" in run_all.skip_reason(left_out, tmp_path, limit, True)

    gated = replace(MODEL, gated=True)
    assert "HF_TOKEN" in run_all.skip_reason(gated, tmp_path, limit, False)
    monkeypatch.setenv("HF_TOKEN", "x")
    assert run_all.skip_reason(gated, tmp_path, limit, False) is None

    (tmp_path / "failed").mkdir()
    (tmp_path / "failed" / f"{MODEL.slug}.txt").write_text("exit 1")
    assert "failed before" in run_all.skip_reason(MODEL, tmp_path, limit, False)
    assert run_all.skip_reason(MODEL, tmp_path, limit, True) is None

    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / f"{MODEL.slug}.json").write_text("{}")
    assert run_all.skip_reason(MODEL, tmp_path, limit, True) == "already measured"


def test_a_timeout_takes_the_whole_process_tree_with_it(tmp_path):
    log = tmp_path / "model.log"
    shared_blobs = "import os; print(os.environ['HF_HUB_DISABLE_SHARED_BLOBS'])"
    assert run_all.run_model([sys.executable, "-c", shared_blobs], log, 30) == 0
    assert log.read_text() == "1\n"  # or weights_dir() is not where the weights are

    # A stand-in for run.py with vLLM's engine process under it: a parent that
    # starts a child of its own and then hangs.
    hangs = (
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "print(child.pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    assert run_all.run_model([sys.executable, "-c", hangs], log, 2) == "timeout"

    child = int(log.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    os.kill(child, signal.SIGKILL)
    raise AssertionError("the hung model's child process outlived the timeout")


def test_weights_dir_follows_the_hub_cache_layout(monkeypatch):
    monkeypatch.setenv("HF_HOME", "/workspace/hf-cache")
    expected = "/workspace/hf-cache/hub/models--example--tiny-1B-instruct"
    assert str(run_all.weights_dir(MODEL)) == expected


def test_summary_reads_run_files_and_prefers_the_registry_date(tmp_path, monkeypatch):
    task = {
        "metric": "accuracy",
        "items": 400,
        "score": 0.8,
        "also": {"token_f1": 0.5},
        "joules_per_item": 3.6,
        "prompt_tokens_per_item": 396.0,
        "output_tokens_per_item": 4.0,
        "mean_watts": 300.0,
        "energy_source": "counter",
    }
    run = {
        "model": MODEL.id,
        "released": "2024-12-25",
        "params": MODEL.params,
        "environment": {"gpu": "Fake L40S"},
        "tasks": {"intent": task},
    }
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / f"{MODEL.slug}.json").write_text(json.dumps(run))
    monkeypatch.setattr(frontier, "RESULTS", tmp_path)

    monkeypatch.setattr(frontier, "load_models", lambda: [MODEL])
    (row,) = frontier.summary()
    assert row["released"] == MODEL.released
    assert row["wh_per_1000_items"] == 1.0
    assert row["token_f1"] == 0.5
    # 3.6 J for a 1B model over 400 tokens: 9 J per billion parameters per 1000 tokens.
    assert row["joules_per_billion_params_per_1000_tokens"] == pytest.approx(9.0)
    assert frontier.per_unit_of_work(3.6, None, 400.0) is None  # no parameter count published

    monkeypatch.setattr(frontier, "load_models", lambda: [])
    assert frontier.summary()[0]["released"] == "2024-12-25"

    # A model the registry leaves out keeps its run file and has no row.
    monkeypatch.setattr(frontier, "load_models", lambda: [replace(MODEL, excluded="reasons")])
    assert frontier.summary() == []
    monkeypatch.setattr(frontier, "load_models", lambda: [MODEL])

    monkeypatch.setattr("sys.argv", ["frontier"])
    frontier.main()
    lines = (tmp_path / "frontier.csv").read_text().splitlines()
    assert lines[0].startswith("bar,chart,kind,task,model")
    # Scoring 0.8, the one model is the line at every bar from 0.05 to 0.80 ...
    assert len(lines) == 1 + 16
    assert lines[1].startswith("0.05,,step,intent,") and lines[16].startswith("0.80,")
    # ... and 0.80 is the bar a chart shows. Released in 2025, it picks no 2023 bar,
    # and one model is not the three a bar needs.
    assert all(line.split(",")[1] == "" for line in lines[1:])
