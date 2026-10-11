import time

import pytest
from fakes import FakeNvml

from llm_energy_per_task.energy import GpuMeter
from llm_energy_per_task.run import timed_passes


def test_counter_and_sampled_power_agree():
    meter = GpuMeter(nvml=FakeNvml(), interval=0.01)
    meter.start()
    time.sleep(0.2)
    reading = meter.stop()

    assert reading.source == "counter"
    assert reading.seconds == pytest.approx(0.2, abs=0.05)
    assert reading.sampled_joules == pytest.approx(200 * reading.seconds, rel=1e-6)
    assert reading.counter_joules == pytest.approx(reading.sampled_joules, rel=0.02)
    assert reading.mean_watts == pytest.approx(200, rel=0.02)
    assert reading.samples > 5


def test_falls_back_to_sampled_power_without_a_counter():
    meter = GpuMeter(nvml=FakeNvml(counter=False), interval=0.01)
    assert meter.describe() == {"gpu": "Fake L40S", "driver": "1.0", "energy_counter": False}
    meter.start()
    time.sleep(0.05)
    reading = meter.stop()
    assert reading.source == "sampled"
    assert reading.joules == reading.sampled_joules > 0


def test_refuses_to_guess_between_gpus():
    with pytest.raises(RuntimeError, match="2 GPUs"):
        GpuMeter(nvml=FakeNvml(gpus=2))
    GpuMeter(index=1, nvml=FakeNvml(gpus=2))


def test_timed_passes_repeats_until_the_window_is_long_enough():
    meter = GpuMeter(nvml=FakeNvml(), interval=0.01)
    ticks = iter(range(100))
    calls = []

    def generate():
        calls.append(len(calls))
        return [f"pass {len(calls)}"]

    # The clock advances one second per reading: began=0, then 1, 2, 3 after each pass.
    first, passes, reading = timed_passes(generate, meter, 3.0, clock=lambda: next(ticks))
    assert first == ["pass 1"]
    assert passes == len(calls) == 3
    assert reading.joules > 0


def test_timed_passes_stops_the_meter_when_generation_fails():
    meter = GpuMeter(nvml=FakeNvml(), interval=0.01)

    def generate():
        raise ValueError("out of memory")

    with pytest.raises(ValueError):
        timed_passes(generate, meter, 1.0)
    meter.start()  # would raise "already metering" had it been left running
    meter.stop()
