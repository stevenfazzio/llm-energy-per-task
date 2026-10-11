"""GPU energy for a stretch of work, read from NVML.

Two independent readings of the same window: the driver's cumulative energy
counter, and sampled power integrated over time. The counter is the figure of
record where the GPU has one; the integral is the fallback and, when both exist,
a check that the counter is behaving.

This is the GPU alone. CPU, memory, cooling and the rest of the server are not
in it, so every number here is a floor on what the work really drew.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Reading:
    seconds: float
    counter_joules: float | None
    sampled_joules: float
    peak_watts: float
    samples: int

    @property
    def joules(self) -> float:
        return self.sampled_joules if self.counter_joules is None else self.counter_joules

    @property
    def source(self) -> str:
        return "sampled" if self.counter_joules is None else "counter"

    @property
    def mean_watts(self) -> float:
        return self.joules / self.seconds


class GpuMeter:
    """Meters one GPU between start() and stop()."""

    def __init__(
        self, index: int | None = None, interval: float = 0.1, nvml=None, clock=time.perf_counter
    ):
        if nvml is None:
            import pynvml as nvml
        self._nvml = nvml
        self._clock = clock
        self._interval = interval
        self._thread: threading.Thread | None = None

        nvml.nvmlInit()
        if index is None:
            # Metering the wrong card would give a plausible number for no work.
            count = nvml.nvmlDeviceGetCount()
            if count != 1:
                raise RuntimeError(f"{count} GPUs are visible; say which one to meter")
            index = 0
        self._handle = nvml.nvmlDeviceGetHandleByIndex(index)

    def describe(self) -> dict:
        name = self._nvml.nvmlDeviceGetName(self._handle)
        driver = self._nvml.nvmlSystemGetDriverVersion()
        return {
            "gpu": name.decode() if isinstance(name, bytes) else name,
            "driver": driver.decode() if isinstance(driver, bytes) else driver,
            "energy_counter": self._counter() is not None,
        }

    def _counter(self) -> float | None:
        try:
            return self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle) / 1000  # mJ
        except self._nvml.NVMLError:
            return None

    def _sample(self) -> tuple[float, float]:
        return self._clock(), self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000  # mW

    def _poll(self) -> None:
        while not self._halt.wait(self._interval):
            self._samples.append(self._sample())

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("already metering")
        self._halt = threading.Event()
        self._counter_start = self._counter()
        self._samples = [self._sample()]
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> Reading:
        if self._thread is None:
            raise RuntimeError("not metering")
        self._halt.set()
        self._thread.join()
        self._thread = None
        self._samples.append(self._sample())
        counter_end = self._counter()

        samples = self._samples
        sampled = sum(
            (t1 - t0) * (w0 + w1) / 2
            for (t0, w0), (t1, w1) in zip(samples, samples[1:], strict=False)
        )
        counted = None
        if self._counter_start is not None and counter_end is not None:
            # The counter restarts with the driver; a negative span means it did.
            if counter_end >= self._counter_start:
                counted = counter_end - self._counter_start
        return Reading(
            seconds=samples[-1][0] - samples[0][0],
            counter_joules=counted,
            sampled_joules=sampled,
            peak_watts=max(watts for _, watts in samples),
            samples=len(samples),
        )

    def idle_watts(self, seconds: float) -> float:
        """Mean draw while nothing is asked of the GPU: the floor under every task."""
        self.start()
        time.sleep(seconds)
        return self.stop().mean_watts
