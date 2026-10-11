"""Stand-ins for the GPU-side libraries, so the measuring code runs without a GPU."""

import time


class FakeNvml:
    """A GPU drawing a constant 200 W, with a counter that agrees."""

    class NVMLError(Exception):
        pass

    def __init__(self, gpus=1, counter=True, watts=200.0):
        self.gpus, self.counter, self.watts = gpus, counter, watts
        self.born = time.perf_counter()

    def nvmlInit(self):
        pass

    def nvmlDeviceGetCount(self):
        return self.gpus

    def nvmlDeviceGetHandleByIndex(self, index):
        return index

    def nvmlDeviceGetName(self, handle):
        return b"Fake L40S"

    def nvmlSystemGetDriverVersion(self):
        return "1.0"

    def nvmlDeviceGetPowerUsage(self, handle):
        return self.watts * 1000

    def nvmlDeviceGetTotalEnergyConsumption(self, handle):
        if not self.counter:
            raise self.NVMLError("not supported")
        return (time.perf_counter() - self.born) * self.watts * 1000
