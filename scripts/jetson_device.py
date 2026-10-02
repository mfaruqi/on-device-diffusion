"""Jetson system-memory capture. Tegrastats is not a per-process GPU allocator.

This adapter preserves raw system RAM/swap samples; it does not manufacture NVML
or per-stage peaks from a monitor with no synchronized stage clock.
"""

import subprocess


class TegrastatsMonitor:
    def __init__(self, path, interval_ms):
        if type(interval_ms) is not int or interval_ms <= 0:
            raise ValueError("tegrastats interval must be a positive integer in milliseconds")
        self.path = path
        self.interval_ms = interval_ms
        self.process = None
        self.stream = None

    def start(self):
        self.stream = self.path.open("w")
        try:
            self.process = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=self.stream, stderr=subprocess.STDOUT,
            )
        except BaseException:
            self.stream.close()
            raise

    def stop(self):
        try:
            if self.process is not None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        finally:
            if self.stream is not None:
                self.stream.close()


def memory_snapshot(path):
    """Best-effort diagnostics must not hide the original benchmark exception."""
    try:
        output = subprocess.check_output(["free", "-h"], text=True)
    except Exception as error:
        output = f"Memory snapshot unavailable: {error!r}\n"
    path.write_text(output)
