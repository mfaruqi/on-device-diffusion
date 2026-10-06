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


class UnavailableDeviceMemory:
    """No NVML-equivalent samples on Jetson; never substitute system RAM here."""
    def peak_between(self, start, end):
        return None


def summarize_monitor(run_dir):
    """Whole-capture tegrastats RAM/swap diagnostics, with raw evidence preserved."""
    import re
    from measurement import write_csv

    samples = []
    for line_number, line in enumerate((run_dir / 'tegrastats.log').read_text().splitlines(), 1):
        match = re.search(r'RAM (\d+)/(\d+)MB .*?SWAP (\d+)/(\d+)MB', line)
        if not match:
            raise RuntimeError(f'Unrecognized tegrastats line {line_number}; inspect raw log')
        ram, total, swap, swap_total = map(int, match.groups())
        temperature = re.search(r'gpu@([\d.]+)C', line)
        samples.append({'source_line': line_number, 'ram_reported_mb': ram,
                        'ram_total_reported_mb': total, 'swap_reported_mb': swap,
                        'swap_total_reported_mb': swap_total, 'ram_gib': ram / 1024,
                        'swap_gib': swap / 1024,
                        'gpu_temperature_c': float(temperature[1]) if temperature else None})
    if not samples:
        raise RuntimeError('No tegrastats samples; no memory result can be reported')
    write_csv(run_dir / 'tegrastats-samples.csv', samples)
    temperatures = [s['gpu_temperature_c'] for s in samples if s['gpu_temperature_c'] is not None]
    return {'samples': len(samples), 'ram_peak_gib': max(s['ram_gib'] for s in samples),
            'ram_total_gib': max(s['ram_total_reported_mb'] for s in samples) / 1024,
            'swap_min_gib': min(s['swap_gib'] for s in samples),
            'swap_max_gib': max(s['swap_gib'] for s in samples),
            'gpu_temperature_max_c': max(temperatures) if temperatures else None,
            'scope': 'whole harness capture including load; not per-generation or per-stage',
            'unit_convention': 'Printed MB treated as MiB; divide by 1024 for approximate GiB'}
