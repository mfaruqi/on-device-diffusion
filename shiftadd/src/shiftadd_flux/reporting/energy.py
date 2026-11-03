"""Gross sampled-power integration; no inference or power-sampling implementation."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def integrate(samples, window):
    required = ("start_s", "end_s", "images", "clock_id", "power_scope", "max_gap_s")
    if any(k not in window for k in required):
        raise ValueError("Incomplete measurement window")
    start, end = float(window["start_s"]), float(window["end_s"])
    gap = float(window["max_gap_s"])
    images = window["images"]
    if not all(math.isfinite(v) for v in (start, end, gap)) or start >= end or gap <= 0:
        raise ValueError("Invalid time bounds or gap limit")
    if type(images) is not int or images <= 0:
        raise ValueError("Require completed-image count")
    if not window["clock_id"] or not window["power_scope"]:
        raise ValueError("Declare clock and power boundary")
    if len(samples) < 3:
        raise ValueError("Insufficient samples")
    if any(
        s["clock_id"] != window["clock_id"] or s["power_scope"] != window["power_scope"]
        for s in samples
    ):
        raise ValueError("Power scope or clock mismatch")
    times = np.array([float(s["timestamp_s"]) for s in samples], np.float64)
    watts = np.array([float(s["power_w"]) for s in samples], np.float64)
    if (
        not np.isfinite(times).all()
        or not np.isfinite(watts).all()
        or np.any(watts < 0)
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("Samples must be finite, nonnegative power, strictly increasing time")
    if start < times[0] or end > times[-1]:
        raise ValueError("Samples do not bracket the measurement window")
    segments = (times[:-1] < end) & (times[1:] > start)
    max_gap = float(np.diff(times)[segments].max())
    if max_gap > gap:
        raise ValueError("Power sample gap exceeds declared limit")
    inside = (times >= start) & (times <= end)
    if int(inside.sum()) < 3:
        raise ValueError("Need at least three actual samples within the window")
    interior = (times > start) & (times < end)
    t = np.concatenate(([start], times[interior], [end]))
    p = np.concatenate(
        (
            [np.interp(start, times, watts)],
            watts[interior],
            [np.interp(end, times, watts)],
        )
    )
    joules = float(np.sum(np.diff(t) * (p[:-1] + p[1:]) * 0.5))
    return {
        "gross_energy_j": joules,
        "gross_j_per_image": joules / images,
        "duration_s": end - start,
        "mean_power_w": joules / (end - start),
        "completed_images": images,
        "samples_within_window": int(inside.sum()),
        "max_sample_gap_s": max_gap,
        "clock_id": window["clock_id"],
        "power_scope": window["power_scope"],
        "method": "trapezoid with interpolated endpoints; no extrapolation or idle subtraction",
        "limitations": "sampled power estimate; excludes unsampled transients and sensor uncertainty",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", required=True)
    p.add_argument("--window", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    with Path(a.samples).open() as f:
        samples = list(csv.DictReader(f))
    window = json.loads(Path(a.window).read_text())
    result = integrate(samples, window)
    with Path(a.output).open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
