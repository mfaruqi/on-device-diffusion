"""Shared experiment bookkeeping. Standard library only; never imports an engine.

Timing stays in each engine adapter. These helpers only label runs, summarize
already measured values, and preserve artifacts if a run fails.
"""

from contextlib import contextmanager
import csv
import datetime as dt
import sys
import traceback

from benchlib import now_utc, summarize, write_json


def check_optimizations(cfg, known_keys, enabled_keys=()):
    """Enabled options need explicit adapter support and their own value checks."""
    options = cfg["optimizations"]
    unknown = set(options) - known_keys
    assert not unknown, f"Unknown optimizations: {sorted(unknown)}"
    assert not any(value for key, value in options.items() if key not in enabled_keys), (
        f"Unsupported enabled optimizations: {options}"
    )


def phase_sequence(protocol):
    """First generations stay separate; warm-ups never enter measured statistics."""
    phases = []
    for phase, key in [("first", "first_runs"), ("warmup", "warmup_runs"), ("measured", "measured_runs")]:
        count = protocol[key]
        assert type(count) is int and count >= 0, f"{key} must be a nonnegative integer"
        phases.extend([phase] * count)
    assert protocol["first_runs"] == 1, "The first_run summary requires exactly one first generation"
    assert protocol["measured_runs"] > 0, "At least one measured generation is required"
    return phases


def run_fields(steps):
    return (
        ["run_index", "phase", "wall_ms", "gpu_span_ms", "text_encode_ms", "denoise_ms"]
        + [f"denoise_step_{i}_ms" for i in range(steps)]
        + ["vae_decode_ms", "postprocess_ms", "other_ms", "peak_alloc_gib", "peak_reserved_gib",
           "device_used_peak_gib", "host_rss_gib", "image_sha256"]
    )


def summarize_runs(rows):
    """Return the shared summary fields without changing engine-specific metrics."""
    assert rows and rows[0]["phase"] == "first", "Missing first generation"
    measured = [row for row in rows if row["phase"] == "measured"]
    assert measured, "Missing measured generations"
    metrics = [key for key in rows[0] if key.endswith("_ms") or key.endswith("_gib")]
    return {
        "measured_runs": len(measured),
        "first_run": {key: rows[0][key] for key in metrics},
        "measured": {key: summarize([row[key] for row in measured]) for key in metrics},
        "deterministic_output": len({row["image_sha256"] for row in measured}) == 1,
    }


def create_run_directory(cfg, out_root, kind):
    """Create a fresh artifact directory; refuse to overwrite an existing run."""
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = out_root / f"{cfg['id']}__{kind}__{stamp}"
    run_dir.mkdir(parents=True)
    write_json(run_dir / "config.json", cfg)
    print(f"Run directory: {run_dir}", flush=True)
    return run_dir


def write_csv(path, rows):
    """Write a completed table. Streaming runners flush rows in their own loop."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@contextmanager
def run_status(run_dir):
    """Record failures, including interrupts, without deleting partial results."""
    status = {"status": "running", "started_utc": now_utc()}
    write_json(run_dir / "status.json", status)
    try:
        yield status
    except BaseException as error:
        status.update(status="failed", finished_utc=now_utc(), error=repr(error),
                      traceback=traceback.format_exc())
        write_json(run_dir / "status.json", status)
        print(f"FAILED; partial results kept in {run_dir}", file=sys.stderr)
        raise
    else:
        status.update(status="complete", finished_utc=now_utc())
        write_json(run_dir / "status.json", status)


@contextmanager
def sampling(sampler):
    """Stop the device monitor even if loading, generation, or profiling fails."""
    sampler.start()
    try:
        yield
    finally:
        sampler.stop()
