#!/usr/bin/env python3
"""Normalize existing measurements into event records, after timed work finishes.

This is an evidence adapter, not new instrumentation. It preserves source clocks
and unknowns, and never infers disk writes or a release destination from a free.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

from benchlib import write_json

SCHEMA_VERSION = 1


def event(kind, name, source, *, start=None, end=None, duration=None,
          clock=None, timing=None, run_index=None, phase=None, details=None):
    return {
        "schema_version": SCHEMA_VERSION, "event": kind, "name": name,
        "run_index": run_index, "phase": phase,
        "start_s": start, "end_s": end, "duration_ms": duration,
        "clock": clock, "timing_method": timing,
        "source": source, "details": details or {},
    }


def read_csv(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return list(csv.DictReader(stream))


def pytorch_events(run_dir):
    """CSV stage durations have no absolute timestamps; do not fabricate a timeline."""
    records, sources = [], []
    path = run_dir / "load.json"
    if path.exists():
        load = json.loads(path.read_text())
        records.append(event("load", "pipeline", {"file": "load.json", "key": "load_total_s"},
                             duration=load["load_total_s"] * 1000, timing="host_wall_duration"))
        sources.append(path)
    for filename, kind, name_key, duration_key, timing in [
        ("runs.csv", "generation", None, "wall_ms", "host_wall_duration"),
        ("stages.csv", "stage", "stage", "gpu_ms", "cuda_event_duration"),
    ]:
        path = run_dir / filename
        if not path.exists():
            continue
        sources.append(path)
        for line, row in enumerate(read_csv(path), 2):
            records.append(event(kind, row[name_key] if name_key else "generate",
                                 {"file": filename, "line": line},
                                 duration=float(row[duration_key]) if row.get(duration_key) else None,
                                 timing=timing, run_index=int(row["run_index"]), phase=row["phase"],
                                 details={"host_ms": float(row["host_ms"])} if row.get("host_ms") else None))
    return records, sources


def harness_events(path, relative_name, phases):
    """Use exact host callback boundaries, including each individual denoising step."""
    records = []
    for line, text in enumerate(path.read_text().splitlines(), 1):
        if not text.strip():
            continue
        row = json.loads(text)
        kind = row["event"]
        if kind not in ("load", "generate"):
            continue
        index = row.get("run_index")
        phase = phases.get(index)
        source = {"file": relative_name, "line": line}

        def interval(event_kind, name, start, end, details=None):
            return event(event_kind, name, source, start=start, end=end,
                         duration=(end - start) * 1000 if start is not None and end is not None else None,
                         clock="sdcpp_steady_clock" if start is not None or end is not None else None,
                         timing="host_callback_interval", run_index=index, phase=phase, details=details)

        records.append(interval("load" if kind == "load" else "generation", "pipeline" if kind == "load" else "generate",
                                row.get("t_start"), row.get("t_end"), {"ok": row["ok"]}))
        if kind != "generate" or not row["ok"]:
            continue
        records.append(interval("stage", "text_encode", row["t_start"], row["t_cond"]))
        previous = row["t_sampling_start"]
        for step, end in enumerate(row["t_step_end"]):
            records.append(interval("stage", f"denoise_step_{step}", previous, end))
            previous = end
        records.append(interval("stage", "vae_decode", row["t_sampling_end"], row["t_decode_end"]))
    return records


def residency_events(log, source_name):
    """Keep engine-reported buffer actions and rounded load diagnostics with line evidence."""
    records = []
    for line_number, text in enumerate(log.splitlines(), 1):
        stamp = re.match(r"^(\d+\.\d+) \[", text)
        timestamp = float(stamp[1]) if stamp else None
        source = {"file": source_name, "line": line_number}
        buffer = re.search(r"model manager (prepared|released) params backend buffers \(\s*([\d.]+) MB, "
                           r"(\d+) tensors, (\d+) blocks, ([^)]+)\) (?:on|from) (\S+)", text)
        if buffer:
            action, amount, tensors, blocks, storage, backend = buffer.groups()
            records.append(event("parameters_prepared" if action == "prepared" else "parameters_released",
                                 "params_backend_buffers", source, start=timestamp, end=timestamp,
                                 clock="sdcpp_steady_clock" if stamp else None, timing="log_observation",
                                 details={"reported_mb": float(amount), "reported_unit": "engine_printed_MB",
                                          "tensors": int(tensors), "blocks": int(blocks),
                                          "storage_label": storage, "backend": backend,
                                          "component": None, "destination": None, "raw": text}))
        load = re.search(r"loading tensors completed, taking ([\d.]+)s", text)
        if load:
            records.append(event("tensor_load_complete", "tensors", source, end=timestamp,
                                 duration=float(load[1]) * 1000,
                                 clock="sdcpp_steady_clock" if stamp else None,
                                 timing="rounded_engine_diagnostic", details={"raw": text}))
    return records


def sdcpp_events(run_dir):
    records, sources = [], []
    harness = run_dir / "engine/results.jsonl"
    if not harness.exists():
        harness = run_dir / "results.jsonl"
    phases = {int(row["run_index"]): row["phase"] for row in read_csv(run_dir / "runs.csv") if "phase" in row}
    # A Jetson profile has header-only CSVs. Its phase is declared in the saved config.
    config_path = run_dir / "config.json"
    if not phases and config_path.exists():
        config = json.loads(config_path.read_text())
        if config.get("protocol", {}).get("attempted_generations") == 1 and "profile" in config.get("record_kind", ""):
            phases = {0: "profile"}
    if harness.exists():
        sources.append(harness)
        records.extend(harness_events(harness, str(harness.relative_to(run_dir)), phases))
    for relative_name in ["engine/sdcpp.log", "sdcpp.log", "generation.log"]:
        path = run_dir / relative_name
        if path.exists():
            sources.append(path)
            records.extend(residency_events(path.read_text(errors="replace"), relative_name))
            break  # These are alternative raw log locations, not independent observations.
    return records, sources


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_events(run_dir, engine, output_dir=None):
    """New runs write beside their artifacts; replay callers should use --out-dir."""
    output_dir = output_dir or run_dir
    if output_dir.resolve().is_relative_to((run_dir / "originals").resolve()):
        raise ValueError("Event exports must not be written inside originals/")
    if engine == "pytorch":
        records, sources = pytorch_events(run_dir)
    elif engine == "sdcpp":
        records, sources = sdcpp_events(run_dir)
    else:
        raise ValueError(f"Unknown engine: {engine}")
    if not sources:
        raise ValueError("No supported measurement sources found")
    # Hash all consulted inputs, including phase labels and profile configuration.
    sources = sorted(set(sources + [path for path in [run_dir / "config.json", run_dir / "runs.csv"] if path.exists()]))
    limitations = [
        "Derived after execution; event export adds no instrumentation inside measured regions.",
        "Clocks are local to their source process/run; do not align different runs or Nsight clocks by raw value.",
        "Records are grouped by source, not a globally sorted timeline. Null timestamps mean unavailable.",
        "Buffer release is not evidence of a write to disk or a transfer to CPU; destination and component remain unknown.",
        "Absence of matching log messages does not establish absence of releases; physical disk I/O is not measured.",
    ]
    if engine == "pytorch":
        limitations.append("PyTorch CSVs retain durations, not absolute stage/load timestamps or weight-release observations.")
    metadata = {
        "schema_version": SCHEMA_VERSION, "engine": engine, "event_count": len(records),
        "parameter_release_observations": sum(row["event"] == "parameters_released" for row in records),
        "source_sha256": {str(path.relative_to(run_dir)): fingerprint(path) for path in sources},
        "limitations": limitations,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    write_json(output_dir / "events-metadata.json", metadata)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--engine", required=True, choices=["pytorch", "sdcpp"])
    parser.add_argument("--out-dir", required=True, type=Path, help="separate destination; preserve existing run evidence")
    args = parser.parse_args()
    if args.out_dir.resolve() == args.run_dir.resolve():
        parser.error("Use a separate --out-dir when replaying existing artifacts")
    records = export_events(args.run_dir, args.engine, args.out_dir)
    print(f"Wrote {len(records)} observations to {args.out_dir}")


if __name__ == "__main__":
    main()
