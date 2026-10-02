#!/usr/bin/env python3
"""Review one Jetson NVTX capture while preserving originals.

Read review_profile() for the sequence. Helpers below validate the timeline,
parse system-memory samples, and serialize derived reports independently.
"""

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import struct
import zlib

from analyze_profile import analyze, render_md
from benchlib import write_json
from jetson_profile import validate_callbacks
from measurement import write_csv

ACTIVITY_TABLES = ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET")
LIMITATIONS = [
    "One profiled generation; not baseline medians.",
    "Stage assignment uses GPU start within CPU NVTX range; no events cross the reviewed stage ends.",
    "Busy time is union of captured GPU activities, not SM utilization; its complement is not proven disk time.",
    "System RAM samples include startup; no reliable wall-clock-to-stage mapping. Printed MB treated as MiB, consistent with existing Jetson metric convention.",
    "GPU timestamps and CPU API time overlap; do not sum across those reports.",
]


def review_profile(run_dir, output_dir):
    originals = run_dir / "originals"
    if output_dir.resolve().is_relative_to(originals.resolve()):
        raise ValueError("Derived outputs must not be written inside originals/")
    config = json.loads((originals / "config.json").read_text())
    steps = config["workload"]["steps"]
    rows = [json.loads(line) for line in (originals / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 2, "Expected one load and one generation"
    load = validate_callbacks(rows, steps)
    generation = next(row for row in rows if row["event"] == "generate")
    ranges, activities = validate_timeline(originals / "trace.sqlite", steps)
    memory = parse_tegrastats((originals / "tegrastats.log").read_text())
    segments = parse_segments((originals / "sdcpp.log").read_text())
    report = build_review_report(ranges, activities, memory, segments)

    output_dir.mkdir(parents=True, exist_ok=True)
    copy_metadata(originals, output_dir)
    write_kernel_reports(originals / "trace.sqlite", output_dir / "profile")
    write_csv(output_dir / "tegrastats-samples.csv", memory)
    write_json(output_dir / "trace-review.json", report)
    save_rgb_image(originals / "raw/run-0.rgb", generation, output_dir / "output.png")
    save_review_summary(originals, output_dir, report, load, generation)
    return report


def validate_timeline(path, steps):
    """Validate CPU stage windows before assigning GPU work by its start timestamp."""
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        assert connection.execute("pragma integrity_check").fetchone()[0] == "ok"
        ranges = connection.execute(
            "SELECT start,end,coalesce(text,s.value) FROM NVTX_EVENTS n "
            "LEFT JOIN StringIds s ON n.textId=s.id WHERE end IS NOT NULL ORDER BY start"
        ).fetchall()
        expected = ["load", "generate", "text_encode"] + [f"denoise_step_{i}" for i in range(steps)] + ["vae_decode"]
        assert [name for start, end, name in ranges] == expected, "Unexpected NVTX stage sequence"
        windows = {name: (start, end) for start, end, name in ranges}
        generation_start, generation_end = windows["generate"]
        children = [row for row in ranges if row[2] not in ("load", "generate")]
        assert all(generation_start <= start < end <= generation_end for start, end, name in children)
        assert all(left[1] <= right[0] for left, right in zip(children, children[1:])), "Stages overlap"
        activities = []
        for table in ACTIVITY_TABLES:
            activities.extend((table, start, end) for start, end in connection.execute("select start,end from " + table))
    crossing = [(kind, start, end, name) for kind, start, end in activities
                for stage_start, stage_end, name in children if stage_start <= start < stage_end < end]
    assert not crossing, "GPU events cross stage ends; inspect before attributing"
    return ranges, activities


def parse_tegrastats(text):
    """Retain line provenance; these system samples have no stage-clock mapping."""
    samples = []
    for number, line in enumerate(text.splitlines(), 1):
        match = re.search(r"RAM (\d+)/(\d+)MB .*?SWAP (\d+)/(\d+)MB", line)
        assert match, line
        ram, ram_total, swap, swap_total = map(int, match.groups())
        samples.append({
            "source_line": number, "timestamp_local": line[:19],
            "ram_reported_mb": ram, "ram_total_reported_mb": ram_total,
            "swap_reported_mb": swap, "ram_gib": ram / 1024, "swap_gib": swap / 1024,
        })
    assert samples, "No tegrastats samples"
    return samples


def parse_segments(log):
    return {name: int(count) for name, count in re.findall(
        r"(qwen3|flux|vae) compute buffer size: .*?peak across (\d+) segments?", log
    )}


def build_review_report(ranges, activities, memory, segments):
    return {
        "source": "originals/trace.sqlite and originals/results.jsonl",
        "sqlite_integrity": "ok",
        "nvtx_ranges": [{"name": name, "start_ns": start, "end_ns": end, "duration_ms": (end - start) / 1e6}
                        for start, end, name in ranges],
        "gpu_activity_counts": {kind: sum(activity[0] == kind for activity in activities)
                                for kind in sorted({activity[0] for activity in activities})},
        "events_crossing_stage_ends": [],  # validate_timeline rejects captures with crossings.
        "segments_reported": segments,
        "monitor_window": {
            "samples": len(memory), "ram_peak_gib": max(sample["ram_gib"] for sample in memory),
            "ram_total_gib": memory[0]["ram_total_reported_mb"] / 1024,
            "swap_min_gib": min(sample["swap_gib"] for sample in memory),
            "swap_max_gib": max(sample["swap_gib"] for sample in memory),
        },
        "limitations": LIMITATIONS,
    }


def copy_metadata(originals, output_dir):
    for name in ["config.json", "environment.json", "command.json", "results.jsonl", "runs.csv", "stages.csv"]:
        shutil.copyfile(originals / name, output_dir / name)
    manifest = {}
    for path in sorted(originals.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            manifest[str(path.relative_to(originals))] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    write_json(output_dir / "artifact-manifest.json", manifest)


def write_kernel_reports(trace, profile_dir):
    profile_dir.mkdir(exist_ok=True)
    for name, pattern in [("denoise_kernels", r"denoise_step_\d+"), ("other_stages", r"text_encode|vae_decode")]:
        result = analyze(trace, pattern, 0)
        write_json(profile_dir / (name + ".json"), result)
        (profile_dir / (name + ".md")).write_text(render_md(result))


def save_rgb_image(raw_path, generation, output):
    """Lossless serialization of existing RGB bytes; this is not model postprocessing."""
    raw = raw_path.read_bytes()
    width, height, channels = generation["width"], generation["height"], generation["channels"]
    assert channels == 3 and len(raw) == width * height * channels

    def png_chunk(tag, data):
        return struct.pack("!I", len(data)) + tag + data + struct.pack("!I", zlib.crc32(tag + data) & 0xffffffff)

    rows = b"".join(b"\0" + raw[y * width * 3:(y + 1) * width * 3] for y in range(height))
    png = (b"\x89PNG\r\n\x1a\n"
           + png_chunk(b"IHDR", struct.pack("!2I5B", width, height, 8, 2, 0, 0, 0))
           + png_chunk(b"IDAT", zlib.compress(rows)) + png_chunk(b"IEND", b""))
    output.write_bytes(png)


def save_review_summary(originals, output_dir, report, load, generation):
    summary = json.loads((originals / "summary.json").read_text())
    summary["trace_timeline_review_pending"] = False
    summary["unavailable_metrics"]["gpu_stage_time"] = (
        "Per-stage captured GPU activity available in profile/*.json; not a clean GPU timing benchmark."
    )
    summary["monitor_window"] = report["monitor_window"]
    summary["segments_reported"] = report["segments_reported"]
    summary["profile_host_diagnostics"] = {
        "load_seconds": load["t_end"] - load["t_start"],
        "generation_seconds": generation["t_end"] - generation["t_start"],
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "load.json", {
        "load_total_s": load["t_end"] - load["t_start"],
        "scope": "Profiled new_sd_ctx; SHA256 pre-read warmed files",
    })
    write_json(output_dir / "status.json", {
        "status": "complete", "trace_timeline_review_pending": False, "evidence": "trace-review.json",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, help="optional separate destination for derived files")
    args = parser.parse_args()
    report = review_profile(args.run_dir, args.out_dir or args.run_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
