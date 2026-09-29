"""Trace-format adapters: Chrome/PyTorch and Nsight SQLite to stage activities.

All returned timestamps and durations use microseconds, even though Nsight stores
nanoseconds. SQLite inputs are opened read-only and closed before returning.
"""

import bisect
import collections
from contextlib import closing
import json
from pathlib import Path
import re

from profile_metrics import categorize

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


def build_op_chains(events):
    """Map each cuda_runtime event (by correlation id) to its enclosing cpu_op chain."""
    by_tid = collections.defaultdict(list)
    for e in events:
        if e.get("cat") == "cpu_op":
            by_tid[e["tid"]].append(("op", e["ts"], -e["dur"], e))
        elif e.get("cat") == "cuda_runtime" and "correlation" in e.get("args", {}):
            by_tid[e["tid"]].append(("rt", e["ts"], 0, e))
    chains = {}
    for items in by_tid.values():
        # ops sort before a launch at the same ts; longer ops first so nesting is outer->inner
        items.sort(key=lambda x: (x[1], 0 if x[0] == "op" else 1, x[2]))
        stack = []
        for kind, ts, _, e in items:
            while stack and stack[-1]["ts"] + stack[-1]["dur"] < ts:
                stack.pop()
            if kind == "op":
                stack.append(e)
            else:
                chains[e["args"]["correlation"]] = list(stack)
    return chains


def load_torch_trace(trace_path, stage_re):
    """Stage windows and GPU activities from a torch.profiler Chrome trace (run_flux.py --profile)."""
    events = json.loads(Path(trace_path).read_text())["traceEvents"]
    windows = sorted(
        (e["ts"], e["ts"] + e["dur"], e["name"])
        for e in events
        if e.get("cat") == "gpu_user_annotation" and re.fullmatch(stage_re, e["name"])
    )
    assert windows, f"no gpu_user_annotation matching {stage_re!r} in {trace_path}"
    starts = [w[0] for w in windows]
    chains = build_op_chains(events)

    stages = {name: {"span_ms": (e - s) / 1000.0, "acts": []} for s, e, name in windows}
    for e in events:
        if e.get("cat") not in GPU_CATS:
            continue
        i = bisect.bisect_right(starts, e["ts"]) - 1
        if i < 0 or e["ts"] > windows[i][1]:
            continue
        chain = chains.get(e.get("args", {}).get("correlation"), [])
        aten = [op for op in chain if op["name"].startswith("aten::")]
        top = aten[0]["name"] if aten else "(no aten op)"
        leaf_op = aten[-1] if aten else None
        leaf = leaf_op["name"] if leaf_op else "(no aten op)"
        stages[windows[i][2]]["acts"].append(
            {
                "name": e["name"],
                "cat": e["cat"],
                "ts": e["ts"],
                "dur_us": e["dur"],
                "top": top,
                "leaf": leaf,
                "leaf_op": leaf_op,
                "group": categorize(e["name"], leaf, e["cat"]),
            }
        )
    return stages


def load_nsys(path, stage_re, generation=-1):
    """Stage windows and GPU activities from an Nsight Systems report (run_sdcpp.py --nsys).

    Stage windows are the harness's NVTX ranges. They are CPU-side ranges, but each ends only
    after ggml's synchronous graph compute has finished, so a kernel belongs to the stage whose
    range contains its GPU start time. ggml kernels have no launching aten op, so "top" is the
    kernel group and "leaf" is the kernel's short name.

    The report covers every generation of the run (first, warm-up, measured), each wrapped in a
    `generate` range. Only one is analysed: `generation` indexes those ranges (default -1, the last,
    which is a measured run).
    """
    import sqlite3
    import subprocess

    path = Path(path)
    if path.suffix == ".nsys-rep":
        db = path.with_suffix(".sqlite")
        if not db.exists():
            subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite", "true", "-o", str(db), str(path)],
                           check=True, capture_output=True)
        path = db
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as con:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        nvtx = con.execute(
            "SELECT n.start, n.end, COALESCE(n.text, s.value) FROM NVTX_EVENTS n "
            "LEFT JOIN StringIds s ON n.textId = s.id WHERE n.end IS NOT NULL").fetchall()
        gens = sorted((a / 1000.0, b / 1000.0) for a, b, name in nvtx if name == "generate")
        assert gens, f"no NVTX 'generate' ranges in {path}"
        g0, g1 = gens[generation]
        windows = sorted((a / 1000.0, b / 1000.0, name) for a, b, name in nvtx
                         if name and re.fullmatch(stage_re, name) and g0 <= a / 1000.0 <= g1)
        assert windows, f"no NVTX range matching {stage_re!r} in generation {generation} of {path}"
        assert len({w[2] for w in windows}) == len(windows), "duplicate stage names within one generation"
        starts = [w[0] for w in windows]
        stages = {name: {"span_ms": (e - s) / 1000.0, "acts": []} for s, e, name in windows}

        rows = [("kernel", a, b, short, full) for a, b, short, full in con.execute(
            "SELECT k.start, k.end, s.value, d.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON k.shortName = s.id JOIN StringIds d ON k.demangledName = d.id")]
        for table, cat in (("CUPTI_ACTIVITY_KIND_MEMCPY", "gpu_memcpy"), ("CUPTI_ACTIVITY_KIND_MEMSET", "gpu_memset")):
            if table in tables:
                rows += [(cat, a, b, cat, cat) for a, b in con.execute(f"SELECT start, end FROM {table}")]
        for cat, a, b, short, full in rows:
            ts = a / 1000.0
            i = bisect.bisect_right(starts, ts) - 1
            if i < 0 or ts > windows[i][1]:
                continue
            group = categorize(full, None, cat)
            stages[windows[i][2]]["acts"].append(
                {"name": full, "cat": cat, "ts": ts, "dur_us": (b - a) / 1000.0,
                 "top": group, "leaf": short, "leaf_op": None, "group": group})
        return stages
