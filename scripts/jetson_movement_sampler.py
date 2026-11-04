#!/usr/bin/env python3
"""Whole-system data-movement counters on the Jetson Orin Nano, sampled beside a run.

  sudo python3 scripts/jetson_movement_sampler.py --out DIR [--interval-ms 100] -- <command...>

Starts the sampler, runs the command, stops the sampler and writes into DIR:
  samples.csv      one row per interval: wall-clock time (time.time_ns) and raw cumulative counters
                   - storage: sectors read/written for every whole block device in /proc/diskstats
                   - page cache / swap: /proc/vmstat pgpgin, pgpgout, pswpin, pswpout, pgmajfault
                   - memory: /proc/meminfo MemAvailable, Cached, Mapped, SwapFree (kB)
  tegrastats.log   `tegrastats --interval N` with a wall-clock prefix per line (as root it reports
                   EMC_FREQ <busy%>@<MHz>, the memory-controller load used for the DRAM estimate)
  sampler.json     interval, devices, command, exit code, root status and the sysfs files that were found
Counters are left raw; deltas and units are computed in analysis. Nothing here changes the command's
execution beyond the sampler's own CPU time (one read of three /proc files per interval).
Swap on this board is zram (compressed RAM), so pswpin/pswpout are memory traffic, not storage reads.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time

VMSTAT = ('pgpgin', 'pgpgout', 'pswpin', 'pswpout', 'pgmajfault')
MEMINFO = ('MemAvailable', 'Cached', 'Mapped', 'SwapFree')


def whole_disks():
    """Block devices that are not partitions (nvme0n1, mmcblk0, sda, zram0)."""
    names = []
    for line in Path('/proc/diskstats').read_text().splitlines():
        name = line.split()[2]
        if Path(f'/sys/block/{name}').exists() and not name.startswith(('loop', 'ram')):
            names.append(name)
    return names


def read_counters(disks):
    row = {}
    for line in Path('/proc/diskstats').read_text().splitlines():
        f = line.split()
        if f[2] in disks:
            row[f'{f[2]}_sectors_read'] = int(f[5])
            row[f'{f[2]}_sectors_written'] = int(f[9])
    vm = dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
    row.update({k: int(vm[k]) for k in VMSTAT})
    mem = {m[1]: int(m[2]) for m in re.finditer(r'^(\w+):\s+(\d+)', Path('/proc/meminfo').read_text(), re.M)}
    row.update({f'{k}_kb': mem[k] for k in MEMINFO})
    return row


def sample(out, interval_s, disks, stop):
    fields = ['t_wall_ns'] + list(read_counters(disks))
    with (out / 'samples.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        next_t = time.monotonic()
        while not stop.is_set():
            writer.writerow({'t_wall_ns': time.time_ns(), **read_counters(disks)})
            stream.flush()
            next_t += interval_s
            stop.wait(max(0.0, next_t - time.monotonic()))
        writer.writerow({'t_wall_ns': time.time_ns(), **read_counters(disks)})


def tegrastats(out, interval_ms, stop):
    proc = subprocess.Popen(['tegrastats', '--interval', str(interval_ms)], stdout=subprocess.PIPE, text=True)
    with (out / 'tegrastats.log').open('w') as stream:
        for line in proc.stdout:
            stream.write(f'{time.time_ns()} {line}')
            stream.flush()
            if stop.is_set():
                break
    proc.terminate()
    proc.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--interval-ms', type=int, default=100)
    ap.add_argument('command', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    assert command, 'give the command to run after --'
    args.out.mkdir(parents=True, exist_ok=False)
    disks = whole_disks()
    stop = threading.Event()
    threads = [threading.Thread(target=sample, args=(args.out, args.interval_ms / 1000, disks, stop)),
               threading.Thread(target=tegrastats, args=(args.out, args.interval_ms, stop))]
    for t in threads:
        t.start()
    started = time.time_ns()
    try:
        rc = subprocess.run(command).returncode
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=10)
    sysfs = [p for p in ('/sys/kernel/actmon_avg_activity/mc_all', '/sys/kernel/debug/bpmp/debug/clk/emc/rate')
             if os.path.exists(p)]
    (args.out / 'sampler.json').write_text(json.dumps({
        'interval_ms': args.interval_ms, 'disks': disks, 'command': command, 'returncode': rc,
        'root': os.geteuid() == 0, 'started_wall_ns': started, 'finished_wall_ns': time.time_ns(),
        'sysfs_found': sysfs, 'swap_note': 'zram swap: pswpin/pswpout are compressed-RAM traffic'}, indent=2))
    raise SystemExit(rc)


if __name__ == '__main__':
    main()
