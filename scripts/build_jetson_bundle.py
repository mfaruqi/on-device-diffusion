#!/usr/bin/env python3
"""Package Jetson profile/baseline entry points and their local helper modules."""

import argparse
import gzip
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "configs/jetson-flux-klein-q4-512-cache-control-smoke.json",
    "configs/jetson-flux-klein-q4-512-cache-control.json",
    "configs/jetson-flux-klein-q4-512-conditioning-cache-smoke.json",
    "configs/jetson-flux-klein-q4-512-conditioning-cache.json",
    "configs/jetson-flux-klein-q4-512-easycache-smoke.json",
    "configs/jetson-flux-klein-q4-512-easycache.json",
    "configs/jetson-flux-klein-q4-512-mmap-smoke.json",
    "configs/jetson-flux-klein-q4-512-mmap.json",
    "configs/jetson-flux-klein-q4-512-no-prefetch-smoke.json",
    "configs/jetson-flux-klein-q4-512-no-prefetch.json",

    "engines/sdcpp/bench.cpp",
    "engines/sdcpp/CMakeLists.txt",
    "configs/jetson-flux-klein-stage-profile.json",
    "scripts/profile_jetson_stages.py",
    "scripts/jetson_profile.py",
    "scripts/jetson_device.py",
    "scripts/measurement.py",
    "scripts/measurement_events.py",
    "scripts/benchlib.py",
    "scripts/run_jetson_sdcpp.py",
    "scripts/sdcpp_engine.py",
    "configs/jetson-flux-klein-q4-512-disk-baseline.json",
    "configs/jetson-flux-klein-q4-512-disk-smoke.json",
    "configs/jetson-flux-klein-q4-512-disk-profile.json",
    "configs/jetson-flux-klein-q4-512-disk-profile-full.json",
    "configs/jetson-flux-klein-q4-512-autofit.json",
    "configs/jetson-flux-klein-q4-512-autofit-smoke.json",
    "configs/jetson-flux-klein-q4-512-disk-lazy.json",
    "configs/jetson-flux-klein-q4-512-disk-lazy-smoke.json",
    "configs/jetson-flux-klein-base-q4-512-disk.json",
    "configs/jetson-flux-klein-base-q4-512-disk-smoke.json",
    "configs/jetson-flux-klein-base-q4-512-easycache.json",
    "configs/jetson-flux-klein-base-q4-512-easycache-smoke.json",
    "configs/jetson-flux-klein-base-q4-512-mmap.json",
    "configs/jetson-flux-klein-base-q4-512-mmap-smoke.json",
    "configs/jetson-flux-klein-base-q4-512-conditioning-cache.json",
    "configs/jetson-flux-klein-base-q4-512-conditioning-cache-smoke.json",
    "configs/jetson-flux-klein-base-q4-512-no-prefetch.json",
    "configs/jetson-flux-klein-base-q4-512-no-prefetch-smoke.json",
)


def build_bundle(output):
    """Fixed metadata makes rebuilds reproducible and excludes machine-specific data."""
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for name in FILES:
                    path = ROOT / name
                    info = archive.gettarinfo(str(path), arcname=name)
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ""
                    info.pax_headers = {}
                    with path.open("rb") as source:
                        archive.addfile(info, source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "bundles/jetson-stage-profile.tar.gz")
    args = parser.parse_args()
    build_bundle(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
