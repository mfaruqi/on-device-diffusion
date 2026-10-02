#!/usr/bin/env python3
"""Package the Jetson capture and its complete standard-library dependency set."""

import argparse
import gzip
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "engines/sdcpp/bench.cpp",
    "engines/sdcpp/CMakeLists.txt",
    "configs/jetson-flux-klein-stage-profile.json",
    "scripts/profile_jetson_stages.py",
    "scripts/jetson_profile.py",
    "scripts/jetson_device.py",
    "scripts/measurement.py",
    "scripts/measurement_events.py",
    "scripts/benchlib.py",
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
