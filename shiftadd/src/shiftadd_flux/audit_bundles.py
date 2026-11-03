"""Read-only bundle integrity and logical storage inventory."""

import argparse
import hashlib
import json
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--bundles", nargs="+", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    source = Path(a.source)
    source_bytes = source.stat().st_size
    report = {
        "source_bytes": source_bytes,
        "source_sha256": file_hash(source),
        "scope": "logical deployment weight file sizes, not live inference memory or workspace disk usage",
        "bundles": {},
    }
    expected_names = None
    for directory in a.bundles:
        root = Path(directory)
        if json.loads((root / "status.json").read_text())["status"] != "completed":
            raise ValueError("Incomplete bundle")
        summary = json.loads((root / "summary.json").read_text())
        rows = summary["results"]
        names = {r["tensor"] for r in rows}
        if len(rows) != 109 or len(names) != 109:
            raise ValueError("Expected 109 unique matrix tensors")
        if expected_names is None:
            expected_names = names
        if names != expected_names:
            raise ValueError("Mismatched matrix coverage")
        size = 0
        for row in rows:
            suffix = (
                ".npz" if summary.get("format") == "symmetric_uniform" else f".p{row['planes']}.npz"
            )
            file = root / (row["tensor"] + suffix)
            if (
                file.stat().st_size != row["artifact_bytes"]
                or file_hash(file) != row["artifact_sha256"]
            ):
                raise ValueError(f"Artifact integrity mismatch: {file}")
            size += file.stat().st_size
        unchanged = root / "unchanged.safetensors"
        size += unchanged.stat().st_size
        report["bundles"][root.name] = {
            "matrices": 109,
            "weight_file_bytes": size,
            "gib": size / 2**30,
            "reduction_vs_source": 1 - size / source_bytes,
            "unchanged_sha256": file_hash(unchanged),
            "matrix_hashes_verified": True,
        }
    with Path(a.output).open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
