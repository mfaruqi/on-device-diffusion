"""Save and load complete transformer weights with an upstream BCQ subset.

The package contains BF16 fallback tensors plus packed BCQ files. Loading still
reconstructs dense parameters; this format is not a packed GPU execution backend.
"""

import argparse
import hashlib
import json
import shutil
import traceback
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .upstream_loader import apply_upstream_artifacts


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def package(config):
    root = Path(config["output"])
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "package-config.json", config)
    write_json(root / "status.json", {"status": "running"})
    try:
        artifacts = Path(config["artifacts"])
        fit = json.loads((artifacts / "config.json").read_text())
        summary = json.loads((artifacts / "summary.json").read_text())
        if json.loads((artifacts / "status.json").read_text())["status"] != "completed":
            raise ValueError("Require completed upstream artifacts")
        source_path = Path(fit["source"])
        if digest(source_path) != config["source_sha256"]:
            raise ValueError("Source checkpoint hash mismatch")
        entries = summary["storage"]
        selected = {r["module"] + ".weight" for r in entries}
        if not selected or len(selected) != len(entries):
            raise ValueError("Empty or duplicate packed selection")
        if selected != {m + ".weight" for m in fit["modules"]}:
            raise ValueError("Fit configuration and storage selection disagree")
        manifest = {
            "format": "upstream-bcq-with-bf16-remainder-v1",
            "source_sha256": config["source_sha256"],
            "source_revision": fit["source_revision"],
            "tensors": {},
            "files": {},
        }
        with safe_open(source_path, framework="pt") as source:
            names = set(source.keys())
            if not selected <= names:
                raise ValueError("Packed tensors absent from source")
            fallback = {}
            for name in sorted(names):
                tensor = source.get_tensor(name)
                if tensor.dtype != torch.bfloat16:
                    raise ValueError("This package requires BF16 source tensors")
                manifest["tensors"][name] = {
                    "shape": list(tensor.shape),
                    "source_dtype": "bfloat16",
                    "representation": "upstream_bcq" if name in selected else "bf16",
                }
                if name not in selected:
                    fallback[name] = tensor
            for row in entries:
                filename = row["module"] + ".npz"
                src = artifacts / filename
                if digest(src) != row["artifact_sha256"]:
                    raise ValueError("Packed artifact hash mismatch")
                shape = manifest["tensors"][row["module"] + ".weight"]["shape"]
                with np.load(src, allow_pickle=False) as data:
                    planes = data["planes"]
                    if (
                        len(shape) != 2
                        or planes.shape[1] != shape[0]
                        or planes.shape[2] * planes.shape[3] * 8 != shape[1]
                        or not np.array_equal(data["output_row_indices"], np.arange(shape[0]))
                    ):
                        raise ValueError("Packed artifact does not cover complete source tensor")
                shutil.copyfile(src, root / filename)
            save_file(fallback, root / "fallback.safetensors")
            with safe_open(root / "fallback.safetensors", framework="pt") as saved:
                if set(saved.keys()) != names - selected:
                    raise ValueError("Fallback coverage mismatch")
                for name in saved.keys():
                    if not torch.equal(saved.get_tensor(name), source.get_tensor(name)):
                        raise ValueError("Fallback tensor differs from source")
        # Preserve the decoder's format metadata; source paths are provenance only.
        for filename in ("config.json", "summary.json"):
            shutil.copyfile(artifacts / filename, root / filename)
        if "architecture_config" in config:
            architecture = Path(config["architecture_config"])
            if digest(architecture) != config["architecture_config_sha256"]:
                raise ValueError("Architecture configuration hash mismatch")
            shutil.copyfile(architecture, root / "transformer-config.json")
        for path in sorted(root.iterdir()):
            if path.name not in ("status.json", "package-config.json"):
                manifest["files"][path.name] = {
                    "sha256": digest(path),
                    "bytes": path.stat().st_size,
                }
        write_json(root / "manifest.json", manifest)
        write_json(root / "status.json", {"status": "completed"})
        weight_bytes = sum(
            row["bytes"]
            for name, row in manifest["files"].items()
            if name.endswith((".npz", ".safetensors"))
        )
        result = {
            "scope": "saved transformer weight package; no runtime memory or latency claim",
            "tensor_count": len(manifest["tensors"]),
            "packed_tensor_count": len(selected),
            "bf16_tensor_count": len(manifest["tensors"]) - len(selected),
            "weight_file_bytes": weight_bytes,
            "package_total_bytes": sum(p.stat().st_size for p in root.iterdir()),
            "source_file_bytes": source_path.stat().st_size,
            "manifest_sha256": digest(root / "manifest.json"),
            "fallback_exact_source_match": True,
        }
        return result
    except BaseException:
        write_json(root / "status.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise


def load_package(transformer, directory, chunk_rows=128, cache_packed_arrays=False):
    """Load into an existing architecture without opening the original checkpoint."""
    root = Path(directory)
    if json.loads((root / "status.json").read_text())["status"] != "completed":
        raise ValueError("Incomplete package")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["format"] != "upstream-bcq-with-bf16-remainder-v1":
        raise ValueError("Unknown package format")
    params = dict(transformer.named_parameters())
    if set(params) != set(manifest["tensors"]):
        raise ValueError("Architecture parameter names differ from package")
    if any(p.device.type == "meta" for p in params.values()):
        raise ValueError("Allocate real model parameters before loading")
    for name, param in params.items():
        if list(param.shape) != manifest["tensors"][name]["shape"]:
            raise ValueError("Architecture parameter shape differs from package")
        if param.dtype != torch.bfloat16:
            raise ValueError("This loader requires BF16 model parameters")
    for filename, row in manifest["files"].items():
        if Path(filename).name != filename:
            raise ValueError("Package file names must be relative basenames")
        path = root / filename
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("Package file integrity mismatch")
    selected = {
        name for name, row in manifest["tensors"].items() if row["representation"] == "upstream_bcq"
    }
    with safe_open(root / "fallback.safetensors", framework="pt") as fallback:
        if set(fallback.keys()) != set(params) - selected:
            raise ValueError("Fallback tensor coverage mismatch")
        with torch.no_grad():
            for name in fallback.keys():
                tensor = fallback.get_tensor(name)
                if tensor.dtype != params[name].dtype or tensor.shape != params[name].shape:
                    raise ValueError("Fallback tensor shape or dtype mismatch")
                params[name].copy_(tensor)
    audit = apply_upstream_artifacts(
        transformer,
        root,
        names=selected,
        chunk_rows=chunk_rows,
        cache_packed_arrays=cache_packed_arrays,
    )
    return {
        **audit,
        "fallback_tensors": len(params) - len(selected),
        "original_checkpoint_used": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(config["report"])
    if output.exists():
        raise FileExistsError(output)
    result = package(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
