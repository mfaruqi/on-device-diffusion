import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from shiftadd_flux.package_upstream import digest, load_package, package, write_json
from shiftadd_flux.upstream_loader import apply_upstream_artifacts


class PackageTests(unittest.TestCase):
    def test_complete_package_and_source_independent_load(self):
        # Keep these tiny diagnostics available for inspection after the test.
        root = Path(tempfile.mkdtemp(prefix="shiftadd-package-test-"))
        source = root / "source.safetensors"
        save_file(
            {
                "proj.weight": torch.zeros(2, 256, dtype=torch.bfloat16),
                "proj.bias": torch.tensor([2, 3], dtype=torch.bfloat16),
            },
            source,
        )
        artifacts = root / "artifacts"
        artifacts.mkdir()
        path = artifacts / "proj.npz"
        np.savez(
            path,
            planes=np.full((1, 2, 2, 16), 255, dtype=np.uint8),
            superblock_scale=np.ones((2, 1), dtype=np.float16),
            scale_codes=np.ones((2, 1, 2, 1), dtype=np.uint8),
            output_row_indices=np.arange(2),
        )
        write_json(
            artifacts / "config.json",
            {
                "source": str(source),
                "source_revision": "synthetic",
                "modules": ["proj"],
                "group_size": 128,
                "scale_format": "u8",
            },
        )
        write_json(
            artifacts / "summary.json",
            {
                "storage": [
                    {
                        "module": "proj",
                        "artifact_sha256": digest(path),
                        "file_bytes": path.stat().st_size,
                    }
                ]
            },
        )
        write_json(artifacts / "status.json", {"status": "completed"})
        bundle = root / "bundle"
        result = package(
            {"artifacts": str(artifacts), "output": str(bundle), "source_sha256": digest(source)}
        )
        self.assertEqual(result["tensor_count"], 2)
        self.assertEqual(result["packed_tensor_count"], 1)
        # Make the provenance path unusable without deleting the retained source.
        config_path = bundle / "config.json"
        config = json.loads(config_path.read_text())
        config["source"] = str(root / "does-not-exist")
        write_json(config_path, config)
        manifest_path = bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["config.json"] = {
            "bytes": config_path.stat().st_size,
            "sha256": digest(config_path),
        }
        write_json(manifest_path, manifest)
        model = torch.nn.Module()
        model.proj = torch.nn.Linear(256, 2, dtype=torch.bfloat16)
        audit = load_package(model, bundle, chunk_rows=1)
        self.assertFalse(audit["original_checkpoint_used"])
        self.assertTrue(torch.equal(model.proj.weight, torch.ones_like(model.proj.weight)))
        self.assertTrue(torch.equal(model.proj.bias, torch.tensor([2, 3], dtype=torch.bfloat16)))
        before = model.proj.weight.detach().clone()
        cached_audit = apply_upstream_artifacts(
            model, bundle, chunk_rows=1, cache_packed_arrays=True
        )
        self.assertTrue(cached_audit["cache_packed_arrays"])
        self.assertTrue(torch.equal(before, model.proj.weight))
        with (bundle / "proj.npz").open("ab") as stream:
            stream.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "integrity"):
            load_package(model, bundle)
        self.assertTrue(torch.equal(before, model.proj.weight))


if __name__ == "__main__":
    unittest.main()
