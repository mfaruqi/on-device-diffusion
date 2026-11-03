import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from shiftadd_flux.format import decode, encode
from shiftadd_flux.loader import apply_artifacts


class LoaderTest(unittest.TestCase):
    def test_saved_artifact_applied_and_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = torch.nn.Linear(128, 2, bias=False)
            p, e = encode(model.weight.detach().numpy())
            artifact = root / "weight.p4.npz"
            np.savez(artifact, planes=p, exponents=e)
            row = {
                "tensor": "weight",
                "shape": [2, 128],
                "planes": 4,
                "artifact_bytes": artifact.stat().st_size,
                "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
            (root / "summary.json").write_text(json.dumps({"results": [row]}))
            (root / "status.json").write_text(json.dumps({"status": "completed"}))
            apply_artifacts(model, root, chunk_rows=1)
            np.testing.assert_array_equal(model.weight.detach().numpy(), decode(p, e))
            before = model.weight.detach().clone()
            with artifact.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaises(ValueError):
                apply_artifacts(model, root)
            torch.testing.assert_close(model.weight, before)


if __name__ == "__main__":
    unittest.main()


class UniformLoaderTest(unittest.TestCase):
    def test_chunked_uniform_loader_and_integrity(self):
        from shiftadd_flux.conventional import decode_uniform, encode_uniform
        from shiftadd_flux.loader import apply_uniform_artifacts

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = torch.nn.Linear(128, 5, bias=False)
            codes, scales = encode_uniform(model.weight.detach().numpy(), 4)
            artifact = root / "weight.npz"
            np.savez(artifact, codes=codes, scales=scales, shape=np.array([5, 128]))
            row = {
                "tensor": "weight",
                "shape": [5, 128],
                "artifact_bytes": artifact.stat().st_size,
                "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
            (root / "summary.json").write_text(
                json.dumps({"format": "symmetric_uniform", "bits": 4, "results": [row]})
            )
            (root / "status.json").write_text(json.dumps({"status": "completed"}))
            apply_uniform_artifacts(model, root, chunk_rows=2)
            np.testing.assert_array_equal(
                model.weight.detach().numpy(),
                decode_uniform(codes, scales, (5, 128), 4),
            )
            before = model.weight.detach().clone()
            with artifact.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaises(ValueError):
                apply_uniform_artifacts(model, root)
            torch.testing.assert_close(model.weight, before)
