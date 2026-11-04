"""CPU-only checks for the portable PyTorch runner's config validation. No torch import.

Run from the repository root: python3 -m unittest tests.test_flux_portable -v
"""

import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_flux_portable import check_config

CONFIG = ROOT / "configs" / "jetson-torch-flux-klein-q4-nf4-512.json"


def config(**changes):
    cfg = copy.deepcopy(json.loads(CONFIG.read_text()))
    for key, value in changes.items():
        if key in cfg["optimizations"]:
            cfg["optimizations"][key] = value
        else:
            cfg[key] = value
    return cfg


TRT = {"name": "tensorrt", "precision": "bfloat16", "require_full_compilation": False}


class PortableConfigTest(unittest.TestCase):
    def test_jetson_t1_config_is_valid(self):
        check_config(config())

    def test_tensorrt_allowed_on_cuda_backends(self):
        for backend in ("cuda", "jetson"):
            check_config(config(backend=backend, compiler=TRT, stage_residency=None))

    def test_tensorrt_refused_on_mps(self):
        with self.assertRaisesRegex(AssertionError, "not available on backend 'mps'"):
            check_config(config(backend="mps", compiler=TRT))

    def test_tensorrt_must_state_full_compilation(self):
        trt = {k: v for k, v in TRT.items() if k != "require_full_compilation"}
        with self.assertRaises((AssertionError, KeyError)):
            check_config(config(backend="cuda", compiler=trt, stage_residency=None))

    def test_unknown_backend_refused(self):
        with self.assertRaisesRegex(AssertionError, "backend must be one of"):
            check_config(config(backend="rocm"))

    def test_unknown_text_encoder_method_refused(self):
        q = copy.deepcopy(config()["optimizations"]["quantization"])
        q["text_encoder"] = {"method": "gptq"}
        with self.assertRaisesRegex(AssertionError, "text encoder method"):
            check_config(config(quantization=q))

    def test_gguf_requires_hash(self):
        q = copy.deepcopy(config()["optimizations"]["quantization"])
        q["transformer"]["sha256"] = ""
        with self.assertRaisesRegex(AssertionError, "sha256"):
            check_config(config(quantization=q))

    def test_legacy_torch_compile_flag_refused(self):
        with self.assertRaisesRegex(AssertionError, "use 'compiler'"):
            check_config(config(torch_compile=True))

    def test_jetson_needs_tegrastats_interval(self):
        cfg = config()
        del cfg["protocol"]["tegrastats_interval_ms"]
        with self.assertRaisesRegex(AssertionError, "tegrastats_interval_ms"):
            check_config(cfg)

    def test_release_after_encode_needs_prequantized_path(self):
        q = copy.deepcopy(config()["optimizations"]["quantization"])
        q["text_encoder"] = {"method": "bnb_nf4"}
        with self.assertRaisesRegex(AssertionError, "pre-quantized encoder"):
            check_config(config(quantization=q))

    def test_unknown_residency_policy_refused(self):
        with self.assertRaisesRegex(AssertionError, "unsupported stage_residency"):
            check_config(config(stage_residency={"transformer": "release_after_denoise"}))

    def test_compiler_refused_with_transformer_reload(self):
        with self.assertRaisesRegex(AssertionError, "compile only together with conditioning_reuse"):
            check_config(config(compiler={"name": "torch_compile", "mode": "default"}))

    def test_compiler_allowed_with_conditioning_reuse(self):
        cfg = config(compiler={"name": "torch_compile", "mode": "default"})
        cfg["optimizations"]["conditioning_reuse"] = True
        check_config(cfg)

    def test_conditioning_reuse_must_be_boolean(self):
        cfg = config()
        cfg["optimizations"]["conditioning_reuse"] = "yes"
        with self.assertRaisesRegex(AssertionError, "conditioning_reuse must be true or false"):
            check_config(cfg)

    def test_step_cache_parameters_are_explicit(self):
        cfg = config()
        cfg["optimizations"]["step_cache"] = {"method": "taylorseer", "cache_interval": 2,
                                              "disable_cache_before_step": 1, "max_order": 1}
        check_config(cfg)
        cfg["optimizations"]["step_cache"] = {"method": "taylorseer"}
        with self.assertRaisesRegex(AssertionError, "needs exactly"):
            check_config(cfg)
        cfg["optimizations"]["step_cache"] = {"method": "magcache"}
        with self.assertRaisesRegex(AssertionError, "unsupported step_cache"):
            check_config(cfg)

    def test_attention_backend_and_tiling_options(self):
        cfg = config()
        cfg["optimizations"]["attention_backend"] = "math"
        cfg["optimizations"]["vae_tiling"] = {"tile_sample_px": 128, "overlap": 0.25}
        check_config(cfg)
        cfg["optimizations"]["attention_backend"] = "cudnn"
        with self.assertRaisesRegex(AssertionError, "attention_backend must be one of"):
            check_config(cfg)


if __name__ == "__main__":
    unittest.main()
