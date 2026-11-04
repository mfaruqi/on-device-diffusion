"""CPU-only admission tests: unsupported targets/precision must not load an engine."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import run_nunchaku as runner


class NunchakuAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT/'configs/a100-nunchaku-flux-klein-int4-r32-smoke.json').read_text())

    def test_pinned_smoke_and_full_configs(self):
        runner.check_config(self.cfg)
        full = json.loads((ROOT/'configs/a100-nunchaku-flux-klein-int4-r32.json').read_text())
        runner.check_config(full)
        self.assertEqual(len(runner.phase_sequence(full['protocol'])), 14)

    def test_rejects_fallbacks_and_unpinned_inputs(self):
        for mutate in [
            lambda c: c['optimizations']['quantization'].update(weights_bits=1),
            lambda c: c['optimizations'].update(cpu_offload=True),
            lambda c: c['quantized_transformer'].update(revision='main'),
            lambda c: c['engine']['wheel'].update(sha256='missing'),
            lambda c: c['model'].update(repo_id='black-forest-labs/FLUX.2-klein-base-4B'),
        ]:
            cfg = copy.deepcopy(self.cfg)
            mutate(cfg)
            with self.assertRaises(AssertionError):
                runner.check_config(cfg)

    def test_only_nonpersistent_qwen_rope_buffers_may_be_fp32(self):
        rope = dict(name="rotary_emb.inv_freq", kind="buffer", leaf_name="inv_freq",
                    owner_class="Qwen3RotaryEmbedding", persistent=False,
                    dtype="torch.float32", floating=True)
        runner.validate_component_dtypes("text_encoder", [rope])
        for change in ({"kind": "parameter"}, {"persistent": True},
                       {"owner_class": "Linear"}, {"leaf_name": "weight"},
                       {"dtype": "torch.float16"}):
            with self.assertRaises(AssertionError):
                runner.validate_component_dtypes("text_encoder", [dict(rope, **change)])
        with self.assertRaises(AssertionError):
            runner.validate_component_dtypes("vae", [rope])

    def test_unsupported_device_stops_before_engine_import_or_output(self):
        for device in ['jetson', 'm5']:
            result = subprocess.run([sys.executable, str(ROOT/'scripts/run_nunchaku.py'),
                                     '--device', device], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn(runner.SUPPORT[device]['reason'], result.stderr)
            self.assertNotIn('Run directory:', result.stdout)
            report = subprocess.check_output([sys.executable, str(ROOT/'scripts/run_nunchaku.py'),
                                              '--device', device, '--check-support'], text=True)
            self.assertEqual(json.loads(report)['status'], 'blocked')


if __name__ == '__main__':
    unittest.main()
