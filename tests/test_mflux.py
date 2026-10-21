"""CPU contract tests: unsafe configurations and mismatched gates fail before loading."""
import copy
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from mflux_engine import check_config, validate_run

ROOT=Path(__file__).resolve().parents[1]


class MfluxTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((ROOT/'configs/m5-mflux-flux-klein-bf16-512.json').read_text())

    def test_six_configurations_and_smokes(self):
        files=list((ROOT/'configs').glob('m5-mflux-*.json'))
        for path in files:
            check_config(json.loads(path.read_text()))
        self.assertGreaterEqual(len(files),12)

    def test_rejects_fallbacks_and_unpinned_sources(self):
        for key,value in [('precision','float16'),('backend','cpu')]:
            cfg=copy.deepcopy(self.cfg);cfg[key]=value
            with self.assertRaises(AssertionError):check_config(cfg)
        for key,value in [('offload',True),('conditioning_reuse','yes'),('vae_tiling',True),('quantization',3),('unreviewed',True)]:
            cfg=copy.deepcopy(self.cfg);cfg['optimizations'][key]=value
            with self.assertRaises(AssertionError):check_config(cfg)
        cfg=copy.deepcopy(self.cfg);cfg['model']['revision']='main'
        with self.assertRaises(AssertionError):check_config(cfg)

    def test_gate_rejects_other_precision(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=copy.deepcopy(self.cfg);cfg['optimizations']['quantization']=4
            (Path(d)/'config.json').write_text(json.dumps(cfg))
            with self.assertRaisesRegex(AssertionError,'optimizations'):
                validate_run(d,self.cfg)

    def test_individual_options_reject_combination(self):
        cfg=copy.deepcopy(self.cfg)
        cfg['optimizations']['conditioning_reuse']=True
        cfg['optimizations']['vae_tiling']={'tile_size_pixels':256,'overlap_pixels':64}
        with self.assertRaisesRegex(AssertionError,'Individual options'):
            check_config(cfg)

    def test_gate_rejects_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'config.json').write_text(json.dumps(self.cfg))
            (Path(d)/'status.json').write_text('{"status":"running"}')
            with self.assertRaises(AssertionError):validate_run(d,self.cfg)


if __name__=='__main__':unittest.main()
