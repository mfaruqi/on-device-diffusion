"""CPU-only roofline math, immutable-source and provenance regression tests."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import plot_roofline as roof


class RooflineMathTests(unittest.TestCase):
    def test_gemm_precision_bytes(self):
        self.assertEqual(roof.gemm_cost(2, 3, 4), (48, 52))
        self.assertEqual(roof.gemm_cost(2, 3, 4, 2, .5625), (48, 34.75))
        self.assertEqual(roof.gemm_cost(2, 3, 4, 2, .5), (48, 34))
        self.assertAlmostEqual(roof.gemm_cost(2, 3, 4)[0]/roof.gemm_cost(2, 3, 4)[1], 48/52)

    def test_attention_non_square_and_batch(self):
        f, b = roof.attention_cost(2, 3, 5, 7, 11)
        self.assertEqual(f, 4*2*3*5*7*11)
        self.assertEqual(b, 2*2*3*(2*5*11+2*7*11))
        f, b = roof.attention_cost(1, 24, 4608, 4608, 128)
        self.assertEqual(f/b, 2304)

    def test_transformer_operator_sum(self):
        image, text, d = 1024, 512, 3072
        s = image+text
        # 13D² linear weights applied per token in each of 25 blocks.
        linear = 2*s*13*d*d*25 + 2*image*128*d*2 + 2*text*7680*d
        attention = 4*24*s*s*128*25
        f, b = roof.transformer_cost(512, 512, 2)
        self.assertEqual(f, linear+attention)
        self.assertGreater(b, 2*2**30)
        f2, b2 = roof.transformer_cost(512, 512, 3)
        self.assertEqual(f, f2)
        self.assertEqual(b2-b, 2**30)
        with self.assertRaises(ValueError):
            roof.transformer_cost(513, 512, 2)

    def test_invalid_math(self):
        for bad in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                roof.gemm_cost(bad, 3, 4)
            with self.assertRaises(ValueError):
                roof.attention_cost(1, 2, 3, 4, 5, bad)

    def test_cache_executed_passes(self):
        c = {'workload': {'steps': 50, 'cfg_scale': 4}, 'optimizations': {'step_cache': {'mode': 'easycache'}}}
        rows = [{'run_index': '2'}, {'run_index': '3'}]
        self.assertEqual(roof.executed_passes(c, {'cache_audit': {'steps_skipped': [0, 0, 23, 25]}}, rows), [54, 50])
        with self.assertRaises(ValueError):
            roof.executed_passes(c, {}, rows)
        c['optimizations']['step_cache'] = None
        self.assertEqual(roof.executed_passes(c, {}, rows), [100, 100])
        c['workload'] = {'steps': 4, 'cfg_scale': 1}
        self.assertEqual(roof.executed_passes(c, {}, rows), [4, 4])

    def test_hardware_override(self):
        specs = roof.hardware_specs({'jetson-orin-nano': {'bw_gbs': 68, 'ceilings': {'fp16_dense': 10}}})
        self.assertEqual(specs['jetson-orin-nano']['bw_gbs'], 68)
        self.assertEqual(roof.HARDWARE_SPECS['jetson-orin-nano']['bw_gbs'], 102)
        with self.assertRaises(ValueError):
            roof.hardware_specs({'macbook-air-m5': {'bw_gbs': 0}})


class RooflineSchemaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.name = 'fixture__baseline__20261012-000000'
        self.run = self.root/'results'/'runs'/self.name
        self.run.mkdir(parents=True)
        config = {'id': 'fixture', 'device_label': 'gilbreth-a100-40gb', 'precision': 'bfloat16',
                  'model': {'repo_id': 'black-forest-labs/FLUX.2-klein-4B'},
                  'workload': {'width': 1024, 'height': 1024, 'steps': 4, 'cfg_scale': 1},
                  'protocol': {'measured_runs': 10}, 'optimizations': {}}
        for name, data in [('config.json', config), ('status.json', {'status': 'complete'}),
                           ('environment.json', {'gpu_name': 'NVIDIA A100-PCIE-40GB'}),
                           ('summary.json', {'measured_runs': 10, 'weights_gib': {'transformer': 7.2},
                                             'measured': {'denoise_ms': {'median': 104.5}}})]:
            roof.write_json(self.run/name, data)
        with (self.run/'runs.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['run_index', 'phase', 'denoise_ms'])
            writer.writeheader()
            writer.writerows([{'run_index': 0, 'phase': 'first', 'denoise_ms': 9000}]+
                            [{'run_index': i+1, 'phase': 'measured', 'denoise_ms': 100+i} for i in range(10)])
        (self.run/'stages.csv').write_text('run_index,phase,stage,host_ms\n')
        self.manifest = {'runs': [{'run_dir': self.name}]}

    def test_schema_and_source_immutability(self):
        before = {p: p.read_bytes() for p in self.run.rglob('*') if p.is_file()}
        data = roof.collect(self.root, self.manifest)
        roof.validate_data(data)
        self.assertEqual(len(data['points']), 1)
        p = data['points'][0]
        self.assertEqual(p['passes'], 4)
        self.assertEqual(p['duration_ms'], 104.5)
        self.assertAlmostEqual(p['achieved_tflops'], p['flops']/104.5/1e9)
        self.assertEqual(set(p['source_sha256']), set(roof.REQUIRED+['environment.json']))
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)
            self.assertEqual(p['source_sha256'][path.name], hashlib.sha256(content).hexdigest())
        json.dumps(data, allow_nan=False)
        changed = copy.deepcopy(data)
        changed['points'][0]['roof_pct'] *= 2
        with self.assertRaises(ValueError):
            roof.validate_data(changed)

    def test_incomplete_and_mismatched_summary_rejected(self):
        roof.write_json(self.run/'status.json', {'status': 'failed'})
        with self.assertRaises(ValueError):
            roof.collect(self.root, self.manifest)
        self.assertFalse(roof.collect(self.root)['points'])
        roof.write_json(self.run/'status.json', {'status': 'complete'})
        summary = json.loads((self.run/'summary.json').read_text())
        summary['measured']['denoise_ms']['median'] = 999
        roof.write_json(self.run/'summary.json', summary)
        with self.assertRaises(ValueError):
            roof.collect(self.root, self.manifest)

    def test_profile_shapes_and_inferred_attention(self):
        profile = self.run/'profile'
        profile.mkdir()
        f, _ = roof.gemm_cost(2, 3, 4)
        stages = {'denoise_step_0': {'gemm_shapes': {'M2 N3 K4': {'calls': 2, 'ms': 1, 'tflops': f*2/1e9}},
                                   'attention_shapes': {}, 'by_group': {'attention': {'ms': 10}},
                                   'by_leaf_op': {'flash_attn_ext_f16': {'ms': 10, 'kernels': 25}}}}
        roof.write_json(profile/'denoise_kernels.json', {'stages': stages})
        roof.write_json(profile/'stage_kernels.json', {'stages': stages})
        data = roof.collect(self.root, self.manifest)
        self.assertEqual(len(data['points']), 3)  # no duplicate parallel export
        roof.validate_data(data)
        op = next(p for p in data['points'] if p['tier'] == 'operator')
        self.assertEqual(op['analytic_bytes'], 52)
        self.assertEqual(op['duration_ms'], .5)
        self.assertEqual(op['source_file'], 'profile/stage_kernels.json')
        self.assertIn('profile/denoise_kernels.json', op['source_sha256'])
        inferred = next(p for p in data['points'] if p['tier'] == 'operator-inferred')
        self.assertEqual(inferred['duration_ms'], .4)

    def test_sxm_not_mislabeled(self):
        roof.write_json(self.run/'environment.json', {'gpu_name': 'NVIDIA A100-SXM4-40GB'})
        with self.assertRaises(ValueError):
            roof.collect(self.root, self.manifest)
        self.assertIn('PCIe', roof.collect(self.root)['exclusions'][0]['reason'])

    def test_no_output_in_sources_or_ancestor(self):
        for output in [self.run/'charts', self.root/'results', self.root/'originals'/'plots']:
            with self.assertRaises(ValueError):
                roof.guard_output(self.root, output)
        self.assertEqual(roof.guard_output(self.root, self.root/'output'), (self.root/'output').resolve())
        with self.assertRaises(ValueError):
            roof.collect(self.root, {'runs': [{'run_dir': '../escape'}]})


if __name__ == '__main__':
    unittest.main()
