"""Trace regression tests use small synthetic GPU timelines, never GPU execution."""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import struct
import sys
import tempfile
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from analyze_profile import analyze, render_md
from profile_metrics import union_ms
from profile_readers import load_nsys
from review_jetson_profile import parse_tegrastats, review_profile, validate_timeline

EVIDENCE = ROOT / 'results/runs/jetson-flux-klein-003__profile__20260928-195625/originals'


def make_nsys(path, generations=1):
    """Two overlapping kernels and a separate copy in each 10-us NVTX stage."""
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.executescript('''
            CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, text TEXT, textId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER, end INTEGER, shortName INTEGER, demangledName INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER, end INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET(start INTEGER, end INTEGER);
            INSERT INTO StringIds VALUES (1, 'gemm_fixture');
            INSERT INTO NVTX_EVENTS VALUES (0, 1000, 'load', NULL);
        ''')
        for generation in range(generations):
            offset = generation * 100000
            connection.execute('INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, NULL)', (offset + 1000, offset + 90000, 'generate'))
            names = ['text_encode'] + [f'denoise_step_{i}' for i in range(4)] + ['vae_decode']
            for index, name in enumerate(names):
                start = offset + 2000 + index * 12000
                connection.execute('INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, NULL)', (start, start + 10000, name))
                for begin, end in [(1000, 3000), (2000, 4000)]:
                    connection.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 1, 1)', (start + begin, start + end))
                connection.execute('INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?)', (start + 5000, start + 6000))


def make_torch(path):
    events = [
        dict(cat='gpu_user_annotation', name='denoise_step_0', ts=100, dur=100),
        dict(cat='cpu_op', name='aten::linear', tid=1, ts=10, dur=70, args={}),
        dict(cat='cpu_op', name='aten::mm', tid=1, ts=20, dur=40,
             args={'Input Dims': [[2, 3], [3, 4]]}),
        dict(cat='cuda_runtime', name='cudaLaunchKernel', tid=1, ts=30, dur=2, args={'correlation': 7}),
        dict(cat='kernel', name='gemm_fixture', ts=110, dur=20, args={'correlation': 7}),
        dict(cat='gpu_memcpy', name='copy', ts=120, dur=20, args={}),
        dict(cat='gpu_memset', name='zero', ts=160, dur=5, args={}),
        dict(cat='kernel', name='outside_stage', ts=210, dur=5, args={}),
    ]
    path.write_text(json.dumps({'traceEvents': events}))


class ProfileAnalysisTests(unittest.TestCase):
    def test_saved_reports_render_identically(self):
        checked = 0
        for path in (ROOT / 'results/runs').glob('*/profile/*.json'):
            markdown = path.with_suffix('.md')
            if not markdown.exists():
                continue
            saved = json.loads(path.read_text())
            if 'stage_regex' not in saved:
                continue
            with self.subTest(report=str(path)):
                self.assertEqual(render_md(saved), markdown.read_text())
                checked += 1
        self.assertGreaterEqual(checked, 5)

    def test_torch_correlation_shapes_and_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            trace = Path(temp) / 'trace.json'
            make_torch(trace)
            stage = analyze(trace, r'denoise_step_\d+')['stages']['denoise_step_0']
            self.assertEqual(stage['n_kernels'], 3)
            self.assertAlmostEqual(stage['busy_ms'], 0.035)
            self.assertAlmostEqual(stage['kernel_sum_ms'], 0.045)
            self.assertAlmostEqual(stage['idle_ms'], 0.065)
            self.assertEqual(stage['by_top_op']['aten::linear'], {'ms': .02, 'kernels': 1})
            self.assertEqual(stage['by_leaf_op']['aten::mm'], {'ms': .02, 'kernels': 1})
            self.assertEqual(stage['gemm_shapes']['M2 N4 K3']['calls'], 1)
            self.assertAlmostEqual(stage['gemm_shapes']['M2 N4 K3']['tflops'], 48 / .00002 / 1e12)

    def test_nsys_units_generation_and_input_immutability(self):
        with tempfile.TemporaryDirectory() as temp:
            trace = Path(temp) / 'trace.sqlite'
            make_nsys(trace, generations=2)
            before = trace.read_bytes()
            for generation in [0, -1]:
                result = analyze(trace, r'denoise_step_\d+', generation)
                self.assertEqual(result['generation'], generation)
                self.assertEqual(len(result['stages']), 4)
                for stage in result['stages'].values():
                    self.assertAlmostEqual(stage['span_ms'], .01)
                    self.assertAlmostEqual(stage['busy_ms'], .004)
                    self.assertAlmostEqual(stage['kernel_sum_ms'], .005)
                    self.assertEqual(stage['n_kernels'], 3)
                    self.assertEqual(stage['gemm_shapes'], {})
            first = load_nsys(trace, 'denoise_step_0', 0)
            last = load_nsys(trace, 'denoise_step_0', -1)
            self.assertEqual(last['denoise_step_0']['acts'][0]['ts'] - first['denoise_step_0']['acts'][0]['ts'], 100)
            self.assertEqual(trace.read_bytes(), before)

    def test_missing_sqlite_is_not_created(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'missing.sqlite'
            with self.assertRaises(sqlite3.OperationalError):
                load_nsys(path, 'denoise_step_0')
            self.assertFalse(path.exists())

    def test_union_handles_nested_and_adjacent_intervals(self):
        self.assertEqual(union_ms([]), 0)
        self.assertEqual(union_ms([(0, 1000), (200, 300), (1000, 1500), (2000, 2500)]), 2)

    def test_jetson_rejects_crossing_gpu_activity(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'trace.sqlite'
            make_nsys(path)
            validate_timeline(path, 4)
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute('INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (11000,13000)')
            with self.assertRaisesRegex(AssertionError, 'cross stage ends'):
                validate_timeline(path, 4)

    def test_tegrastats_preserves_system_scope_and_line_provenance(self):
        text = '09-28-2026 19:56:23 RAM 4096/8192MB (lfb 1x4MB) SWAP 1024/4096MB\n'
        sample = parse_tegrastats(text)[0]
        self.assertEqual(sample['ram_gib'], 4)
        self.assertEqual(sample['swap_gib'], 1)
        self.assertEqual(sample['source_line'], 1)
        self.assertNotIn('stage', sample)
        with self.assertRaises(AssertionError):
            parse_tegrastats('')

    def test_complete_jetson_review_keeps_originals_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / 'run'
            originals = run / 'originals'
            originals.mkdir(parents=True)
            for name in ['config.json', 'environment.json', 'command.json', 'results.jsonl', 'runs.csv', 'stages.csv', 'summary.json']:
                (originals / name).write_bytes((EVIDENCE / name).read_bytes())
            rows = [json.loads(line) for line in (originals / 'results.jsonl').read_text().splitlines()]
            rows[1].update(width=2, height=1)
            (originals / 'results.jsonl').write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
            make_nsys(originals / 'trace.sqlite')
            (originals / 'tegrastats.log').write_text('09-28-2026 19:56:23 RAM 4096/8192MB (lfb 1x4MB) SWAP 1024/4096MB\n')
            (originals / 'sdcpp.log').write_text('flux compute buffer size: 128 MB (peak across 3 segments)\n')
            (originals / 'raw').mkdir()
            raw = bytes([1, 2, 3, 4, 5, 6])
            (originals / 'raw/run-0.rgb').write_bytes(raw)
            before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in originals.rglob('*') if path.is_file()}
            output = Path(temp) / 'derived'
            report = review_profile(run, output)
            self.assertEqual(report['segments_reported'], {'flux': 3})
            self.assertEqual(report['monitor_window']['ram_peak_gib'], 4)
            self.assertEqual(json.loads((output / 'status.json').read_text())['trace_timeline_review_pending'], False)
            self.assertIsNone(json.loads((output / 'summary.json').read_text())['measured'])
            png = (output / 'output.png').read_bytes()
            self.assertTrue(png.startswith(b'\x89PNG\r\n\x1a\n'))
            offset = 8
            pixels = b''
            while offset < len(png):
                length = struct.unpack('!I', png[offset:offset + 4])[0]
                if png[offset + 4:offset + 8] == b'IDAT':
                    pixels += png[offset + 8:offset + 8 + length]
                offset += length + 12
            self.assertEqual(zlib.decompress(pixels), b'\0' + raw)
            for path, digest in before.items():
                self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), digest)
            with self.assertRaisesRegex(ValueError, 'inside originals'):
                review_profile(run, originals / 'derived')


if __name__ == '__main__':
    unittest.main()
