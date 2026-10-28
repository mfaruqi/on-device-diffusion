"""MPS instrumentation must retain CFG accounting and honest missing peaks."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from mps_device import MPSStageRecorder
from run_flux import result_row


class MPSTests(unittest.TestCase):
    def test_events_without_inner_sync_or_invented_peaks(self):
        class Event:
            def __init__(self, enable_timing): self.enabled=enable_timing
            def record(self): pass
            def elapsed_time(self, other): return 12.5
        rec=MPSStageRecorder(SimpleNamespace(mps=SimpleNamespace(Event=Event,current_allocated_memory=lambda:0)))
        rec.start('text_encode');rec.end()
        rows,alloc,reserved,segments=rec.finish()
        self.assertEqual(rows[0]['gpu_ms'],12.5)
        self.assertIsNone(alloc);self.assertIsNone(reserved)
        self.assertIsNone(rows[0]['peak_alloc_gib'])
        self.assertEqual(segments,[])

    def test_cfg_two_passes_sum_per_logical_step(self):
        stages=[{'stage':name,'gpu_ms':1} for name in
                ['text_encode','text_encode','denoise_step_0','denoise_step_0','vae_decode','postprocess']]
        result=dict(stages=stages,calls_per_step=2,scheduler_steps=1,wall_ms=8,gpu_span_ms=7,
                    peak_alloc_gib=None,peak_reserved_gib=None,t0=0,t1=1)
        row=result_row(0,'first',SimpleNamespace(tobytes=lambda:b'image'),result,1,
                       SimpleNamespace(peak_between=lambda a,b:None),
                       SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=0)))
        self.assertEqual(row['denoise_ms'],2)
        self.assertEqual(row['text_encode_ms'],2)
        self.assertIsNone(row['peak_alloc_gib'])
        result['scheduler_steps']=2
        with self.assertRaises(AssertionError):
            result_row(0,'first',None,result,1,None,None)
